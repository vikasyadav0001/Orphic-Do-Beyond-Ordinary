"""
User MCP Tool Node
===================
Graph node that executes user-configured MCP tools via the prebuilt ToolNode.

Flow:
  1. Load user's MCP config from Redis (DB fallback)
  2. Connect to their servers via MultiServerMCPClient
  3. Get live executable tools with prefixed names
  4. Delegate execution to LangGraph's prebuilt ToolNode

This node only receives MCP tool calls — the Send() router in graph.py
filters out fixed provider tools before dispatching here.
"""

import asyncio

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from sqlalchemy import select

from db.models import UserMCPConfig, async_session_maker
from services.cache import (
    get_cached_mcp_config,
    set_cached_mcp_config,
)
from utils.logger import get_logger

logger = get_logger(__name__)

MCP_EXEC_TIMEOUT = 30  # seconds — longer than discovery since tools do real work


async def _load_user_config(user_id: str) -> dict | None:
    """
    Load user MCP config: Redis first, DB fallback.
    Caches in Redis on DB hit for next time.
    """
    # Try Redis
    config = await get_cached_mcp_config(user_id)
    if config:
        logger.info(f"[MCPToolNode] MCP config cache HIT for user {user_id}: servers {list(config.keys())}")
        return config

    # DB fallback
    try:
        async with async_session_maker() as db:
            stmt = select(UserMCPConfig.config_json).where(
                UserMCPConfig.user_id == user_id
            )
            result = await db.execute(stmt)
            row = result.scalar_one_or_none()

        if row:
            logger.info(f"[MCPToolNode] MCP config cache MISS for user {user_id}. Retain from DB: servers {list(row.keys())}")
            await set_cached_mcp_config(user_id, row)
            return row
        else:
            logger.info(f"[MCPToolNode] MCP config DB lookup for user {user_id}: No servers configured")
    except Exception as e:
        logger.error(f"[MCPToolNode] DB fallback failed loading MCP config for user {user_id}: {e}")

    return None


async def mcp_tool_node(state: dict, config: RunnableConfig) -> dict:
    """
    Execute user MCP tool calls using the prebuilt ToolNode.
    
    This node only sees MCP tool calls — fixed provider calls are
    routed to the existing tool_node by the Send() dispatcher.
    """
    user_id = state["user_id"]
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        logger.info(f"[MCPToolNode] No remote MCP tool calls to execute for user {user_id}")
        return {"messages": []}

    tool_names = [tc.get("name") for tc in tool_calls]
    logger.info(f"[MCPToolNode] Remote MCP tool execution START for user {user_id}: {tool_names}")

    # 1. Load user's MCP config
    user_config = await _load_user_config(user_id)
    if not user_config:
        logger.warning(f"[MCPToolNode] No MCP config found for user {user_id}. Returning missing configuration message.")
        return {
            "messages": [
                ToolMessage(
                    content="No MCP servers configured. Add servers in Settings → MCP Config.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    # 2. Filter user_config to only connect to servers required for the active tool call(s)
    requested_servers = set()
    sorted_server_names = sorted(user_config.keys(), key=len, reverse=True)
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        for server_name in sorted_server_names:
            if tool_name.startswith(f"{server_name}_") or tool_name.startswith(f"{server_name}__"):
                requested_servers.add(server_name)
                break
    
    if not requested_servers:
        logger.warning(f"[MCPToolNode] Could not match tool calls to any registered server for user {user_id}: {tool_names}")
        return {
            "messages": [
                ToolMessage(
                    content=f"Could not identify which MCP server owns tool '{tc.get('name')}'. This may be a configuration issue.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    target_servers = {name: user_config[name] for name in requested_servers if name in user_config}
    
    mcp_client_config = {
        name: {
            "transport": server.get("transport", "http"),
            "url": server["url"],
            "headers": server.get("headers", {})
        }
        for name, server in target_servers.items()
    }
    logger.info(f"[MCPToolNode] Connecting to {len(mcp_client_config)} requested MCP server(s) for user {user_id}: {list(mcp_client_config.keys())}")

    # 3. Connect, get live tools, execute via prebuilt ToolNode
    try:
        async with asyncio.timeout(MCP_EXEC_TIMEOUT):
            client = MultiServerMCPClient(mcp_client_config, tool_name_prefix=True)
            tools = await client.get_tools()
            logger.info(f"[MCPToolNode] MultiServerMCPClient connected successfully. Discovered {len(tools)} live tool(s) for user {user_id}")

            tools_by_name = {t.name: t for t in tools}
            results = []
            for tc in tool_calls:
                tname = tc.get("name")
                t_id = tc.get("id")
                clean_args = {k: v for k, v in tc.get("args", {}).items() if v is not None}

                if tname in tools_by_name:
                    try:
                        output = await tools_by_name[tname].ainvoke(clean_args, config=config)
                    except Exception as err:
                        output = f"Tool '{tname}' execution failed: {err}"
                else:
                    output = f"Tool '{tname}' not found on connected remote MCP servers."

                results.append(ToolMessage(content=str(output), tool_call_id=t_id))

            logger.info(f"[MCPToolNode] Remote MCP tool execution finished for user {user_id}: {tool_names}")
            return {"messages": results}

    except asyncio.TimeoutError:
        logger.error(f"[MCPToolNode] Remote MCP tool execution TIMED OUT after {MCP_EXEC_TIMEOUT}s for user {user_id}: {tool_names}")
        return {
            "messages": [
                ToolMessage(
                    content=f"MCP tool execution timed out after {MCP_EXEC_TIMEOUT}s.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    except Exception as e:
        logger.error(f"[MCPToolNode] Remote MCP tool execution FAILED for user {user_id}: {e}", exc_info=True)
        return {
            "messages": [
                ToolMessage(
                    content=f"MCP tool execution failed: {str(e)}",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }
