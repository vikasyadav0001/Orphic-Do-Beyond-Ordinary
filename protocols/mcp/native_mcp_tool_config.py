"""
LangGraph + LangChain MCP Adapters — REMOTE-ONLY Server Configuration
======================================================================
Every server here is a hosted HTTP endpoint or brdge remote servers. Zero local processes.
npx, uvx, subprocesses. Just a URL + token.while for notion bridge remote server ussed via npx
Native tool for agent that are connected to the agent a the startup

"""
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import get_settings
import os
import json
from utils.logger import get_logger
import asyncio

settings = get_settings()
logger = get_logger(__name__)

# Retrieve tokens from settings
EXA_API_KEY = settings.exa_api_key
FIRECRAWL_API_KEY =None
N8N_MCP_TOKEN =settings.n8n_mcp_token
BROWSERLESS_API_KEY=None


missing = [k for k, v in {
    "FIRECRAWL_API_KEY": FIRECRAWL_API_KEY,
    "EXA_API_KEY": EXA_API_KEY,
    "N8N_TOKEN": N8N_MCP_TOKEN,
    "BROWSERLESS_API_KEY": BROWSERLESS_API_KEY,
}.items() if not v]

if missing:
    logger.warning(f"Missing MCP tokens: {', '.join(missing)}. Some tools may not work.")

# ── MCP server connection config ──────────────────────────────────────────────
# Defined here as a plain dict — the actual client is built inside the async
# function below, NOT at module level.
MCP_CONFIG = {}

if FIRECRAWL_API_KEY:
    MCP_CONFIG["firecrawl"] = {
        "transport": "http",
        "url": "https://mcp.firecrawl.dev/v2/mcp",
        "headers": {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
    }

if EXA_API_KEY:
    MCP_CONFIG["exa"] = {
        "transport": "http",
        "url": "https://mcp.exa.ai/mcp",
        "headers": {"x-api-key": EXA_API_KEY},
    }



if BROWSERLESS_API_KEY:
    MCP_CONFIG["playwright"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
        "env": {
            "PLAYWRIGHT_CONNECT_URL": f"wss://chrome.browserless.io?token={BROWSERLESS_API_KEY}"
        }
    }

if BROWSERLESS_API_KEY:
    MCP_CONFIG["playwright"] = {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@playwright/mcp@latest"],
        "env": {
            "PLAYWRIGHT_CONNECT_URL": f"wss://chrome.browserless.io?token={BROWSERLESS_API_KEY}"
        }
    }


# ── Native Tool loader ───────────────────────────────────────────────────────────────
async def get_native_mcp_tools():
    """
    Build the MCP client and return all tools from connected servers.
    FIX 6: client is constructed INSIDE the async function, not at module level.
    """
    if not MCP_CONFIG:
        logger.warning("No MCP servers configured")
        return []

    logger.info(f"Connecting to {len(MCP_CONFIG)} MCP server(s)...")

    connected_tools = []
    failed_servers = []

    # Try each server individually to isolate failures
    for server_name, config in MCP_CONFIG.items():
        try:
            logger.info(f"Connecting to {server_name}...")
            client = MultiServerMCPClient({server_name: config})
            tools = await client.get_tools()

            # for t in tools:
            #     t.handle_tool_error = True

            connected_tools.extend(tools)
            logger.info(f"Connected to {server_name}: {len(tools)} tools loaded")
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            failed_servers.append(server_name)

    if failed_servers:
        logger.warning(f"Failed to connect to servers: {', '.join(failed_servers)}")

    logger.info(f"Total MCP tools loaded: {len(connected_tools)}")

    return connected_tools



# ── MCP server connection config for Workflow Agent ──────────────────────────────────────────────
# Defined here as a plain dict — the actual client is built inside the async
# function below, NOT at module level.

WORKFLOW_AGENT_CONFIG = {}

if N8N_MCP_TOKEN:
    WORKFLOW_AGENT_CONFIG["n8n-mcp"] = {
        "transport" : "http",
        "url": "http://localhost:5678/mcp-server/http",
        "headers": {
        "Authorization": f"Bearer {N8N_MCP_TOKEN}"
      }
    }

# ── Tool loader for Workflow Agent ───────────────────────────────────────────────────────────────

async def get_workflow_agent_tools():
    """
    Build the MCP client for workflow agent and return all tools from connected servers.
    """
    if not WORKFLOW_AGENT_CONFIG:
        logger.warning("No MCP servers configured for workflow agent.")
        return []

    logger.info(f"Connecting to {len(WORKFLOW_AGENT_CONFIG)} MCP server(s)...")

    connected_tools = []
    failed_servers = []

    for server_name, config in WORKFLOW_AGENT_CONFIG.items():
        try:
            logger.info(f"Connecting to {server_name}...")
            client = MultiServerMCPClient({server_name: config})
            tools = await client.get_tools()

            connected_tools.extend(tools)
            logger.info(f"Connected to {server_name}: {len(tools)} tools loaded")
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {e}")
            failed_servers.append(server_name)

    if failed_servers:
        logger.warning(f"Failed to connect to servers: {', '.join(failed_servers)}")

    logger.info(f"Total MCP tools loaded for workflow agent: {len(connected_tools)}")

    return connected_tools     


if __name__ == "__main__":
    #asyncio.run(get_native_mcp_tools())
    tools = asyncio.run(get_workflow_agent_tools())
    for tool in tools:
        if tool.name == "get_node_types":
            print(tool)
