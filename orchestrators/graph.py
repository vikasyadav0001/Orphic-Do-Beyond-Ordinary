"""
Orphic — Main Agent Graph
==========================
Nodes:  agent_node, tool_node
Edges:  START → agent_node → (tool_node → agent_node)* → END
HITL:   interrupt() in tool_node propagates to compiled graph via GraphInterrupt
Resume: resume_response() sends Command(resume=action) to continue paused graph
"""

import asyncio
import json
from typing import Annotated, Any

from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt, Send
from langchain_core.callbacks.manager import dispatch_custom_event

from config import get_settings
from memory.graph_checkpointer import pool, setup_db
from memory.long_term_memory import retrieve_memory, setup_memory_store
from prompts.system_persona_prompt import get_prompt
from protocols.mcp.remote_mcp_client_config import execute_mcp_tool, load_schemas
from protocols.mcp.user_mcp_tool_node import _load_user_config, mcp_tool_node
from services.cache import (
    get_cached_mcp_config,
    get_cached_mcp_tool_schemas,
    get_mcp_config_version,
    set_cached_mcp_config,
)
from services.mcp_config_service import build_schema_tools, reload_and_cache_user_mcp
from services.cache import (
    cache_token,
    get_cached_token,
    get_redis,
    get_user_connections,
    set_user_connections,
)
from services.connections.oauth_engine import OAuthEngine
from services.connections.providers import PROVIDERS
from services.rate_limiter import check_and_increment
from tools.custom_tools import custom_tools
from utils.agent_helpers import cancel_all, fetch_connections_from_db, resolve_token
from db.models import UserConnection, async_session_maker
from utils.interrupts import read_resume_value
from utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)
oauth_engine = OAuthEngine()

# ── Supported LLMs ───────────────────────────────────────────────────────────────────────

SUPPORTED_MODELS = {

    "gpt-5.6-luna": ChatOpenAI(
        model="gpt-5.6-luna",
        api_key=settings.openai_api_key,
        streaming=True,
        reasoning={
            "effort": "low",
            "summary": "auto"
        }
    ),

    "gpt-5.4-mini": ChatOpenAI(
        model="gpt-5.4-mini",
        api_key=settings.openai_api_key,
        streaming=True,
        reasoning={
            "effort": "low",
            "summary": "auto"
        }
    ),

    "gpt-5-nano": ChatOpenAI(
        model="gpt-5-nano",
        api_key=settings.openai_api_key,
        streaming=True,
        reasoning={
            "effort": "medium",  
            "summary": "auto"
        }
    ),
}

# ── Sync schema load — disk only, no network ──────────────────────────────────

schema_tools, tool_provider_map = load_schemas()
from services.tool_retriever import tool_retriever_service
tool_retriever_service.init_global_fixed_tools(schema_tools, tool_provider_map)

# ── Vector-based tool retriever (semantic search via Qdrant + FastEmbed) ────────
# Runs alongside BM25 — activated when search_tools meta-tool is called by the LLM
try:
    from services.vector_tool_retriever import vector_tool_retriever_service
    vector_tool_retriever_service.init_global_fixed_tools(schema_tools, tool_provider_map)
except Exception as _vtr_err:
    logger.warning(f"[Startup] Vector tool retriever init skipped: {_vtr_err}")
    vector_tool_retriever_service = None

# ── Module-level state ────────────────────────────────────────────────────────

_llm_with_tools = None
_executable_tools: dict = {t.name: t for t in custom_tools}
compiled_graph = None
_init_lock = asyncio.Lock()
_bounded_models: dict[str, Any] = {}

# ─────────────────────────────────────────────────────────────────────────────
# AgentState
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    thread_id: str
    model: str
    summary: str


# ─────────────────────────────────────────────────────────────────────────────
# AgentHelpers
# ─────────────────────────────────────────────────────────────────────────────
def emit_activity(status:str, tool:str, label:str, config: RunnableConfig):
    """Dispatches a custom event that astream_events will catch."""
    dispatch_custom_event(
        "tool_activity",
        {"id": f"task_{tool}", "status":status, "tool": tool, "label":label},
        config=config
    )


def _extract_tool_call_ids(msg) -> set[str]:
    """Extract tool call IDs from both msg.tool_calls and msg.additional_kwargs."""
    ids = set()
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            if isinstance(tc, dict) and "id" in tc:
                ids.add(tc["id"])
            elif hasattr(tc, "id"):
                ids.add(getattr(tc, "id"))
    if hasattr(msg, "additional_kwargs") and isinstance(msg.additional_kwargs, dict):
        tool_calls = msg.additional_kwargs.get("tool_calls", [])
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict) and "id" in tc:
                    ids.add(tc["id"])
    return ids


def _sanitize_messages(messages: list) -> list:
    """Ensure message sequence integrity and optimize token usage across turns.
    
    1. Purges ToolMessages and AIMessage tool_calls (including additional_kwargs)
       from PAST completed turns (messages before the last HumanMessage).
    2. Keeps full fidelity ToolMessages for the CURRENT active turn.
    3. Enforces API sequence integrity (drops unresponded tool calls or orphan ToolMessages).
    """
    if not messages:
        return []

    # Find the index of the last HumanMessage (start of current active turn)
    last_human_idx = -1
    for idx, msg in enumerate(messages):
        if (hasattr(msg, "type") and msg.type == "human") or isinstance(msg, HumanMessage):
            last_human_idx = idx

    pruned = []
    for idx, msg in enumerate(messages):
        is_past_turn = (last_human_idx != -1 and idx < last_human_idx)
        
        if is_past_turn:
            # ── Past Completed Turns ─────────────────────────────────────
            # Strip raw ToolMessages from past turns
            if (hasattr(msg, "type") and msg.type == "tool") or isinstance(msg, ToolMessage):
                continue
            
            # For AIMessages with tool_calls (or additional_kwargs tool_calls) from past turns:
            tool_ids = _extract_tool_call_ids(msg)
            if tool_ids:
                content = getattr(msg, "content", "")
                if content and isinstance(content, str) and content.strip():
                    # Create a clean text AIMessage without tool_calls or additional_kwargs
                    pruned.append(AIMessage(content=content))
                continue
        
        pruned.append(msg)

    # Re-evaluate valid tool_calls vs responses
    tool_call_ids_responded = {
        msg.tool_call_id 
        for msg in pruned 
        if hasattr(msg, "tool_call_id") and msg.tool_call_id
    }
    
    sanitized = []
    for msg in pruned:
        tool_ids = _extract_tool_call_ids(msg)
        if tool_ids:
            all_responded = all(t_id in tool_call_ids_responded for t_id in tool_ids)
            if not all_responded:
                # Unresponded tool calls: if it has text, keep as clean text AIMessage, else drop
                content = getattr(msg, "content", "")
                if content and isinstance(content, str) and content.strip():
                    sanitized.append(AIMessage(content=content))
                continue
        elif hasattr(msg, "tool_call_id") and msg.tool_call_id:
            valid_tc_ids = set()
            for m in pruned:
                valid_tc_ids.update(_extract_tool_call_ids(m))
            if msg.tool_call_id not in valid_tc_ids:
                continue  # drop orphan ToolMessage
        sanitized.append(msg)
    
    return sanitized


# ─────────────────────────────────────────────────────────────────────────────
# agent_node
# ─────────────────────────────────────────────────────────────────────────────


async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    user_id = state["user_id"]
    thread_id = state.get("thread_id", "unknown")
    requested_model = state.get("model", "gpt-5-nano") or config.get("configurable", {}).get("model", "gpt-5-nano")

    logger.info(f"[AgentNode] Turn START | User: {user_id} | Thread: {thread_id} | Model: {requested_model}")

    # 1. Memory — Redis-cached 120s, falls back to pgvector on miss
    memories: list[str] = []
    try:
        redis = await get_redis()
        cached = await redis.get(f"memory:{user_id}")
        if cached:
            memories = json.loads(cached)
            logger.info(f"[AgentNode] Memory cache HIT for user {user_id}: {len(memories)} item(s)")
        else:
            memories = await retrieve_memory(
                user_id=user_id,
                query="user details preferences goals",
            )
            if memories:
                await redis.setex(f"memory:{user_id}", 120, json.dumps(memories))
                logger.info(f"[AgentNode] Memory cache MISS for user {user_id}. Retained {len(memories)} item(s) from pgvector")
            else:
                logger.info(f"[AgentNode] Memory cache MISS for user {user_id}: No stored memory records found")
    except Exception as e:
        logger.error(f"[AgentNode] Memory retrieval failed for user {user_id}: {e}")

    # 2. Connected providers — Redis → DB fallback
    connections: set[str] = set()
    try:
        connections = await get_user_connections(user_id) or set()
        if connections:
            logger.info(f"[AgentNode] OAuth connections cache HIT for user {user_id}: {sorted(connections)}")
        else:
            connections = await fetch_connections_from_db(user_id)
            if connections:
                await set_user_connections(user_id, connections)
                logger.info(f"[AgentNode] OAuth connections cache MISS for user {user_id}. Loaded from DB: {sorted(connections)}")
            else:
                logger.info(f"[AgentNode] OAuth connections for user {user_id}: None connected")
    except Exception as e:
        logger.error(f"[AgentNode] Connection fetch failed for user {user_id}: {e}")

    # 3. Dynamic tool binding & info for user MCPs
    user_mcp_tools = []
    user_mcp_servers: dict[str, int] = {}
    try:
        raw_schemas = None
        version = await get_mcp_config_version(user_id)
        if version:
            raw_schemas = await get_cached_mcp_tool_schemas(user_id)
            if raw_schemas:
                logger.info(f"[AgentNode] MCP schemas cache HIT for user {user_id} (version: {version[:8]}...)")
            else:
                logger.warning(f"[AgentNode] MCP version exists ({version[:8]}...) but schemas expired. Reloading from DB...")
                raw_schemas = await reload_and_cache_user_mcp(user_id)
        else:
            logger.info(f"[AgentNode] MCP version cache MISS for user {user_id}. Checking DB directly...")
            raw_schemas = await reload_and_cache_user_mcp(user_id)

        if raw_schemas:
            user_mcp_tools = build_schema_tools(raw_schemas)
            cached_user_config = await get_cached_mcp_config(user_id)
            if not cached_user_config:
                cached_user_config = await _load_user_config(user_id)
            sorted_server_names = sorted(cached_user_config.keys(), key=len, reverse=True) if cached_user_config else []
            for s in raw_schemas:
                tool_name = s.get("name", "")
                matched_srv = None
                for server_name in sorted_server_names:
                    if tool_name.startswith(f"{server_name}_") or tool_name.startswith(f"{server_name}__"):
                        matched_srv = server_name
                        break
                if not matched_srv:
                    matched_srv = tool_name.split("_", 1)[0] if "_" in tool_name else "mcp"
                user_mcp_servers[matched_srv] = user_mcp_servers.get(matched_srv, 0) + 1

            logger.info(f"[AgentNode] User MCP tools READY for user {user_id}: {len(user_mcp_tools)} tools across servers {user_mcp_servers}")
        else:
            logger.info(f"[AgentNode] User MCP tools for user {user_id}: None configured")
    except Exception as e:
        logger.warning(f"[AgentNode] User MCP tool loading failed for user {user_id}: {e}")

    # 4. System prompt
    try:
        system_prompt = get_prompt(
            user_memory=memories,
            connected_providers=connections,
            available_providers=set(PROVIDERS.keys()) - connections,
            user_mcp_servers=user_mcp_servers,
        )
        # Inject persisted summary as background context
        summary = state.get("summary", "")
        if summary:
            system_prompt += (
                "\n\n<conversation_history_summary>\n"
                "The following is a summary of the earlier conversation. "
                "Use it as background context only. Do NOT repeat, restate, "
                "or reference this summary in your response.\n"
                f"{summary}\n"
                "</conversation_history_summary>"
            )
        logger.info(f"[AgentNode] System prompt generated for user {user_id} (OAuth: {len(connections)}, Remote MCP Servers: {len(user_mcp_servers)})")
    except Exception as e:
        logger.error(f"[AgentNode] get_prompt failed for user {user_id}: {e}")
        system_prompt = "You are Orphic, a helpful AI assistant."

    messages = state["messages"]

    # 5. LLM-Controlled Vector Tool Discovery
    #    No pre-retrieval. The LLM calls search_tools to discover tools,
    #    which stores results in _last_search_results. On agent_node re-entry,
    #    we read those results and dynamically bind the discovered tools.
    sorted_user_servers = list(user_mcp_servers.keys())
    vector_retrieved_tools: list = []
    vector_provider_map: dict = {}
    try:
        if vector_tool_retriever_service is not None:
            # Index user's dynamic MCP tools into Qdrant (no-op if version unchanged)
            if user_mcp_tools:
                vector_tool_retriever_service.index_user_mcp_tools(
                    user_id=user_id,
                    version_hash=version or "v0",
                    user_mcp_tools=user_mcp_tools,
                    server_names=sorted_user_servers,
                )

            # Check if search_tools was called in a previous loop iteration
            from tools.search_tools import get_last_search_results
            searched_tool_names = get_last_search_results(user_id, thread_id)
            if searched_tool_names:
                vector_retrieved_tools, vector_provider_map = (
                    vector_tool_retriever_service.resolve_tools_by_names(
                        tool_names=searched_tool_names,
                        user_id=user_id,
                    )
                )
                # Merge vector provider map into tool_provider_map ONLY for fixed OAuth providers
                for tname, p_name in vector_provider_map.items():
                    if p_name in PROVIDERS:
                        tool_provider_map[tname] = p_name
                logger.info(
                    f"[AgentNode] search_tools resolved {len(vector_retrieved_tools)} tool(s) "
                    f"for dynamic binding: {searched_tool_names}"
                )
    except Exception as e:
        logger.warning(f"[AgentNode] Vector retriever step failed (non-fatal): {e}")

    # 5b. Pass connected_providers through config so search_tools can filter by them
    if "configurable" not in config:
        config["configurable"] = {}
    config["configurable"]["connected_providers"] = list(connections)

    base_llm = SUPPORTED_MODELS.get(requested_model, SUPPORTED_MODELS["gpt-5-nano"])
    # Bind: system/native tools (includes search_tools) + vector-retrieved tools (if any)
    all_tools = list(_executable_tools.values()) + vector_retrieved_tools
    active_llm = base_llm.bind_tools(all_tools)
    logger.info(
        f"[AgentNode] Bound {len(all_tools)} tools "
        f"({len(_executable_tools)} system/native + "
        f"{len(vector_retrieved_tools)} vector-retrieved) to LLM '{requested_model}'"
    )
    
    try:
        clean_messages = _sanitize_messages(messages)
        logger.info(f"[AgentNode] Invoking LLM for user {user_id} with {len(clean_messages)} history message(s)...")
        response = await active_llm.ainvoke(
            [SystemMessage(content=system_prompt)] + clean_messages,
            config=config,
        )
        tool_calls_count = len(getattr(response, "tool_calls", []) or [])
        logger.info(f"[AgentNode] Turn SUCCESS | User: {user_id} | Response generated | Tool calls requested: {tool_calls_count}")
    except Exception as e:
        logger.error(f"[AgentNode] LLM call failed for user {user_id}: {e}", exc_info=True)
        return {
            "messages": [
                AIMessage(content="I encountered an error. Please try again.")
            ]
        }

    return {"messages": [response]}


DESTRUCTIVE_KEYWORDS = {
    "delete", "remove", "destroy", "drop", "purge", "wipe", 
    "erase", "terminate", "uninstall", "unlink", "revoke"
}


def is_destructive_tool(tool_name: str) -> bool:
    name_lower = tool_name.lower()
    return any(keyword in name_lower for keyword in DESTRUCTIVE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# tool_node
# ─────────────────────────────────────────────────────────────────────────────

async def tool_node(state: AgentState, config: RunnableConfig) -> dict:
    """
    Two-phase execution:
      Phase 1 — validate ALL remote tool calls (auth + rate limit + destructive action confirmation) before executing any.
                 Interrupts here have zero side effects — no execution has happened.
      Phase 2 — execute ALL tool calls.
                 GraphInterrupt re-raised explicitly so HITL from custom tools
                 (e.g. create_workflow) propagates to the compiled graph.
    """
    user_id = state["user_id"]
    session_id = state["thread_id"]
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", [])

    if not tool_calls:
        logger.info(f"[ToolNode] No tool calls to execute for user {user_id}")
        return {"messages": []}

    tool_names = [tc.get("name") for tc in tool_calls]
    logger.info(f"[ToolNode] Processing {len(tool_calls)} system/fixed tool call(s) for user {user_id}: {tool_names}")

    # ── Phase 1: validate all remote tool calls ───────────────────────────────

    # 1. Destructive action confirmation interrupt check
    for tc in tool_calls:
        tool_name = tc["name"]
        if is_destructive_tool(tool_name):
            provider = tool_provider_map.get(tool_name)
            provider_label = PROVIDERS[provider].label if provider and provider in PROVIDERS else (provider or "Orphic")
            logger.info(f"[ToolNode] Destructive tool '{tool_name}' detected. Triggering confirmation interrupt for user {user_id}")
            
            resume_val = interrupt({
                "id": f"confirm_{tool_name}_{tc['id']}",
                "kind": "action_confirmation",
                "type": "interrupt",
                "interrupt_type": "confirmation_required",
                "provider": provider_label,
                "tool_name": tool_name,
                "arguments": tc.get("args", {}),
                "message": f"Are you sure you want to run {tool_name}? This action is irreversible.",
                "actions": ["approve", "cancel"]
            })

            decision, user_input = read_resume_value(resume_val)
            if decision in {"cancel", "reject", "skip", "no"}:
                logger.info(f"[ToolNode] User {user_id} CANCELLED/REJECTED destructive tool '{tool_name}' execution")
                return cancel_all(tool_calls)
            logger.info(f"[ToolNode] User {user_id} APPROVED destructive tool '{tool_name}' execution")

    validated_tokens: dict[str, str] = {}
    checked_providers: set[str] = set()

    for tc in tool_calls:
        provider = tool_provider_map.get(tc["name"])
        if provider is None or provider in checked_providers:
            continue
        checked_providers.add(provider)

        provider_label = PROVIDERS[provider].label if provider in PROVIDERS else provider

        # Auth check
        token = await resolve_token(user_id, provider)

        if token is None:
            logger.warning(f"[ToolNode] Provider '{provider}' auth MISS for user {user_id}. Triggering auth interrupt.")
            try:
                auth_url = await oauth_engine.get_authorization_url(provider, user_id, session_id)
            except Exception as e:
                logger.error(f"Auth URL generation failed for {provider}: {e}")
                auth_url = ""

            resume_val = interrupt({
                "id" : f"auth_{provider}_{tc['id']}",
                "kind": "connector_required",
                "type": "interrupt",
                "interrupt_type": "auth_required",
                "provider": provider_label,
                "provider_name": provider,
                "auth_url": auth_url,
                "message": f"Connect {provider_label} to continue.",
                "actions": ["connected", "cancel"]
            })
            
            decision, user_input = read_resume_value(resume_val)
            if decision in {"cancel", "reject", "skip"}:
                logger.info(f"[ToolNode] User {user_id} CANCELLED auth prompt for '{provider}'")
                return cancel_all(tool_calls)

            token = await resolve_token(user_id, provider)
            if token is None:
                logger.warning(f"[ToolNode] User {user_id} resumed but provider '{provider}' is still unauthenticated.")
                return {
                    "messages": [
                        ToolMessage(
                            content=(
                                f"{provider_label} is still not connected. "
                                f"Please connect via Settings and try again."
                            ),
                            tool_call_id=tc["id"],
                        )
                        for tc in tool_calls
                        if tool_provider_map.get(tc["name"]) == provider
                    ]
                }
            logger.info(f"[ToolNode] Provider '{provider}' auth RESOLVED after interrupt for user {user_id}")
        else:
            logger.info(f"[ToolNode] Provider '{provider}' auth HIT for user {user_id}")

        # Rate limit check
        try:
            rate_result = await check_and_increment(user_id, provider)
        except Exception as e:
            logger.error(f"Rate limit check failed for {provider}/{user_id}: {e}")
            validated_tokens[provider] = token
            continue

        if not rate_result.allowed:
            logger.warning(f"[ToolNode] Rate limit EXCEEDED for provider '{provider}' (user {user_id}). Retry after {rate_result.retry_after}s")
            action = interrupt({
                "interrupt_id": f"rate_{provider}_{tc['id']}",
                "kind": "rate_limited",
                "type": "interrupt",
                "interrupt_type": "rate_limited",
                "provider": provider_label,
                "provider_name": provider,
                "retry_after": rate_result.retry_after,
                "message": (
                    f"Rate limit reached for {provider_label}. "
                    f"Try again in {rate_result.retry_after}s."
                ),
                "actions": ["connected", "cancel"]
            })

            decision, user_input = read_resume_value(resume_val)
            
            if decision in {"cancel", "reject", "skip"}:
                logger.info(f"[ToolNode] User {user_id} CANCELLED rate-limit wait for '{provider}'")
                return cancel_all(tool_calls)

            try:
                rate_result = await check_and_increment(user_id, provider)
            except Exception as e:
                logger.error(f"Rate limit re-check failed for {provider}/{user_id}: {e}")
                validated_tokens[provider] = token
                continue

            if not rate_result.allowed:
                logger.warning(f"[ToolNode] User {user_id} still rate-limited on '{provider}' after retry.")
                return {
                    "messages": [
                        ToolMessage(
                            content=(
                                f"Still rate limited on {provider_label}. "
                                f"Try again in {rate_result.retry_after}s."
                            ),
                            tool_call_id=tc["id"],
                        )
                        for tc in tool_calls
                        if tool_provider_map.get(tc["name"]) == provider
                    ]
                }
            logger.info(f"[ToolNode] Rate limit check PASSED after retry for '{provider}' (user {user_id})")
        else:
            logger.info(f"[ToolNode] Rate limit check PASSED for provider '{provider}' (user {user_id})")

        validated_tokens[provider] = token

    # ── Phase 2: execute all tool calls ──────────────────────────────────────

    tool_results: list[ToolMessage] = []

    for tc in tool_calls:
        tool_name = tc["name"]
        arguments = tc.get("args", {})
        tool_call_id = tc["id"]
        provider = tool_provider_map.get(tool_name)

        emit_activity("running", tool_name, f"Running {tool_name}...", config)

        try:
            if provider:
                # Remote MCP tool
                token = validated_tokens.get(provider) or await resolve_token(user_id, provider)
                if not token:
                    output = (
                        f"[Error] No valid token for {provider}. "
                    )
                else:
                    output = await execute_mcp_tool(
                        tool_name=tool_name,
                        arguments=arguments,
                        token=token,
                        provider_name=provider,
                    )

                    # ── Phase 2.5: Runtime auth failure detection ─────────
                    # If the MCP server returned an auth/permission error,
                    # the token in DB is stale (expired, revoked, wrong scope).
                    # Clean up and trigger an interrupt so the user gets the
                    # authorization widget instead of a confusing text error.
                    output_lower = str(output).lower()
                    _AUTH_ERROR_SIGNALS = [
                        "unauthorized", "unauthenticated", "permission denied",
                        "invalid_grant", "token has been expired or revoked",
                        "access denied", "insufficient permission", "403",
                        "requires authentication", "not authorized",
                        "invalid credentials", "authentication required",
                    ]
                    if any(signal in output_lower for signal in _AUTH_ERROR_SIGNALS):
                        logger.warning(
                            f"Runtime auth failure for {provider}/{tool_name}: "
                            f"token exists but MCP rejected it. Triggering interrupt."
                        )
                        # Invalidate the stale token so next attempt starts clean
                        from services.cache import invalidate_token, invalidate_user_connections
                        await invalidate_token(user_id, provider)
                        await invalidate_user_connections(user_id)
                        # Delete the stale DB row so resolve_token returns None next time
                        try:
                            from sqlalchemy import delete as sa_delete
                            async with async_session_maker() as db:
                                await db.execute(
                                    sa_delete(UserConnection).where(
                                        UserConnection.user_id == user_id,
                                        UserConnection.provider == provider,
                                    )
                                )
                                await db.commit()
                        except Exception as cleanup_err:
                            logger.error(f"Failed to clean stale connection: {cleanup_err}")

                        provider_label = PROVIDERS[provider].label if provider in PROVIDERS else provider
                        try:
                            auth_url = await oauth_engine.get_authorization_url(provider, user_id, session_id)
                        except Exception as e:
                            logger.error(f"Auth URL generation failed for {provider}: {e}")
                            auth_url = ""

                        interrupt({
                            "id": f"reauth_{provider}_{tc['id']}",
                            "kind": "connector_required",
                            "type": "interrupt",
                            "interrupt_type": "auth_required",
                            "provider": provider_label,
                            "provider_name": provider,
                            "auth_url": auth_url,
                            "message": f"Reconnect {provider_label} to continue.",
                            "actions": ["connected", "cancel"]
                        })

            elif tool_name in _executable_tools:
                # Native or custom tool
                output = await _executable_tools[tool_name].ainvoke(
                    arguments, config=config
                )

            else:
                logger.warning(f"Unknown tool called: {tool_name}")
                output = (
                    f"Tool '{tool_name}' is not available. "
                    f"It may have been removed or renamed."
                )
            
            logger.info(f"[ToolNode] Tool '{tool_name}' executed SUCCESSFULLY for user {user_id}")
            emit_activity("completed", tool_name, f"Completed {tool_name}", config)

        except GraphInterrupt:
            # HITL interrupt from inside a custom tool (e.g. create_workflow auth check)
            # or from our Phase 2.5 runtime auth failure detection above.
            # Must propagate to compiled graph — never swallow
            logger.info(f"[ToolNode] HITL GraphInterrupt raised during tool '{tool_name}' execution for user {user_id}")
            raise

        except Exception as e:
            logger.error(
                f"[ToolNode] Tool execution FAILED: {tool_name} for user {user_id}: "
                f"{type(e).__name__}: {e}",
                exc_info=True,
            )
            output = (
                f"Tool '{tool_name}' failed: {str(e)}. "
                f"Try a different approach."
            )

            emit_activity("failed", tool_name, f"Failed {tool_name}", config)

        tool_results.append(
            ToolMessage(content=str(output), tool_call_id=tool_call_id)
        )

    logger.info(f"[ToolNode] Phase 2 Execution COMPLETE for user {user_id}: {len(tool_results)} ToolMessage(s) returned")
    return {"messages": tool_results}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_agent(state: AgentState):
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not getattr(last, "tool_calls", None):
        return END

    fixed_calls = []
    mcp_calls = []

    for tc in last.tool_calls:
        name = tc.get("name", "")
        if name in tool_provider_map or name in _executable_tools:
            fixed_calls.append(tc)
        else:
            mcp_calls.append(tc)

    sends = []
    if fixed_calls:
        logger.info(f"Routing fixed/system tool calls for user {state['user_id']}: {[tc.get('name') for tc in fixed_calls]}")
        filtered_msg = AIMessage(content=last.content, tool_calls=fixed_calls, id=last.id)
        filtered_state = {**state, "messages": state["messages"][:-1] + [filtered_msg]}
        sends.append(Send("tool_node", filtered_state))

    if mcp_calls:
        logger.info(f"Routing user remote MCP tool calls for user {state['user_id']}: {[tc.get('name') for tc in mcp_calls]}")
        filtered_msg = AIMessage(content=last.content, tool_calls=mcp_calls, id=last.id)
        filtered_state = {**state, "messages": state["messages"][:-1] + [filtered_msg]}
        sends.append(Send("mcp_tool_node", filtered_state))

    if not sends:
        # Tool calls exist but matched nothing — hallucinated tool names
        tool_names = [tc.get("name") for tc in last.tool_calls]
        logger.warning(
            f"Unroutable tool calls for user {state['user_id']}: {tool_names}. "
            "Likely hallucinated tool names — ending turn."
        )
        return END

    return sends

# ─────────────────────────────────────────────────────────────────────────────
# summarize_conversation  (runs after final agent response when needed)
# ─────────────────────────────────────────────────────────────────────────────

# Optimal Summarization Thresholds (Hybrid: Token Count + Message Count)
SUMMARIZE_MAX_MESSAGES = 16   # Summarize if message count in state exceeds 16
SUMMARIZE_MAX_TOKENS = 4000    # Summarize if total tokens in state exceed 4000 (~16,000 chars)
KEEP_LAST_N_MESSAGES = 10     # Keep last 10 messages (5 turns) verbatim for immediate context

from langchain_core.messages import RemoveMessage

def _estimate_message_tokens(messages: list) -> int:
    """
    Fast, reliable token estimator (~4 chars per token average).
    Counts string content and structured message content blocks.
    """
    total_chars = 0
    for m in messages:
        if hasattr(m, "content") and m.content:
            if isinstance(m.content, str):
                total_chars += len(m.content)
            elif isinstance(m.content, list):
                for item in m.content:
                    if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                        total_chars += len(item["text"])
                    elif isinstance(item, str):
                        total_chars += len(item)
    return max(1, total_chars // 4)


async def summarize_conversation(state: AgentState, config: RunnableConfig) -> dict:
    """
    Runs before agent_node on every turn. Uses a hybrid strategy:
    Triggers summarization if message count > SUMMARIZE_MAX_MESSAGES (16)
    OR total estimated tokens > SUMMARIZE_MAX_TOKENS (3,500).
    """
    messages = state.get("messages", [])
    user_id = state.get("user_id", "unknown")
    thread_id = state.get("thread_id", "unknown")
    total_msgs = len(messages)
    estimated_tokens = _estimate_message_tokens(messages)

    # Hybrid trigger evaluation
    exceeds_msg_count = total_msgs > SUMMARIZE_MAX_MESSAGES
    exceeds_token_count = estimated_tokens > SUMMARIZE_MAX_TOKENS
    should_summarize = exceeds_msg_count or exceeds_token_count

    if not should_summarize:
        logger.debug(
            f"[SummarizeNode] User {user_id} | Thread {thread_id}: "
            f"Msgs ({total_msgs}/{SUMMARIZE_MAX_MESSAGES}), Tokens (~{estimated_tokens}/{SUMMARIZE_MAX_TOKENS}). Skipping."
        )
        return {}

    existing_summary = state.get("summary", "")

    # Messages to summarize = everything except the last KEEP_LAST_N_MESSAGES
    cutoff_idx = max(0, len(messages) - KEEP_LAST_N_MESSAGES)
    # Adjust cutoff index backwards if it lands on a ToolMessage to avoid splitting an AIMessage from its tool responses
    while cutoff_idx > 0 and (isinstance(messages[cutoff_idx], ToolMessage) or getattr(messages[cutoff_idx], "tool_call_id", None)):
        cutoff_idx -= 1

    messages_to_summarize = messages[:cutoff_idx]
    messages_to_keep = messages[cutoff_idx:]

    if not messages_to_summarize:
        return {}

    trigger_reason = []
    if exceeds_msg_count:
        trigger_reason.append(f"msgs ({total_msgs} > {SUMMARIZE_MAX_MESSAGES})")
    if exceeds_token_count:
        trigger_reason.append(f"tokens (~{estimated_tokens} > {SUMMARIZE_MAX_TOKENS})")
    reason_str = " & ".join(trigger_reason)

    logger.info(
        f"[SummarizeNode] Triggering hybrid summarization for User {user_id} (Thread: {thread_id}) | "
        f"Reason: {reason_str} | Total msgs: {total_msgs} (~{estimated_tokens} tokens) | "
        f"Summarizing {len(messages_to_summarize)} msgs | Keeping {len(messages_to_keep)} msgs | "
        f"Has existing summary: {bool(existing_summary)}"
    )

    # Build summarization prompt
    conversation_text = "\n".join(
        f"{getattr(m, 'type', 'unknown').upper()}: {m.content[:500]}"
        for m in messages_to_summarize
        if hasattr(m, "content") and isinstance(m.content, str) and m.content.strip()
    )

    if existing_summary:
        prompt = (
            f"Below is an existing summary of earlier conversation:\n\n"
            f"{existing_summary}\n\n"
            f"And here are new messages since that summary:\n\n"
            f"{conversation_text}\n\n"
            f"Produce an updated, concise summary that captures all important context, "
            f"user preferences, and decisions. Keep it factual and under 500 words."
        )
    else:
        prompt = (
            f"Summarize this conversation concisely, capturing all important context, "
            f"user preferences, and decisions. Keep it factual and under 500 words.\n\n"
            f"{conversation_text}"
        )

    # Use a cheap model for summarization
    try:
        from langchain_openai import ChatOpenAI as SummarizeChatOpenAI
        summarizer_llm = SummarizeChatOpenAI(
            model="gpt-5-nano",
            api_key=settings.openai_api_key,
        )
        response = await summarizer_llm.ainvoke(
            [SystemMessage(content="You are a conversation summarizer. Produce a concise factual summary."),
             HumanMessage(content=prompt)]
        )
        new_summary = response.content
    except Exception as e:
        logger.error(f"[SummarizeNode] Summarization LLM call failed for User {user_id}: {e}", exc_info=True)
        return {}

    # Prune old messages from checkpointer via RemoveMessage
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if hasattr(m, "id") and m.id]

    logger.info(
        f"[SummarizeNode] Summarization complete for User {user_id} | "
        f"New summary length: {len(new_summary)} chars | "
        f"Pruned {len(delete_messages)} old messages from state."
    )

    return {
        "summary": new_summary,
        "messages": delete_messages,
    }




# ─────────────────────────────────────────────────────────────────────────────
# initialize_graph
# ─────────────────────────────────────────────────────────────────────────────

async def initialize_graph() -> None:
    """
    Called once from app lifespan. asyncio.Lock prevents double-init
    under Gunicorn multi-worker. Lazy fallback in stream_response
    handles cases where lifespan init is skipped.
    """
    global _llm_with_tools, _executable_tools, compiled_graph, _bounded_models

    async with _init_lock:
        if _llm_with_tools is not None:
            return

        logger.info("Initializing Orphic graph...")

        # DB + checkpointer schema
        try:
            await setup_db()
        except Exception as e:
            logger.error(f"DB setup failed: {e}")
            raise

        # Memory store (pgvector)
        try:
            await setup_memory_store()
        except Exception as e:
            logger.error(f"Memory store setup failed: {e}")
            raise

        # Native MCP tools (Playwright, Browserless etc.)
        native_tools: list = []
        try:
            from protocols.mcp.native_mcp_tool_config import get_native_mcp_tools
            native_tools = await get_native_mcp_tools()
            logger.info(f"Loaded {len(native_tools)} native MCP tools")
        except Exception as e:
            logger.warning(f"Native MCP tools failed to load, continuing without: {e}")


        # Build executable tool lookup
        all_runnable = native_tools + custom_tools
        for t in all_runnable:
            _executable_tools[t.name] = t

        logger.info(f"Registered {len(_executable_tools)} system/native executable tools & {len(schema_tools)} fixed MCP schema tools")

        # Compile graph
        #   START → summarize_node → agent_node → (tool_node ↔ agent_node)* → END
        builder = StateGraph(AgentState)
        builder.add_node("summarize_node", summarize_conversation)
        builder.add_node("agent_node", agent_node)
        builder.add_node("tool_node", tool_node)
        builder.add_node("mcp_tool_node", mcp_tool_node)
        builder.add_edge(START, "summarize_node")
        builder.add_edge("summarize_node", "agent_node")
        builder.add_conditional_edges("agent_node", _route_after_agent)
        builder.add_edge("tool_node", "agent_node")
        builder.add_edge("mcp_tool_node", "agent_node")

        compiled_graph = builder.compile(
            checkpointer=AsyncPostgresSaver(pool),
        ).with_config({"recursion_limit": 50})

        logger.info("Orphic graph compiled and ready")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

async def stream_response(user_message: str, thread_id: str, user_id: str, model:str = "gpt-5-nano"):
    """
    Async generator → (event_type, data) tuples.

    event_type:
      "token"     — str, streaming text chunk
      "interrupt" — dict, HITL payload for frontend
      "error"     — str, non-fatal, stream ends
      "done"      — None, stream complete
    """
    if compiled_graph is None:
        try:
            await initialize_graph()
        except Exception as e:
            logger.error(f"Graph init failed: {e}")
            yield ("error", "Service is starting up. Please try again.")
            return

    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id, "model": model},
        "run_name": "chat_turn",
        "metadata": {"user_id": str(user_id), "thread_id": thread_id}, "tags":[f"user: {user_id}"],
    }
    total_tokens = 0
    try:
        async for event in compiled_graph.astream_events(
            {
                "messages": [HumanMessage(content=user_message)],
                "user_id": user_id,
                "thread_id": thread_id,
                "model": model,
            },
            config=config,
            version="v2",
        ):
            if (
                event["event"] == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") == "agent_node"
            ):
                chunk = event["data"].get("chunk")
                if chunk:
                    reasoning_content = getattr(chunk, "reasoning_content", None)
                    if not reasoning_content and hasattr(chunk, "additional_kwargs") and isinstance(chunk.additional_kwargs, dict):
                        reasoning_content = chunk.additional_kwargs.get("reasoning_content") or chunk.additional_kwargs.get("thinking")
                    
                    if reasoning_content:
                        yield ("token", [{"type": "reasoning", "reasoning": reasoning_content}])

                    if hasattr(chunk, "content") and chunk.content:
                        yield ("token", chunk.content)
            
            elif event["event"] == "on_custom_event" and event["name"] =="tool_activity":
                yield ("activity", event["data"])
            
            if event["event"] == "on_chat_model_end":
                output = event["data"].get("output")
                if output and hasattr(output, "usage_metadata") and output.usage_metadata:
                    total_tokens += output.usage_metadata.get("total_tokens", 0)


    except Exception as e:
        logger.error(
            f"Stream error thread={thread_id}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        yield ("error", "An error occurred. Please try again.")
        return

    # Interrupt detection — astream_events does not surface interrupts.
    # aget_state() after loop end reveals any pending interrupt.
    try:
        snapshot = await compiled_graph.aget_state(config)
        if snapshot.next and snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    yield ("interrupt", task.interrupts[0].value)
                    yield ("done", {"total_tokens": total_tokens})
                    return
    except Exception as e:
        logger.error(f"Interrupt detection failed thread={thread_id}: {e}")

    yield ("done", {"total_tokens": total_tokens})


async def resume_response(thread_id: str, user_id: str, resume_value: dict):
    """
    Resume a paused graph after HITL interrupt.
    Yields same event types as stream_response.
    Handles chained interrupts (multi-provider auth).
    """
    if compiled_graph is None:
        yield ("error", "Agent not initialized.")
        return

    config = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "run_name": "chat_resume",
        "metadata": {"user_id": str(user_id), "thread_id": thread_id}, "tags":[f"user: {user_id}"],
    }

    total_tokens = 0
    try:
        async for event in compiled_graph.astream_events(
            Command(resume=resume_value),
            config=config,
            version="v2",
        ):
            if (
                event["event"] == "on_chat_model_stream"
                and event.get("metadata", {}).get("langgraph_node") == "agent_node"
            ):
                chunk = event["data"].get("chunk")
                if chunk:
                    reasoning_content = getattr(chunk, "reasoning_content", None)
                    if not reasoning_content and hasattr(chunk, "additional_kwargs") and isinstance(chunk.additional_kwargs, dict):
                        reasoning_content = chunk.additional_kwargs.get("reasoning_content") or chunk.additional_kwargs.get("thinking")
                    
                    if reasoning_content:
                        yield ("token", [{"type": "reasoning", "reasoning": reasoning_content}])

                    if hasattr(chunk, "content") and chunk.content:
                        yield ("token", chunk.content)
            
            elif event["event"] == "on_custom_event" and event["name"] =="tool_activity":
                yield ("activity", event["data"])
            
            if event["event"] == "on_chat_model_end":
                output = event["data"].get("output")
                if output and hasattr(output, "usage_metadata") and output.usage_metadata:
                    total_tokens += output.usage_metadata.get("total_tokens", 0)

       

    except Exception as e:
        logger.error(
            f"Resume error thread={thread_id}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        yield ("error", "An error occurred while resuming. Please try again.")
        return

    # Check for chained interrupt (e.g. second provider auth after first resolved)
    try:
        snapshot = await compiled_graph.aget_state(config)
        if snapshot.next and snapshot.tasks:
            for task in snapshot.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    yield ("interrupt", task.interrupts[0].value)
                    yield ("done", {"total_tokens": total_tokens})
                    return
    except Exception as e:
        logger.error(f"Post-resume interrupt detection failed thread={thread_id}: {e}")

    yield ("done", {"total_tokens": total_tokens})


async def update_agent_state(
    thread_id: str,
    user_id: str,
    messages: list,
) -> bool:
    """
    Inject messages directly into graph state.
    Used by chat_router for file upload events.
    Returns False on failure — non-fatal, caller logs and continues.
    """
    if compiled_graph is None:
        try:
            await initialize_graph()
        except Exception as e:
            logger.warning(f"Cannot update state — graph not initialized: {e}")
            return False

    try:
        await compiled_graph.aupdate_state(
            {"configurable": {"thread_id": thread_id, "user_id": user_id}},
            {"messages": messages},
            as_node="__start__",
        )
        logger.info(f"Updated agent state for user: {user_id}")
        return True
    except Exception as e:
        logger.error(
            f"State update failed thread={thread_id}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return False

async def get_thread_state(thread_id: str, user_id: str):
    """
    Load the full checkpointed state for a thread.
    Used by conversations router to display chat history.
    Returns None if graph is not initialized or state doesn't exist.
    """
    if compiled_graph is None:
        try:
            await initialize_graph()
        except Exception as e:
            logger.warning(f"Cannot get state — graph not initialized: {e}")
            return False

    try:
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        state = await compiled_graph.aget_state(config)
        return state
    except Exception as e:
        logger.error(
            f"State fetch failed thread={thread_id}: {type(e).__name__}: {e}",
            exc_info=True,
        )
        return None
