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
from langgraph.types import Command, interrupt
from langchain_core.callbacks.manager import dispatch_custom_event

from config import get_settings
from memory.graph_checkpointer import pool, setup_db
from memory.long_term_memory import retrieve_memory, setup_memory_store
from prompts.system_persona_prompt import get_prompt
from protocols.mcp.remote_mcp_client_config import execute_mcp_tool, load_schemas
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
from middleware.summarization_middleware import maybe_summarize
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

# ── Module-level state ────────────────────────────────────────────────────────

_llm_with_tools = None
_executable_tools: dict = {}
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
    model:str


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


def _sanitize_messages(messages: list) -> list:
    """Remove AIMessages with tool_calls that have no matching ToolMessage."""
    tool_call_ids_responded = {
        msg.tool_call_id 
        for msg in messages 
        if hasattr(msg, "tool_call_id")
    }
    
    sanitized = []
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Keep only if all tool calls have responses
            all_responded = all(
                tc["id"] in tool_call_ids_responded 
                for tc in msg.tool_calls
            )
            if not all_responded:
                continue  # drop the orphaned AIMessage
        sanitized.append(msg)
    
    return sanitized

# ─────────────────────────────────────────────────────────────────────────────
# agent_node
# ─────────────────────────────────────────────────────────────────────────────



async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    user_id = state["user_id"]

    # 1. Memory — Redis-cached 120s, falls back to pgvector on miss
    memories: list[str] = []
    try:
        redis = await get_redis()
        cached = await redis.get(f"memory:{user_id}")
        if cached:
            memories = json.loads(cached)
        else:
            memories = await retrieve_memory(
                user_id=user_id,
                query="user details preferences goals",
            )
            if memories:
                await redis.setex(f"memory:{user_id}", 120, json.dumps(memories))
    except Exception as e:
        logger.error(f"Memory retrieval failed for {user_id}: {e}")

    # 2. Connected providers — Redis → DB fallback
    connections: set[str] = set()
    try:
        connections = await get_user_connections(user_id) or set()
        if not connections:
            connections = await fetch_connections_from_db(user_id)
            if connections:
                await set_user_connections(user_id, connections)
    except Exception as e:
        logger.error(f"Connection fetch failed for {user_id}: {e}")

    # 3. Summarize if over token threshold
    messages = state["messages"]

    try:
        messages = await maybe_summarize(messages, user_id)
    except Exception as e:
        logger.error(f"Summarization failed for {user_id}: {e}")

    # 4. System prompt
    try:
        
        system_prompt = get_prompt(
            user_memory=memories,
            connected_providers=connections,
            available_providers=set(PROVIDERS.keys()) - connections,
        )
    except Exception as e:
        logger.error(f"get_prompt failed for {user_id}: {e}")
        system_prompt = "You are Orphic, a helpful AI assistant."

    # 5. LLM call
    requested_model = state.get("model", "gpt-5-nano") or config.get("configurable", {}).get("model", "gpt-5-nano")
    
    active_llm = _bounded_models.get(requested_model, _bounded_models["gpt-5-nano"])
    
    try:
        clean_messages = _sanitize_messages(messages)
        response = await active_llm.ainvoke(
            [SystemMessage(content=system_prompt)] + clean_messages,
            config=config,
        )
    except Exception as e:
        logger.error(f"LLM call failed for {user_id}: {e}", exc_info=True)
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
        return {"messages": []}

    # ── Phase 1: validate all remote tool calls ───────────────────────────────

    # 1. Destructive action confirmation interrupt check
    for tc in tool_calls:
        tool_name = tc["name"]
        if is_destructive_tool(tool_name):
            provider = tool_provider_map.get(tool_name)
            provider_label = PROVIDERS[provider].label if provider and provider in PROVIDERS else (provider or "Orphic")
            
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
                return cancel_all(tool_calls)

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
                return cancel_all(tool_calls)

            token = await resolve_token(user_id, provider)
            if token is None:
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

        # Rate limit check
        try:
            rate_result = await check_and_increment(user_id, provider)
        except Exception as e:
            logger.error(f"Rate limit check failed for {provider}/{user_id}: {e}")
            validated_tokens[provider] = token
            continue

        if not rate_result.allowed:
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
                return cancel_all(tool_calls)

            try:
                rate_result = await check_and_increment(user_id, provider)
            except Exception as e:
                logger.error(f"Rate limit re-check failed for {provider}/{user_id}: {e}")
                validated_tokens[provider] = token
                continue

            if not rate_result.allowed:
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
            
            emit_activity("completed", tool_name, f"Completed {tool_name}", config)

        except GraphInterrupt:
            # HITL interrupt from inside a custom tool (e.g. create_workflow auth check)
            # or from our Phase 2.5 runtime auth failure detection above.
            # Must propagate to compiled graph — never swallow
            raise

        except Exception as e:
            logger.error(
                f"Tool execution failed: {tool_name} for {user_id}: "
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

    return {"messages": tool_results}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "tool_node"
    return END


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

        # Bind all tools to LLM
        # LLM sees schema tools (remote MCP) + native + custom + workflow
        all_tools = schema_tools + all_runnable
        # _llm_with_tools = llm.bind_tools(all_tools)
        # logger.info(f"LLM bound with {len(all_tools)} tools")
        for model_key, model_instance in SUPPORTED_MODELS.items():
            _bounded_models[model_key] = model_instance.bind_tools(all_tools)

        logger.info(f"Pre-bound {len(all_tools)} tools to {len(_bounded_models)} models")

        # Compile graph
        builder = StateGraph(AgentState)
        builder.add_node("agent_node", agent_node)
        builder.add_node("tool_node", tool_node)
        builder.add_edge(START, "agent_node")
        builder.add_conditional_edges("agent_node", _route_after_agent)
        builder.add_edge("tool_node", "agent_node")

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
    if not _bounded_models:
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
                if chunk and hasattr(chunk, "content") and chunk.content:
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
    if not _bounded_models:
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
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield ("token", chunk.content)
                elif event["event"] == "on_custom_event" and event["name"] == "tool_activity":
                    yield ("activity", event["data"])
            
            #check for token useage
            if event["event"] == "on_chat_model_end":
                output = event["data"].get("output")
                if output and hasattr(output, "usage_metadata") and output.usage_metadata:
                    total_tokens += output.usage_metadata.get("total_tokens", 0)


        yield ("done", {"total_tokens": total_tokens})

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
                    yield ("done", None)
                    return
    except Exception as e:
        logger.error(f"Post-resume interrupt detection failed thread={thread_id}: {e}")

    yield ("done", None)


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
