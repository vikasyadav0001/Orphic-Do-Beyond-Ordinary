"""
MCP Schema Loader + Lazy Executor
==================================

Two responsibilities:

1. load_schemas()
   Called ONCE at startup. Reads schemas.json, builds LangChain
   StructuredTool objects (schema-only, no execution), and builds
   the tool→provider map. No tokens. No network calls.

2. execute_mcp_tool()
   Called per invocation inside tool_node AFTER auth + rate limit checks.
   Opens a fresh MCP connection with the user's token, executes, closes.
   One connection per call. Guaranteed cleanup via async context manager.

Critical design note:
   The adapter's get_tools() already handles MCP JSON Schema → Pydantic
   conversion internally. For load_schemas() we rebuild from stored JSON,
   which requires careful schema reconstruction. For execute_mcp_tool()
   we let the adapter do the full conversion at runtime — no manual
   schema building needed.
"""

import json
import os
import logging
from typing import Any, Optional, Union, List

from pydantic import BaseModel, Field, create_model
from pydantic.fields import FieldInfo
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from services.connections.providers import PROVIDERS
from utils.logger import get_logger

import logging
import sys

# httpx_logger = logging.getLogger("httpx")
# httpx_logger.setLevel(logging.DEBUG)
# httpx_logger.addHandler(logging.StreamHandler(sys.stdout))

# httpcore_logger = logging.getLogger("httpcore")
# httpcore_logger.setLevel(logging.DEBUG)
# httpcore_logger.addHandler(logging.StreamHandler(sys.stdout))

logger = get_logger(__name__)

SCHEMAS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "mcp_schemas", "schemas.json"
)

# Seconds before we give up on a hung MCP server.
# Prevents Gunicorn worker starvation.
MCP_TOOL_TIMEOUT = 15

# JSON Schema type → Python type mapping
# "any" fallback used for unknown/complex types so they don't crash the model builder
_JSON_TYPE_MAP: dict[str, type] = {
    "string":  str,
    "integer": int,
    "number":  float,
    "boolean": bool,
    "array":   list,
    "object":  dict,
    "null":    type(None),
}


def _json_type_to_python(type_str: str | list | None) -> type:
    """
    Convert a JSON Schema type string to a Python type.

    Handles:
      - Simple strings: "string" → str
      - Arrays (union types): ["string", "null"] → Optional[str]
      - None/missing: falls back to Any

    Why robust handling here?
      MCP servers from different providers use different subsets of
      JSON Schema. GitHub uses clean types. Notion may use union types.
      A crash here means the tool never loads — silent failure in prod.
    """
    if type_str is None:
        return Any

    if isinstance(type_str, list):
        # Union type e.g. ["string", "null"] — take first non-null type
        non_null = [t for t in type_str if t != "null"]
        if not non_null:
            return Any
        base = _JSON_TYPE_MAP.get(non_null[0], Any)
        if "null" in type_str:
            return Optional[base]
        return base

    return _JSON_TYPE_MAP.get(type_str, Any)


def _build_args_model(tool_name: str, input_schema: dict) -> type[BaseModel] | None:
    """
    Build a Pydantic BaseModel from a JSON Schema dict.

    This is what gives the LLM accurate argument information.
    Without this, the LLM doesn't know what args to pass → wrong calls.

    Handles gracefully:
      - Missing properties (no-arg tools like ping/status)
      - Unknown types (falls back to Any)
      - Nested objects (treated as dict — deep nesting not reconstructed)
      - Missing descriptions (field still created, just no description)
      - Schema build failures (returns None → tool loads with no schema)

    Why not reconstruct nested Pydantic models?
      Nested JSON Schema ($defs/$ref) reconstruction is complex and
      brittle. Treating nested objects as dict is safe — the LLM sends
      a dict, the MCP server validates it on its end. We don't need
      Pydantic to deep-validate MCP inputs.
    """
    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    if not properties:
        return None  # no-arg tool — valid, no model needed

    fields: dict[str, tuple] = {}

    for prop_name, prop_def in properties.items():
        try:
            # Handle $ref — nested schema reference, treat as dict
            if "$ref" in prop_def or prop_def.get("type") == "object":
                python_type = dict
            else:
                python_type = _json_type_to_python(prop_def.get("type"))

            description = prop_def.get("description", "")
            default = prop_def.get("default", ...)  # ... = required in Pydantic

            is_required = prop_name in required

            if is_required:
                if description:
                    fields[prop_name] = (python_type, Field(..., description=description))
                else:
                    fields[prop_name] = (python_type, ...)
            else:
                # Optional field — wrap in Optional if not already
                if not (hasattr(python_type, "__origin__") and python_type.__origin__ is Union):
                    opt_type = Optional[python_type]
                else:
                    opt_type = python_type
                actual_default = prop_def.get("default", None)
                if description:
                    fields[prop_name] = (opt_type, Field(actual_default, description=description))
                else:
                    fields[prop_name] = (opt_type, actual_default)

        except Exception as e:
            # Single field failure → skip that field, don't crash the whole tool
            logger.warning(
                f"Skipping field '{prop_name}' in tool '{tool_name}': {e}"
            )
            continue

    if not fields:
        return None

    try:
        # Sanitize model name — Pydantic requires valid Python identifier
        safe_name = "".join(c if c.isalnum() else "_" for c in tool_name) + "_args"
        return create_model(safe_name, **fields)
    except Exception as e:
        logger.warning(f"Failed to build args model for '{tool_name}': {e}")
        return None


def _build_schema_only_tool(
    name: str,
    description: str,
    input_schema: dict,
) -> StructuredTool | None:
    """
    Build a LangChain StructuredTool with full args schema but no execution.

    The LLM uses this to:
      - Know the tool exists
      - Know what arguments to send
      - Decide when to call it

    Execution happens in execute_mcp_tool() via fresh MCP connection.
    The func raises NotImplementedError — safety net if tool is somehow
    invoked directly bypassing tool_node.

    Returns None if the tool can't be built — caller skips it.
    Skipping one broken tool is better than crashing all tool loading.
    """
    args_model = _build_args_model(name, input_schema)

    def _not_executable(**kwargs):
        raise NotImplementedError(
            f"Tool '{name}' is schema-only. "
            f"Use execute_mcp_tool() inside tool_node."
        )

    try:
        return StructuredTool.from_function(
            func=_not_executable,
            name=name,
            description=description or f"MCP tool: {name}",
            args_schema=args_model,
            # infer_schema=False prevents LangChain from overriding our
            # carefully built args_schema with a schema inferred from
            # _not_executable's signature (which has no real args)
            infer_schema=False,
        )
    except Exception as e:
        logger.warning(f"Failed to build StructuredTool for '{name}': {e}")
        return None


def load_schemas() -> tuple[list[StructuredTool], dict[str, str]]:
    """
    Load MCP tool schemas from schemas.json at startup.
    No tokens. No network. Pure file read + Pydantic model building.

    Returns:
      tools:             list of StructuredTool (schema only, no execution)
      tool_provider_map: {"list_issues": "github", "create_page": "notion", ...}

    Failures are logged and skipped — a broken tool schema never crashes startup.
    If schemas.json is missing, returns empty lists (agent starts with no MCP tools).
    """
    abs_path = os.path.abspath(SCHEMAS_PATH)

    if not os.path.exists(abs_path):
        logger.warning(
            f"schemas.json not found at {abs_path}. "
            f"Run mcp_schemas/extract_schemas.py to generate it. "
            f"Agent will start with no MCP tools."
        )
        return [], {}

    try:
        with open(abs_path) as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"schemas.json is malformed: {e}. No MCP tools loaded.")
        return [], {}

    if not raw:
        logger.warning("schemas.json is empty. No MCP tools loaded.")
        return [], {}

    tools: list[StructuredTool] = []
    tool_provider_map: dict[str, str] = {}
    skipped = 0

    for provider_name, tool_schemas in raw.items():
        if not isinstance(tool_schemas, dict):
            logger.warning(f"Invalid schema format for provider '{provider_name}' — skipping")
            continue

        for schema in tool_schemas.values():
            tool_name = schema.get("name")
            if not tool_name:
                logger.warning(f"Unnamed tool in provider '{provider_name}' — skipping")
                skipped += 1
                continue

            tool = _build_schema_only_tool(
                name=tool_name,
                description=schema.get("description", ""),
                input_schema=schema.get("inputSchema", {}),
            )

            if tool is None:
                logger.warning(f"Skipped tool '{tool_name}' from '{provider_name}' — build failed")
                skipped += 1
                continue

            tools.append(tool)
            tool_provider_map[tool_name] = provider_name

    logger.info(
        f"Loaded {len(tools)} MCP tool schemas from {len(raw)} providers: "
        f"{', '.join(raw.keys())}"
        + (f" ({skipped} skipped)" if skipped else "")
    )

    return tools, tool_provider_map


async def execute_mcp_tool(
    tool_name: str,
    arguments: dict,
    token: str,
    provider_name: str,
) -> str:
    """
    Execute a single MCP tool call with the user's live token.
    Called from tool_node AFTER auth + rate limit checks pass.

    Connection lifecycle:
      - Fresh connection opened per call (lazy, per-user token)
      - async context manager guarantees close even on exception
      - 15s timeout prevents Gunicorn worker starvation on hung servers

    Why get_tools() + search instead of direct call_tool()?
      MultiServerMCPClient doesn't expose a direct call_tool() method
      in the current Python API. get_tools() returns executable tool
      objects — we find ours and ainvoke() it. The adapter handles
      all MCP protocol details including error conversion.

    Error handling:
      - Tool execution errors (MCP isError=True): returned as string
        to agent (handle_tool_errors=True default)
      - Transport/connection errors: caught here, returned as string
      - Timeout: returned as string with duration
      All errors are strings — agent sees them as tool output and
      can self-correct rather than crashing.
    """
    provider = PROVIDERS.get(provider_name)
    if not provider:
        return f"[Error] Unknown provider '{provider_name}'"

    if not provider.mcp_url or provider.mcp_url == "...":
        return f"[Error] No MCP URL configured for provider '{provider_name}'"

    if not token:
        return f"[Error] No token provided for '{provider_name}'"

    logger.info(f"Args before clean-up: {arguments}")
    arguments = {k: v for k, v in arguments.items() if v is not None}
    logger.info(f"Args after clean-up: {arguments}")
    import httpx

    class DebugTransport(httpx.AsyncHTTPTransport):
        async def handle_async_request(self, request):
            body = request.content.decode("utf-8", errors="replace")
            logger.info(f"→ {request.method} {request.url}")
            logger.info(f"→ body: {body[:500]}")
            response = await super().handle_async_request(request)
            logger.info(f"← {response.status_code}")
            return response

    import asyncio
    from mcp.client.streamable_http import streamable_http_client
    from mcp import ClientSession
    try:
        logger.info(f"Executing tool {tool_name}..")
        async with asyncio.timeout(MCP_TOOL_TIMEOUT):
            async with httpx.AsyncClient(transport=DebugTransport(), headers={
                provider.token_header: f"{provider.token_prefix} {token}",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }) as client:
                async with streamable_http_client(provider.mcp_url, http_client=client) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments)
                        logger.info(f"{tool_name} tool call result: {result}")
                        return str(result)


    except asyncio.TimeoutError:
        logger.error(
            f"MCP timeout ({MCP_TOOL_TIMEOUT}s): '{tool_name}' on '{provider_name}'"
        )
        return (
            f"[Error] Tool '{tool_name}' timed out after {MCP_TOOL_TIMEOUT} seconds. "
            f"The {provider_name} service may be slow or unavailable."
        )

    except Exception as e:
        logger.error(
            f"MCP execution failed: '{tool_name}' on '{provider_name}': {e}",
            exc_info=True,
        )
        return f"[Error] Tool '{tool_name}' failed: {str(e)}"
