"""
MCP Config Service
===================
Validates user MCP configs, discovers tools from remote servers,
and reconstructs schema-only tools from cached schemas for bind_tools().
"""

import json
import hashlib
import asyncio
import ipaddress
from sqlalchemy import select
from urllib.parse import urlparse

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool
from protocols.mcp.remote_mcp_client_config import _build_schema_only_tool
from db.models import UserMCPConfig, async_session_maker
from services.cache import (
    set_cached_mcp_config,
    set_cached_mcp_tool_schemas,
    set_mcp_config_version,
)
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)
settings = get_settings()

# ── Constants ─────────────────────────────────────────────────────────────
MAX_SERVERS = 10
MAX_CONFIG_SIZE_BYTES = 65536  # 64KB
DISCOVERY_TIMEOUT = 10  # seconds

# Private/reserved IP ranges to block (SSRF protection)
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("10.0.0.0/8"),        # private class A
    ipaddress.ip_network("172.16.0.0/12"),     # private class B
    ipaddress.ip_network("192.168.0.0/16"),    # private class C
    ipaddress.ip_network("169.254.0.0/16"),    # link-local
    ipaddress.ip_network("0.0.0.0/8"),         # "this" network
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
]

_BLOCKED_HOSTNAMES = {"localhost", "0.0.0.0", "[::]", "[::1]"}

RESERVED_SERVER_NAMES = {
    # 9 Built-in OAuth Providers & common aliases
    "github", "google_gmail", "google_drive", "google_sheets", 
    "google_docs", "google_calendar", "notion", "slack", "atlassian",
    "gmail", "sheets", "docs", "drive", "calendar", "jira", "confluence",
    # Native System Tools
    "exa", "playwright", "n8n", "browserless"
}

CORE_RESERVED_KEYWORDS = {
    "notion", "github", "gmail", "drive", "sheets", "docs", "calendar", 
    "slack", "atlassian", "jira", "confluence", "exa", "playwright", "n8n"
}

# ── Function 1: Validate Config ──────────────────────────────────────────

def validate_config(config_json: dict) -> dict:
    """
    Validate a user-submitted MCP server config before saving.
    
    Enforces:
      - Remote-only (no stdio/command entries)
      - SSRF protection (no private/reserved IPs)
      - HTTPS in production
      - Server count and size limits
      - Required fields
    
    Returns the validated config dict.
    Raises ValueError with a specific message on any violation.
    """
    import json

    if not isinstance(config_json, dict):
        raise ValueError("Config must be a JSON object mapping server names to server configs.")

    if len(config_json) > MAX_SERVERS:
        raise ValueError(f"Maximum {MAX_SERVERS} servers allowed, got {len(config_json)}.")

    # Size check — prevent absurdly large configs
    config_size = len(json.dumps(config_json).encode("utf-8"))
    if config_size > MAX_CONFIG_SIZE_BYTES:
        raise ValueError(f"Config exceeds maximum size of {MAX_CONFIG_SIZE_BYTES // 1024}KB.")

    validated = {}

    for server_name, server_config in config_json.items():
        if not isinstance(server_name, str) or not server_name.strip():
            raise ValueError("Server names must be non-empty strings.")

        norm_name = server_name.lower().strip()
        if norm_name in RESERVED_SERVER_NAMES or any(k in norm_name for k in CORE_RESERVED_KEYWORDS):
            raise ValueError(
                f"'{server_name}' Server is already integrated in connectors. "
                f"Please remove '{server_name}' and use existing connectors"
            )

        if not isinstance(server_config, dict):
            raise ValueError(f"Config for '{server_name}' must be a JSON object.")

        # ── Reject stdio/local process configs ────────────────────────
        if any(key in server_config for key in ("command", "args")):
            raise ValueError(
                f"Server '{server_name}': local process configs (command/args) are not allowed. "
                f"Only remote HTTP/SSE servers are supported."
            )

        raw_transport = server_config.get("transport") or server_config.get("type") or "http"
        if raw_transport == "stdio":
            raise ValueError(
                f"Server '{server_name}': stdio transport is not allowed. "
                f"Only remote HTTP/SSE servers are supported."
            )

        valid_transports = {"http", "sse", "streamable_http", "streamable-http"}
        if raw_transport not in valid_transports:
            raise ValueError(
                f"Server '{server_name}': unsupported transport '{raw_transport}'. "
                f"Supported transports are: 'http', 'sse'."
            )

        transport = "http" if raw_transport in {"http", "streamable_http", "streamable-http"} else raw_transport

        # ── Require URL ───────────────────────────────────────────────
        url = server_config.get("url")
        if not url or not isinstance(url, str):
            raise ValueError(f"Server '{server_name}': 'url' is required and must be a string.")

        # ── SSRF protection ───────────────────────────────────────────
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            raise ValueError(f"Server '{server_name}': could not parse hostname from URL '{url}'.")

        # Block known dangerous hostnames
        if hostname.lower() in _BLOCKED_HOSTNAMES:
            raise ValueError(f"Server '{server_name}': URL points to a blocked host '{hostname}'.")

        # Resolve hostname to IP and check against blocked ranges
        try:
            addr = ipaddress.ip_address(hostname)
        except ValueError:
            # hostname is a domain name, not an IP — that's fine
            addr = None

        if addr and any(addr in network for network in _BLOCKED_NETWORKS):
            raise ValueError(
                f"Server '{server_name}': URL resolves to a private/reserved IP address."
            )

        # ── Enforce HTTPS in production ───────────────────────────────
        is_production = getattr(settings, "environment", "production") == "production"
        if is_production and parsed.scheme != "https":
            raise ValueError(
                f"Server '{server_name}': only HTTPS URLs are allowed in production."
            )

        # ── Validate headers if present ───────────────────────────────
        headers = server_config.get("headers", {})
        if not isinstance(headers, dict):
            raise ValueError(f"Server '{server_name}': 'headers' must be a JSON object.")

        validated[server_name] = {
            "url": url,
            "transport": transport,
            "headers": headers,
        }

    return validated


# ── Function 2: Discover Tools ───────────────────────────────────────────

async def discover_tools(config_json: dict) -> list[dict]:
    """
    Connect to each MCP server in the config and discover available tools.
    Called at config save time only — NOT on every agent turn.
    
    Returns a list of serializable tool schema dicts:
      [{"name": "exa_search", "description": "...", "inputSchema": {...}}, ...]
    
    Tool names are prefixed with server name (tool_name_prefix=True)
    to prevent collisions across servers.
    
    Raises ValueError if any server is unreachable.
    """
    if not config_json:
        return []

    # Build MultiServerMCPClient-compatible config
    mcp_client_config = {}
    for server_name, server_config in config_json.items():
        mcp_client_config[server_name] = {
            "transport": server_config.get("transport", "http"),
            "url": server_config["url"],
            "headers": server_config.get("headers", {}),
        }

    try:
        async with asyncio.timeout(DISCOVERY_TIMEOUT):
            client = MultiServerMCPClient(mcp_client_config, tool_name_prefix=True)
            tools = await client.get_tools()
    except asyncio.TimeoutError:
        raise ValueError(
            f"Tool discovery timed out after {DISCOVERY_TIMEOUT}s. "
            f"One or more servers may be unreachable."
        )
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, "exceptions") and e.exceptions:
            sub_errs = [str(se) for se in e.exceptions]
            err_msg = "; ".join(sub_errs)
        raise ValueError(f"Failed to discover tools: {err_msg}")

    # Extract serializable schema dicts from live tool objects
    tool_schemas = []
    for tool in tools:
        if isinstance(tool.args_schema, dict):
            input_schema = tool.args_schema
        elif tool.args_schema and hasattr(tool.args_schema, "model_json_schema"):
            input_schema = tool.args_schema.model_json_schema()
        elif tool.args_schema and hasattr(tool.args_schema, "schema"):
            input_schema = tool.args_schema.schema()
        elif hasattr(tool, "args") and isinstance(tool.args, dict):
            input_schema = {"type": "object", "properties": tool.args}
        else:
            input_schema = {}

        schema_dict = {
            "name": tool.name,
            "description": tool.description or f"MCP tool: {tool.name}",
            "inputSchema": input_schema,
        }
        tool_schemas.append(schema_dict)

    logger.info(f"Discovered {len(tool_schemas)} tools from {len(config_json)} servers")
    return tool_schemas


# ── Function 3: Build Schema Tools ───────────────────────────────────────

def build_schema_tools(tool_schemas: list[dict]) -> list[StructuredTool]:
    """
    Reconstruct schema-only StructuredTool objects from cached JSON schemas.
    Used in agent_node for bind_tools(). Zero network calls.
    
    Reuses _build_schema_only_tool() from remote_mcp_client_config.py
    which handles JSON Schema → Pydantic model conversion.
    """
    tools = []
    for schema in tool_schemas:
        tool = _build_schema_only_tool(
            name=schema["name"],
            description=schema["description"],
            input_schema=schema.get("inputSchema", {}),
        )
        if tool is not None:
            tools.append(tool)
        else:
            logger.warning(f"Failed to build schema tool for '{schema.get('name', 'unknown')}'")

    logger.info(f"Built {len(tools)} schema-only tools from cache")
    return tools

async def reload_and_cache_user_mcp(user_id: str) -> list[dict]:
    """
    DB → discover → re-cache. Called on Redis schema miss.
    Returns schema list or empty list on any failure.
    """
    try:
        async with async_session_maker() as db:
            stmt = select(UserMCPConfig.config_json).where(
                UserMCPConfig.user_id == user_id
            )
            result = await db.execute(stmt)
            config_json = result.scalar_one_or_none()

        if not config_json:
            return []

        schemas = await discover_tools(config_json)
        version = hashlib.md5(
            json.dumps(config_json, sort_keys=True).encode()
        ).hexdigest()

        await set_cached_mcp_config(user_id, config_json)
        await set_cached_mcp_tool_schemas(user_id, schemas)
        await set_mcp_config_version(user_id, version)

        return schemas
    except Exception as e:
        logger.warning(f"reload_and_cache_user_mcp failed for {user_id}: {e}")
        return []