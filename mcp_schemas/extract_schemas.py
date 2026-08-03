# """
# One-off dev script — NOT production code.

# Connects to each MCP provider using dev tokens from .env,
# extracts tool schemas, and saves to mcp_schemas/schemas.json.

# Usage:
#     python mcp_schemas/extract_schemas.py

# Requires DEV tokens in .env:
#     DEV_GITHUB_TOKEN=ghp_xxx
#     DEV_GOOGLE_TOKEN=ya29.xxx

# Re-run when:
#     - A provider adds/removes/changes tools
#     - You add a new provider to PROVIDERS

# Output:
#     mcp_schemas/schemas.json
# """

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient
from services.connections.providers import PROVIDERS
from utils.logger import get_logger

logger = get_logger(__name__)


def _extract_tool_schema(tool) -> dict:
    """
    Robustly extract schema from a LangChain tool returned by get_tools().

    langchain-mcp-adapters returns tools with args_schema as a Pydantic
    BaseModel class. We need the raw JSON Schema dict to store in schemas.json.

    Tries multiple extraction paths in order of preference:
      1. model_json_schema()  — Pydantic v2 standard (preferred)
      2. schema()             — Pydantic v1 / v2 compat shim (deprecated but works)
      3. tool.args            — LangChain's own property (always available)
      4. {}                   — Empty fallback (tool still usable, just no args)

    Why not just use tool.args?
      tool.args returns a processed dict, not the raw JSON Schema. It loses
      field descriptions which the LLM needs for correct argument generation.
      model_json_schema() preserves descriptions.
    """
    args_schema = getattr(tool, "args_schema", None)

    if args_schema is not None:
        # Path 1: Pydantic v2 — preferred
        if hasattr(args_schema, "model_json_schema"):
            try:
                return args_schema.model_json_schema()
            except Exception as e:
                logger.warning(f"model_json_schema() failed for {tool.name}: {e}")

        # Path 2: Pydantic v1 / compat shim
        if hasattr(args_schema, "schema"):
            try:
                return args_schema.schema()
            except Exception as e:
                logger.warning(f"schema() failed for {tool.name}: {e}")

    # Path 3: LangChain's processed args dict — loses descriptions but safe
    if hasattr(tool, "args") and tool.args:
        try:
            logger.warning(
                f"Using tool.args fallback for {tool.name} — "
                f"field descriptions may be lost"
            )
            return {
                "type": "object",
                "properties": {
                    k: {"type": "string", "description": str(v)}
                    for k, v in tool.args.items()
                },
            }
        except Exception as e:
            logger.warning(f"tool.args fallback failed for {tool.name}: {e}")

    # Path 4: Empty schema — tool still loads, LLM calls it with no args
    logger.warning(f"No schema extracted for {tool.name} — using empty schema")
    return {"type": "object", "properties": {}}


# async def extract():
#     """
#     Connect to each provider's MCP server using dev tokens,
#     extract tool schemas via get_tools(), serialize to JSON.

#     get_tools() returns fully formed LangChain StructuredTool objects
#     with proper Pydantic args_schema already built by the adapter.
#     We extract the JSON Schema from those and store it.
#     """
#     schemas: dict[str, list[dict]] = {}
#     failed_providers: list[str] = []

#     for provider_name, provider in PROVIDERS.items():
#         if not provider.mcp_url or provider.mcp_url == "...":
#             logger.info(f"Skipping {provider_name} — no MCP URL configured")
#             continue

#         dev_token = os.getenv(f"DEV_{provider_name.upper()}_TOKEN")
#         if not dev_token:
#             logger.warning(
#                 f"Skipping {provider_name} — "
#                 f"no DEV_{provider_name.upper()}_TOKEN in .env"
#             )
#             continue

#         logger.info(f"Connecting to {provider_name} MCP server at {provider.mcp_url}...")

#         mcp_config = {
#             provider_name: {
#                 "transport": provider.mcp_transport,
#                 "url": provider.mcp_url,
#                 "headers": {
#                     provider.token_header: f"{provider.token_prefix} {dev_token}"
#                 },
#             }
#         }

#         try:
#             client = MultiServerMCPClient(mcp_config)
#             tools = await client.get_tools()

#             if not tools:
#                 logger.warning(f"No tools returned from {provider_name}")
#                 continue

#             provider_schemas = []
#             for tool in tools:
#                 input_schema = _extract_tool_schema(tool)
#                 schema_entry = {
#                     "name": tool.name,
#                     "description": tool.description or f"MCP tool: {tool.name}",
#                     "input_schema": input_schema,
#                 }
#                 provider_schemas.append(schema_entry)
#                 logger.debug(f"  Extracted: {tool.name} ({len(input_schema.get('properties', {}))} args)")

#             schemas[provider_name] = provider_schemas
#             logger.info(f"✓ {provider_name}: {len(provider_schemas)} tools extracted")

#         except Exception as e:
#             logger.error(f"✗ Failed to extract from {provider_name}: {e}", exc_info=True)
#             failed_providers.append(provider_name)
#             continue

#     # Guard: don't write empty file
#     if not schemas:
#         print("\n❌ No schemas extracted.")
#         print("Check: DEV tokens set? MCP URLs configured? Providers reachable?")
#         sys.exit(1)

#     output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schemas.json")
#     with open(output_path, "w") as f:
#         json.dump(schemas, f, indent=2, default=str)

#     total_tools = sum(len(v) for v in schemas.values())
#     print(f"\n✅ Extracted {total_tools} tools from {len(schemas)} providers")
#     print(f"   Providers: {', '.join(schemas.keys())}")
#     if failed_providers:
#         print(f"   ⚠ Failed: {', '.join(failed_providers)}")
#     print(f"📄 Output: {output_path}")


# if __name__ == "__main__":
    # asyncio.run(extract())

# import os
# from dotenv import load_dotenv
# load_dotenv()
# from services.connections.providers import PROVIDERS

# import asyncio
# import httpx
# import json
# from allowed_tools import ALLOWED_TOOLS
# from mcp.client.streamable_http import streamable_http_client
# from mcp import ClientSession


# all_schemas = {}

# for name, provider in PROVIDERS.items():
#     token = os.getenv(f"DEV_{name.upper()}_TOKEN")
#     print(f"======================={name.upper()}======================")
    
#     allowed = ALLOWED_TOOLS.get(name, set())
#     provider_tools = {}
    
#     async def extract(url, token, allowed):
#         async with httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}) as client:
#             async with streamable_http_client(url, http_client=client) as (read, write, _):
#                 async with ClientSession(read, write) as session:
#                     await session.initialize()
#                     result = await session.list_tools()
#                     print(f"Tools count: {len(result.tools)}")
#                     tools = {}
#                     for tool in result.tools:
#                         if tool.name in allowed:
#                             print(f"  ✓ {tool.name}")
#                             tools[tool.name] = {
#                                 "name": tool.name,
#                                 "description": tool.description,
#                                 "inputSchema": tool.inputSchema,
#                             }
#                         else:
#                             print(f"  ✗ {tool.name} (filtered)")
#                     return tools

#     provider_tools = asyncio.run(extract(provider.mcp_url, token, allowed))
#     all_schemas[name] = provider_tools

# with open("mcp_schemas/schemas.json", "w") as f:
#     json.dump(all_schemas, f, indent=2, default=str)