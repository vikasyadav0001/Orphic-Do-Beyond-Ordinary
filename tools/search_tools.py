"""
Search Tools — LLM-Controlled Tool Discovery Meta-Tool
========================================================

This meta-tool allows the LLM to search for relevant MCP tools at runtime
instead of relying on pre-retrieval. The LLM generates precise search queries
based on user intent and conversation context, achieving much higher retrieval
accuracy than raw user queries.

Flow:
1. LLM calls search_tools("github search list repositories")
2. This tool queries Qdrant vector store via VectorToolRetrieverService
3. Returns formatted tool descriptions to the LLM
4. Agent loop re-enters agent_node, which reads retrieved tool names from
   _last_search_results and dynamically binds them for the next LLM call
5. LLM then calls the actual tools natively (e.g. search_repositories(...))

Thread Safety:
- _last_search_results is keyed by (user_id, thread_id) to isolate concurrent sessions
"""

import json
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-session search result cache
# Key: (user_id, thread_id) → List[str] of tool_names from last search
# Read by agent_node to dynamically bind retrieved tools
# ---------------------------------------------------------------------------
_last_search_results: dict[tuple[str, str], list[str]] = {}


def get_last_search_results(user_id: str, thread_id: str) -> list[str]:
    """
    Called by agent_node to retrieve the tool names from the last
    search_tools invocation for this session.
    """
    return _last_search_results.get((user_id, thread_id), [])


def clear_search_results(user_id: str, thread_id: str) -> None:
    """Clear cached search results for a session."""
    _last_search_results.pop((user_id, thread_id), None)


@tool
async def search_tools(query: str, config: RunnableConfig) -> str:
    """
    Search for available tools to accomplish a task.

    Use this tool FIRST before attempting any action. Provide a clear,
    descriptive search query about what you need to do.

    Examples:
    - "github search list repositories" → finds GitHub repo tools
    - "gmail compose send email message" → finds Gmail send tools
    - "notion list search pages database" → finds Notion tools
    - "neon run sql query database" → finds Neon database tools
    - "firecrawl search web scrape" → finds Firecrawl tools

    Tips for best results:
    - Include service/provider names (github, notion, gmail, firecrawl, exa, supabase, etc.)
    - Include action verbs (search, list, create, update, delete, send)
    - IMPORTANT: For multi-task requests, make a SINGLE search_tools call combining all service names and tasks (e.g., "firecrawl exa search news supabase list tables notion list pages"). Do NOT make multiple parallel or sequential search calls.

    Args:
        query: Combined descriptive search query covering all services and actions needed.
    """
    user_id = config.get("configurable", {}).get("user_id", "")
    thread_id = config.get("configurable", {}).get("thread_id", "")

    if not query or not query.strip():
        return "Please provide a search query describing what tools you need."

    try:
        from services.vector_tool_retriever import vector_tool_retriever_service

        # Get connected providers from config (set by agent_node)
        connected_providers = set(
            config.get("configurable", {}).get("connected_providers", [])
        )

        # Perform semantic search (top_k=8 ensures multi-service requests return candidates for all providers)
        results = vector_tool_retriever_service.search_tools(
            query=query,
            user_id=user_id,
            connected_providers=connected_providers,
            top_k=8,
            score_threshold=0.3,
        )

        if not results:
            return (
                "No matching tools found for your query. Try:\n"
                "- Using the full service name (e.g., 'github' not 'gh')\n"
                "- Being more specific about the action needed\n"
                "- Checking if the service is connected"
            )

        # Cache the tool names for agent_node to pick up on re-entry (combine for parallel searches)
        existing = _last_search_results.get((user_id, thread_id), [])
        tool_names = [r["tool_name"] for r in results]
        combined = list(dict.fromkeys(existing + tool_names))
        _last_search_results[(user_id, thread_id)] = combined

        # Format results for the LLM
        lines = [f"Found {len(results)} relevant tool(s):\n"]
        for r in results:
            lines.append(
                f"  • {r['tool_name']} ({r['provider']}) "
                f"[score: {r['score']}]\n"
                f"    {r['description'][:200]}"
            )

        lines.append(
            "\nThese tools are now available for you to call directly. "
            "Proceed to use them as needed."
        )

        logger.info(
            f"[SearchTools] user={user_id} | query='{query[:60]}' | "
            f"found={len(results)} tools: {tool_names}"
        )

        return "\n".join(lines)

    except ImportError:
        logger.warning("[SearchTools] vector_tool_retriever not available, falling back")
        return (
            "Tool search is not available. "
            "The vector retrieval service is not configured."
        )
    except Exception as e:
        logger.error(f"[SearchTools] Search failed: {e}", exc_info=True)
        return f"Tool search failed: {str(e)}. Please try again."
