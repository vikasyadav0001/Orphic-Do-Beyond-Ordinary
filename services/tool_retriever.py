from typing import List, Dict, Tuple, Optional, Set
from langchain_core.tools import BaseTool, StructuredTool
from utils.logger import get_logger

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25Okapi = None
    BM25_AVAILABLE = False

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider alias map — used by the guarantee step to detect provider intent
# from natural language queries. Add aliases as new providers are onboarded.
# ---------------------------------------------------------------------------
PROVIDER_ALIASES: Dict[str, List[str]] = {
    "github": ["github", "gh", "repo", "repository", "pull request", "pr", "issue", "commit", "branch"],
    "notion": ["notion", "workspace", "pages", "page", "database", "notes", "docs"],
    "google_gmail": ["gmail", "email", "mail", "inbox", "message", "compose", "send email"],
    "google_sheets": ["sheets", "spreadsheet", "excel", "csv", "rows", "columns", "cells"],
    "google_calendar": ["calendar", "gcal", "event", "meeting", "schedule", "appointment"],
    "google_drive": ["drive", "gdrive", "file", "folder", "document", "upload"],
    "google_docs": ["docs", "document", "google doc", "writing"],
    "slack": ["slack", "channel", "message", "dm", "direct message", "notify", "post"],
    "atlassian": ["jira", "confluence", "atlassian", "ticket", "sprint", "board", "project"],
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def resolve_mcp_server_name(tool_name: str, server_names: List[str]) -> str:
    """
    Resolves which MCP server a tool belongs to by checking startswith
    against known server names sorted longest-first to prevent partial matches.
    
    Example: "google_sheets_read" correctly resolves to "google_sheets"
             not "google" (which would match first if unsorted).
    
    Returns "user_mcp" for unmatched tools rather than a naive split
    which would produce misleading provider tags.
    """
    sorted_servers = sorted(server_names, key=len, reverse=True)
    for server_name in sorted_servers:
        if tool_name.startswith(f"{server_name}_") or tool_name.startswith(f"{server_name}__"):
            return server_name
    return "user_mcp"


def build_tool_corpus_text(tool: BaseTool, provider_name: str = "") -> str:
    """
    Builds an expanded text corpus string for BM25 indexing.
    
    Expands beyond just tool name + description to include:
    - Provider name explicitly (so "notion" in query matches all Notion tools
      even if the tool description uses different verbs)
    - Parameter/field names (so "issue_number" in query matches github tools
      that accept issue_number as a parameter)
    
    This closes the semantic gap between user natural language and tool
    descriptions without requiring embedding models.
    """
    desc = (tool.description or "").strip()
    param_names = ""

    if hasattr(tool, "args_schema") and tool.args_schema:
        try:
            fields = (
                getattr(tool.args_schema, "model_fields", None)
                or getattr(tool.args_schema, "__fields__", None)
                or {}
            )
            param_names = " ".join(fields.keys())
        except Exception:
            pass

    return f"{tool.name} {desc} {provider_name} {param_names}".lower().strip()


# ---------------------------------------------------------------------------
# ToolRetrieverService
# ---------------------------------------------------------------------------

class ToolRetrieverService:
    """
    BM25-based tool retrieval service for Orphic's agent_node.

    Architecture:
    - Global fixed index: built ONCE at startup over all fixed OAuth MCP tools.
      Per-turn cost is only get_scores() — < 0.2ms.
    - Per-user dynamic index: built and cached in-process when a user's MCP
      config is first seen or changes (version hash mismatch). Subsequent
      turns reuse the cached index.
    - Connection scoping: fixed tools are pre-filtered to connected OAuth
      providers before scoring — unconnected provider tools never enter
      the candidate pool.
    - Provider guarantee: after top-K selection, alias matching ensures
      explicitly mentioned providers always have at least one tool included
      regardless of BM25 score.
    """

    def __init__(self):
        # Global fixed tool index — set at startup, never mutated after
        self._global_fixed_tools: List[StructuredTool] = []
        self._global_fixed_index: Optional[object] = None  # BM25Okapi or None
        self._global_tool_provider_map: Dict[str, str] = {}

        # Per-user dynamic MCP index cache
        # Format: user_id -> (version_hash, BM25Okapi|None, tools, tool->provider map)
        self._user_mcp_cache: Dict[
            str,
            Tuple[str, Optional[object], List[StructuredTool], Dict[str, str]]
        ] = {}

    # ------------------------------------------------------------------
    # Startup initialization
    # ------------------------------------------------------------------

    def init_global_fixed_tools(
        self,
        schema_tools: List[StructuredTool],
        tool_provider_map: Dict[str, str]
    ) -> None:
        """
        Called ONCE at application startup after load_schemas() completes.
        Builds the global BM25 index over all fixed OAuth remote MCP tools.
        
        Args:
            schema_tools: All fixed tool objects loaded from schemas.json
            tool_provider_map: Maps tool_name -> provider_name for fixed tools
        """
        self._global_fixed_tools = schema_tools
        self._global_tool_provider_map = tool_provider_map

        if not BM25_AVAILABLE:
            logger.warning(
                "[ToolRetriever] rank_bm25 not installed. "
                "BM25 pre-retrieval disabled — will fall back to connection-filtered binding. "
                "Install with: pip install rank-bm25"
            )
            return

        if not schema_tools:
            logger.warning("[ToolRetriever] No fixed schema tools provided — global index not built.")
            return

        corpus = [
            build_tool_corpus_text(tool, tool_provider_map.get(tool.name, "")).split()
            for tool in schema_tools
        ]
        self._global_fixed_index = BM25Okapi(corpus)
        logger.info(
            f"[ToolRetriever] Global BM25 index built over {len(schema_tools)} fixed schema tools at startup."
        )

    # ------------------------------------------------------------------
    # Per-user dynamic MCP index
    # ------------------------------------------------------------------

    def get_or_build_user_mcp_index(
        self,
        user_id: str,
        version_hash: str,
        user_mcp_tools: List[StructuredTool],
        server_names: List[str]
    ) -> Tuple[Optional[object], Dict[str, str]]:
        """
        Returns the cached BM25 index for this user's MCP tools.
        Rebuilds only when version_hash changes (user added/removed a server).
        
        Args:
            user_id: User identifier for cache keying
            version_hash: Hash of user's current MCP config — cache key
            user_mcp_tools: Current list of user's dynamic MCP tool objects
            server_names: List of server names from user's config keys
            
        Returns:
            Tuple of (BM25Okapi index or None, tool->server_name mapping dict)
        """
        if not user_mcp_tools:
            return None, {}

        cached = self._user_mcp_cache.get(user_id)
        if cached and cached[0] == version_hash:
            return cached[1], cached[3]

        # Build new index for this user
        tool_provider_map: Dict[str, str] = {}
        corpus: List[List[str]] = []

        for tool in user_mcp_tools:
            srv = resolve_mcp_server_name(tool.name, server_names)
            tool_provider_map[tool.name] = srv
            corpus.append(build_tool_corpus_text(tool, srv).split())

        bm25_index = BM25Okapi(corpus) if (BM25_AVAILABLE and corpus) else None
        self._user_mcp_cache[user_id] = (version_hash, bm25_index, user_mcp_tools, tool_provider_map)

        logger.info(
            f"[ToolRetriever] User MCP BM25 index built for user {user_id} "
            f"(hash={version_hash[:8]}..., tools={len(user_mcp_tools)}, servers={len(server_names)})"
        )
        return bm25_index, tool_provider_map

    def invalidate_user_mcp_cache(self, user_id: str) -> None:
        """
        Explicitly evicts a user's cached MCP index.
        Call this from the MCP config router on add/remove/update
        in addition to Redis cache invalidation.
        """
        if user_id in self._user_mcp_cache:
            del self._user_mcp_cache[user_id]
            logger.info(f"[ToolRetriever] Evicted in-process MCP index cache for user {user_id}")

    # ------------------------------------------------------------------
    # Main retrieval entry point
    # ------------------------------------------------------------------

    def retrieve_top_mcp_tools(
        self,
        query: str,
        connected_providers: Set[str],
        user_id: str,
        user_mcp_version: str,
        user_mcp_tools: List[StructuredTool],
        user_server_names: List[str],
        top_k: int = 12,
        min_score: float = 0.1
    ) -> List[BaseTool]:
        """
        Retrieves the top-K most relevant MCP tools for the given query.

        Pipeline:
        1. Connection scope: only fixed tools whose provider the user connected
        2. BM25 scoring: score query against global fixed index + per-user index
        3. Top-K selection: filter by min_score (default 0.1), sort by score, take top_k
        4. Provider guarantee: ensure explicitly mentioned providers have >= 1 tool

        Falls back to connection-filtered binding if BM25 is unavailable.
        
        Args:
            query: Latest user message content
            connected_providers: Set of provider names the user has OAuth connections for
            user_id: Used to look up per-user MCP cache
            user_mcp_version: Hash of user's MCP config — triggers index rebuild on change
            user_mcp_tools: User's dynamic MCP tool objects (from build_schema_tools)
            user_server_names: List of server name keys from user's MCP config
            top_k: Maximum tools to return (default 12, tuned dynamically in agent_node)
            min_score: Minimum BM25 score threshold to qualify a tool (default 0.1)
            
        Returns:
            List of BaseTool objects to be included in bind_tools()
        """
        # ------------------------------------------------------------------
        # BM25 unavailable — fallback to connection filter only
        # ------------------------------------------------------------------
        if not BM25_AVAILABLE or self._global_fixed_index is None:
            fixed_filtered = [
                tool for tool in self._global_fixed_tools
                if self._global_tool_provider_map.get(tool.name) in connected_providers
            ]
            result = (fixed_filtered + user_mcp_tools)[:top_k]
            logger.info(
                f"[ToolRetriever] BM25 unavailable — returning {len(result)} connection-filtered tools "
                f"(fixed={len(fixed_filtered)}, user_mcp={len(user_mcp_tools)}, capped at top_k={top_k})"
            )
            return result

        # ------------------------------------------------------------------
        # Empty query fallback — return top_k from connection-filtered pool
        # ------------------------------------------------------------------
        tokenized_query = query.lower().split() if query else []
        if not tokenized_query:
            fixed_filtered = [
                tool for tool in self._global_fixed_tools
                if self._global_tool_provider_map.get(tool.name) in connected_providers
            ]
            result = (fixed_filtered + user_mcp_tools)[:top_k]
            logger.info(
                f"[ToolRetriever] Empty query — returning {len(result)} connection-filtered tools (top_k={top_k})"
            )
            return result

        # ------------------------------------------------------------------
        # Step 1 + 2: Score fixed + user MCP tools
        # ------------------------------------------------------------------
        scored_candidates: List[Tuple[BaseTool, float, str]] = []

        # Fixed OAuth tools — connection-scoped then scored
        fixed_scores = self._global_fixed_index.get_scores(tokenized_query)
        for tool, score in zip(self._global_fixed_tools, fixed_scores):
            provider = self._global_tool_provider_map.get(tool.name, "")
            if provider in connected_providers:
                scored_candidates.append((tool, float(score), provider))

        # User dynamic MCP tools — scored via per-user cached index
        user_index, user_prov_map = self.get_or_build_user_mcp_index(
            user_id, user_mcp_version, user_mcp_tools, user_server_names
        )
        if user_index and user_mcp_tools:
            user_scores = user_index.get_scores(tokenized_query)
            for tool, score in zip(user_mcp_tools, user_scores):
                srv = user_prov_map.get(tool.name, "user_mcp")
                scored_candidates.append((tool, float(score), srv))

        if not scored_candidates:
            logger.info(f"[ToolRetriever] No candidates found for user {user_id} — returning empty list.")
            return []

        # ------------------------------------------------------------------
        # Step 3: Sort and take top-K (only for tools with BM25 score >= min_score)
        # ------------------------------------------------------------------
        relevant_candidates = [item for item in scored_candidates if item[1] >= min_score]
        relevant_candidates.sort(key=lambda x: x[1], reverse=True)

        selected_tools: List[BaseTool] = [item[0] for item in relevant_candidates[:top_k]]
        selected_providers: Set[str] = {item[2] for item in relevant_candidates[:top_k]}

        # ------------------------------------------------------------------
        # Step 4: Alias provider guarantee
        # Ensures providers explicitly mentioned in the query always have
        # at least one tool in the selection regardless of BM25 score.
        # Uses alias map to handle natural language variants:
        # "emails" -> google_gmail, "spreadsheet" -> google_sheets etc.
        # ------------------------------------------------------------------
        query_lower = query.lower()
        guarantee_additions = []

        for provider in connected_providers:
            if provider in selected_providers:
                continue
            aliases = PROVIDER_ALIASES.get(provider, [provider])
            if any(alias in query_lower for alias in aliases):
                # Find highest-scoring unselected tool for this provider
                for tool, score, p_name in scored_candidates:
                    if p_name == provider and tool not in selected_tools:
                        selected_tools.append(tool)
                        selected_providers.add(provider)
                        guarantee_additions.append(f"{tool.name} ({provider})")
                        break

        if guarantee_additions:
            logger.info(
                f"[ToolRetriever] Provider guarantee added {len(guarantee_additions)} tool(s): "
                f"{guarantee_additions}"
            )

        logger.info(
            f"[ToolRetriever] user={user_id} | "
            f"query='{query[:60]}...' | "
            f"candidates={len(scored_candidates)} | "
            f"selected={len(selected_tools)} | "
            f"top_k={top_k} | "
            f"providers={selected_providers}"
        )

        return selected_tools


# ---------------------------------------------------------------------------
# Module-level singleton — imported and used directly in graph.py
# ---------------------------------------------------------------------------
tool_retriever_service = ToolRetrieverService()