"""
Vector-Based Tool Retrieval Service
=====================================

Production-grade semantic tool retrieval using:
- Embedding: nomic-ai/nomic-embed-text-v1.5 via FastEmbed (ONNX CPU, ~270MB RAM)
- Vector DB: Qdrant Embedded (Rust engine, ~80MB RAM, local disk persistence)

Architecture:
- Fixed OAuth tools (GitHub, Notion, Gmail, etc.) are embedded ONCE at startup.
- User dynamic MCP tools are embedded on-demand and cached per version_hash.
- All tools live in a single Qdrant collection with payload-based multi-tenant isolation.
- Queries are filtered by user_scope (global + user-specific) and provider (connected only).

This service is designed to be used by the `search_tools` meta-tool which the LLM
calls during the agent loop to discover relevant tools at runtime.
"""

import os
import hashlib
from typing import List, Dict, Set, Optional, Tuple

from langchain_core.tools import BaseTool, StructuredTool
from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — fail gracefully if dependencies not installed
# ---------------------------------------------------------------------------

_EMBEDDING_READY = False     # True once FastEmbed model is loaded
_QDRANT_READY = False        # True once Qdrant client is connected
_qdrant_client = None
_embedding_model = None

QDRANT_PERSIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "qdrant_tools"
)
COLLECTION_NAME = "mcp_tools"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384  # bge-small-en-v1.5 output dimension


def _ensure_embedding_model() -> bool:
    """
    Load the FastEmbed model only (no file locks).
    Safe to call at module import time from any process.
    """
    global _EMBEDDING_READY, _embedding_model

    if _EMBEDDING_READY:
        return True

    try:
        from fastembed import TextEmbedding
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME)
        _EMBEDDING_READY = True
        logger.info(f"[VectorToolRetriever] FastEmbed model '{EMBEDDING_MODEL_NAME}' loaded successfully")
        return True
    except ImportError as e:
        logger.warning(f"[VectorToolRetriever] fastembed not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"[VectorToolRetriever] FastEmbed load failed: {e}")
        return False


def _ensure_qdrant() -> bool:
    """
    Open Qdrant embedded client (acquires file lock).
    Only called from the worker process on first actual use — NOT at import time.
    This prevents the uvicorn --reload dual-process lock conflict.
    """
    global _QDRANT_READY, _qdrant_client

    if _QDRANT_READY:
        return True

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        os.makedirs(QDRANT_PERSIST_PATH, exist_ok=True)
        _qdrant_client = QdrantClient(path=QDRANT_PERSIST_PATH)

        # Ensure collection exists
        collections = [c.name for c in _qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            _qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"[VectorToolRetriever] Created Qdrant collection '{COLLECTION_NAME}'")

        _QDRANT_READY = True
        logger.info("[VectorToolRetriever] Qdrant client connected")
        return True
    except Exception as e:
        logger.error(f"[VectorToolRetriever] Qdrant init failed: {e}")
        return False


def _ensure_dependencies() -> bool:
    """Ensure both embedding model and Qdrant are ready."""
    return _ensure_embedding_model() and _ensure_qdrant()


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using FastEmbed."""
    if _embedding_model is None:
        return []
    # FastEmbed returns a generator, convert to list
    embeddings = list(_embedding_model.embed(texts))
    return [emb.tolist() for emb in embeddings]


def _build_tool_document(tool: BaseTool, provider_name: str = "") -> str:
    """
    Build a rich text document from a tool for embedding.
    Includes tool name, description, provider, and parameter names
    to maximize semantic coverage.
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

    # Build a comprehensive document for embedding
    parts = [tool.name.replace("_", " ")]
    if provider_name:
        parts.append(provider_name.replace("_", " "))
    if desc:
        parts.append(desc)
    if param_names:
        parts.append(param_names)

    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# VectorToolRetrieverService
# ---------------------------------------------------------------------------

class VectorToolRetrieverService:
    """
    Semantic vector-based tool retrieval service for Orphic's agent.

    Architecture:
    - Global fixed tools: embedded ONCE at startup, stored with user_scope="global"
    - Per-user MCP tools: embedded on config change, stored with user_scope="user_{id}"
    - Single Qdrant collection with payload-based filtering for multi-tenant isolation
    - Queries filtered by connected providers and user scope

    This service powers the `search_tools` meta-tool that the LLM calls
    during reasoning to discover relevant tools dynamically.
    """

    def __init__(self):
        # Track which tool sets have been indexed
        self._global_indexed: bool = False
        self._user_versions: Dict[str, str] = {}  # user_id -> version_hash

        # In-memory tool registries for mapping search results back to tool objects
        self._global_tool_registry: Dict[str, BaseTool] = {}       # tool_name -> tool
        self._global_provider_map: Dict[str, str] = {}             # tool_name -> provider
        self._user_tool_registries: Dict[str, Dict[str, BaseTool]] = {}  # user_id -> {tool_name -> tool}
        self._user_provider_maps: Dict[str, Dict[str, str]] = {}   # user_id -> {tool_name -> provider}

    # ------------------------------------------------------------------
    # Startup initialization — Fixed OAuth tools
    # ------------------------------------------------------------------

    def init_global_fixed_tools(
        self,
        schema_tools: List[StructuredTool],
        tool_provider_map: Dict[str, str]
    ) -> None:
        """
        Called at application startup after load_schemas() completes.

        Instantaneous (0.00s overhead): Populates in-memory tool registries.
        Embedding generation and Qdrant persistence are deferred to first use.
        """
        if self._global_indexed:
            return

        if not schema_tools:
            logger.warning("[VectorToolRetriever] No fixed schema tools provided")
            return

        # Store tool references in memory instantly (0s)
        self._global_tool_registry = {t.name: t for t in schema_tools}
        self._global_provider_map = dict(tool_provider_map)
        self._pending_schema_tools = schema_tools
        self._pending_provider_map = tool_provider_map

        logger.info(
            f"[VectorToolRetriever] Registered {len(schema_tools)} global fixed tools in memory "
            f"across {len(set(tool_provider_map.values()))} providers (vector indexing deferred)"
        )

    def _flush_global_to_qdrant(self) -> bool:
        """
        Phase 2: Check Qdrant disk storage for existing global tool vectors.
        - If Qdrant already has vectors stored from a previous run: SKIPS re-embedding completely (0s).
        - If Qdrant is empty: Generates embeddings ONCE and writes to disk.
        Called lazily on first search/index — runs ONLY in the worker process.
        """
        if self._global_indexed:
            return True

        if not hasattr(self, '_pending_schema_tools') or not self._pending_schema_tools:
            return False

        if not _ensure_qdrant():
            return False

        from qdrant_client.models import Filter, FieldCondition, MatchValue, PointStruct

        schema_tools = self._pending_schema_tools
        tool_provider_map = self._pending_provider_map

        # Check if Qdrant disk storage already has these global tool vectors indexed
        try:
            global_filter = Filter(
                must=[FieldCondition(key="user_scope", match=MatchValue(value="global"))]
            )
            count_result = _qdrant_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=global_filter,
                exact=True
            )
            if count_result.count == len(schema_tools):
                self._global_indexed = True
                self._pending_schema_tools = None
                logger.info(
                    f"[VectorToolRetriever] Loaded {count_result.count} existing global tool vectors "
                    f"from Qdrant disk storage (0 embeddings recomputed)"
                )
                return True
        except Exception as e:
            logger.debug(f"[VectorToolRetriever] Qdrant disk count check failed, will re-index: {e}")

        # If not indexed on disk, compute embeddings ONCE
        if not _ensure_embedding_model():
            logger.warning("[VectorToolRetriever] Cannot embed — FastEmbed unavailable")
            return False

        documents = []
        for tool in schema_tools:
            provider = tool_provider_map.get(tool.name, "")
            doc = _build_tool_document(tool, provider)
            documents.append(doc)

        logger.info(f"[VectorToolRetriever] Computing embeddings for {len(schema_tools)} global tools for first-time Qdrant disk storage...")
        embeddings = _embed_texts(documents)
        if not embeddings:
            logger.error("[VectorToolRetriever] Embedding generation failed for global tools")
            return False

        points = []
        for i, (tool, embedding) in enumerate(zip(schema_tools, embeddings)):
            provider = tool_provider_map.get(tool.name, "unknown")
            point_id = hashlib.md5(f"global_{tool.name}".encode()).hexdigest()
            int_id = int(point_id[:16], 16)

            points.append(PointStruct(
                id=int_id,
                vector=embedding,
                payload={
                    "tool_name": tool.name,
                    "provider": provider,
                    "user_scope": "global",
                    "description": (tool.description or "")[:500],
                    "document": documents[i][:500],
                }
            ))

        _qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        self._global_indexed = True
        self._pending_schema_tools = None  # Free memory

        logger.info(
            f"[VectorToolRetriever] Successfully indexed {len(schema_tools)} global tools "
            f"across {len(set(tool_provider_map.values()))} providers into Qdrant disk storage"
        )
        return True

    # ------------------------------------------------------------------
    # Per-user dynamic MCP tools
    # ------------------------------------------------------------------

    def index_user_mcp_tools(
        self,
        user_id: str,
        version_hash: str,
        user_mcp_tools: List[StructuredTool],
        server_names: List[str],
    ) -> None:
        """
        Index a user's dynamic MCP tools into Qdrant.
        Only re-indexes when version_hash changes (user added/removed a server).

        Args:
            user_id: User identifier
            version_hash: Hash of user's current MCP config
            user_mcp_tools: User's dynamic MCP tool objects
            server_names: List of server name keys from user's MCP config
        """
        if not _ensure_dependencies():
            return

        # Flush deferred global tool embeddings on first use
        self._flush_global_to_qdrant()

        # Check if already indexed with this version
        if self._user_versions.get(user_id) == version_hash:
            return

        if not user_mcp_tools:
            # Clean up old points if user removed all MCP servers
            self._delete_user_points(user_id)
            self._user_versions[user_id] = version_hash
            return

        from qdrant_client.models import PointStruct
        from services.tool_retriever import resolve_mcp_server_name

        # Delete old user points first
        self._delete_user_points(user_id)

        # Build documents and embed
        documents = []
        tool_provider_map: Dict[str, str] = {}
        for tool in user_mcp_tools:
            srv = resolve_mcp_server_name(tool.name, server_names)
            tool_provider_map[tool.name] = srv
            doc = _build_tool_document(tool, srv)
            documents.append(doc)

        embeddings = _embed_texts(documents)
        if not embeddings:
            logger.error(f"[VectorToolRetriever] Embedding failed for user {user_id} MCP tools")
            return

        # Build Qdrant points
        user_scope = f"user_{user_id}"
        points = []
        for i, (tool, embedding) in enumerate(zip(user_mcp_tools, embeddings)):
            provider = tool_provider_map.get(tool.name, "user_mcp")
            point_id = hashlib.md5(f"{user_scope}_{tool.name}".encode()).hexdigest()
            int_id = int(point_id[:16], 16)

            points.append(PointStruct(
                id=int_id,
                vector=embedding,
                payload={
                    "tool_name": tool.name,
                    "provider": provider,
                    "user_scope": user_scope,
                    "description": (tool.description or "")[:500],
                    "document": documents[i][:500],
                }
            ))

        # Upsert into Qdrant
        _qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        # Store tool references in memory
        self._user_tool_registries[user_id] = {t.name: t for t in user_mcp_tools}
        self._user_provider_maps[user_id] = tool_provider_map
        self._user_versions[user_id] = version_hash

        logger.info(
            f"[VectorToolRetriever] Indexed {len(user_mcp_tools)} MCP tools for user {user_id} "
            f"(version: {version_hash[:8]}..., servers: {server_names})"
        )

    def _delete_user_points(self, user_id: str) -> None:
        """Remove all existing Qdrant points for a specific user."""
        if not _qdrant_client:
            return

        from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

        try:
            _qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="user_scope",
                                match=MatchValue(value=f"user_{user_id}")
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            logger.warning(f"[VectorToolRetriever] Failed to delete user points: {e}")

    def invalidate_user_cache(self, user_id: str) -> None:
        """
        Explicitly evict a user's cached MCP embeddings.
        Call from MCP config router on add/remove/update.
        """
        self._delete_user_points(user_id)
        self._user_versions.pop(user_id, None)
        self._user_tool_registries.pop(user_id, None)
        self._user_provider_maps.pop(user_id, None)
        logger.info(f"[VectorToolRetriever] Evicted vector cache for user {user_id}")

    # ------------------------------------------------------------------
    # Semantic search — called by the search_tools meta-tool
    # ------------------------------------------------------------------

    def search_tools(
        self,
        query: str,
        user_id: str,
        connected_providers: Set[str],
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Dict]:
        """
        Perform semantic search over all available tools for this user.

        This is the core method called by the `search_tools` LangChain tool
        when the LLM wants to discover tools at runtime.

        Args:
            query: The LLM-generated search query (e.g., "github search repositories")
            user_id: User identifier for scoping
            connected_providers: Set of providers the user has OAuth connections for
            top_k: Maximum number of tools to return
            score_threshold: Minimum cosine similarity score (0.0 - 1.0)

        Returns:
            List of dicts with tool metadata: [{"tool_name": ..., "provider": ..., "description": ..., "score": ...}]
        """
        if not _ensure_dependencies() or not _qdrant_client:
            logger.warning("[VectorToolRetriever] Search unavailable — dependencies not loaded")
            return []

        # Flush deferred global tool embeddings on first use
        self._flush_global_to_qdrant()

        if not query or not query.strip():
            return []

        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        # Build query embedding
        query_embeddings = _embed_texts([query])
        if not query_embeddings:
            return []
        query_vector = query_embeddings[0]

        # Build scope filter: global + this user's MCP tools
        user_scope = f"user_{user_id}"
        scope_filter = Filter(
            must=[
                # Must be either global or this user's tools
                FieldCondition(
                    key="user_scope",
                    match=MatchAny(any=["global", user_scope])
                ),
            ]
        )

        # If we have connected providers, filter to only those + user MCP
        # User MCP tools don't have standard provider names, so we include all user-scoped
        if connected_providers:
            # Get all user MCP provider names for this user
            user_providers = set(self._user_provider_maps.get(user_id, {}).values())
            all_allowed_providers = connected_providers | user_providers

            scope_filter = Filter(
                must=[
                    FieldCondition(
                        key="user_scope",
                        match=MatchAny(any=["global", user_scope])
                    ),
                    FieldCondition(
                        key="provider",
                        match=MatchAny(any=list(all_allowed_providers))
                    ),
                ]
            )

        # Perform semantic search over a wider candidate pool for provider-diversity
        fetch_limit = max(top_k * 4, 32)
        try:
            response = _qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=scope_filter,
                limit=fetch_limit,
                score_threshold=score_threshold,
            )
            raw_hits = response.points
        except Exception as e:
            logger.error(f"[VectorToolRetriever] Qdrant search failed: {e}")
            return []

        # Format raw hits
        all_hits = []
        for hit in raw_hits:
            all_hits.append({
                "tool_name": hit.payload.get("tool_name", ""),
                "provider": hit.payload.get("provider", ""),
                "description": hit.payload.get("description", ""),
                "score": round(hit.score, 4) if hit.score is not None else 0.0,
            })

        # Dynamic Per-Provider Diversity Filtering
        # Prevents a single provider with many high-scoring tools from starving
        # other requested providers when the LLM makes a combined multi-intent query.
        max_per_provider = 2 if len(all_hits) > top_k else top_k
        provider_counts: Dict[str, int] = {}
        diverse_results: List[Dict] = []
        remaining_hits: List[Dict] = []

        for hit in all_hits:
            p = hit["provider"]
            cnt = provider_counts.get(p, 0)
            if cnt < max_per_provider:
                diverse_results.append(hit)
                provider_counts[p] = cnt + 1
            else:
                remaining_hits.append(hit)

        # Fill remaining slots up to top_k if needed
        if len(diverse_results) < top_k:
            for hit in remaining_hits:
                diverse_results.append(hit)
                if len(diverse_results) >= top_k:
                    break

        search_results = sorted(diverse_results, key=lambda x: x["score"], reverse=True)[:top_k]

        logger.info(
            f"[VectorToolRetriever] search query='{query[:60]}' | "
            f"user={user_id} | raw_hits={len(all_hits)} | diverse_hits={len(search_results)} | "
            f"providers={list(provider_counts.keys())}"
        )

        return search_results

    # ------------------------------------------------------------------
    # Resolve search results back to executable tool objects
    # ------------------------------------------------------------------

    def resolve_tools_by_names(
        self,
        tool_names: List[str],
        user_id: str,
    ) -> Tuple[List[BaseTool], Dict[str, str]]:
        """
        Given a list of tool names from search results, resolve them
        back to actual BaseTool objects and their provider mappings.

        Returns:
            Tuple of (list of tool objects, {tool_name: provider_name} map)
        """
        tools: List[BaseTool] = []
        provider_map: Dict[str, str] = {}

        for name in tool_names:
            # Check global registry first
            if name in self._global_tool_registry:
                tools.append(self._global_tool_registry[name])
                provider_map[name] = self._global_provider_map.get(name, "")
            # Then check user's MCP registry
            elif user_id in self._user_tool_registries and name in self._user_tool_registries[user_id]:
                tools.append(self._user_tool_registries[user_id][name])
                provider_map[name] = self._user_provider_maps.get(user_id, {}).get(name, "user_mcp")
            else:
                logger.warning(f"[VectorToolRetriever] Tool '{name}' not found in any registry")

        return tools, provider_map


# ---------------------------------------------------------------------------
# Module-level singleton — import and use directly
# ---------------------------------------------------------------------------
vector_tool_retriever_service = VectorToolRetrieverService()
