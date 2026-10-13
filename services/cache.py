"""
Centralized Redis service — single connection pool, shared across all modules.

Design: Cache is "dumb" (Option A). Methods only get/set/delete.
They never call OAuthEngine or any business logic internally.
Callers own fallback logic. This keeps cache.py dependency-free.
"""

import redis.asyncio as aioredis
import json
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings= get_settings()

# Connection Pool — one per process, lazy-initialized

_redis_client: aioredis.Redis | None = None

async def get_redis() -> aioredis.Redis:
    """
    Returns the shared Redis client, creating it on first call.

    Why these specific settings:
    - decode_responses=True: gives us str instead of bytes everywhere,
      so we never need .decode() calls scattered through the codebase.
    - max_connections=20: sized for 2-worker Gunicorn on 4GB VM.
      Each worker gets its own pool (forked process), so total is ~40.
      Redis default max clients is 10,000 — we're nowhere near that.
    - socket_connect_timeout=5: fail fast if Redis is down on startup
      rather than hanging indefinitely.
    - socket_timeout=5: any single Redis command that takes >5s is broken,
      don't let it block the event loop.
    - retry_on_timeout=True: transient network blips between app container
      and Redis container shouldn't crash requests.
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
            encoding="utf-8",
            max_connections=20,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
    return _redis_client

# ---------------------------------------------------------------------------
# CacheService — stateless, all methods are module-level async functions.
#
# Why not a class? There's no instance state. Every method just calls
# get_redis() and operates on keys. A class would add __init__ boilerplate
# and force callers to manage an instance for zero benefit. Module-level
# functions are simpler and equally testable (mock get_redis).
# ---------------------------------------------------------------------------
# ── Token Cache ───────────────────────────────────────────────────────────
# Key: token:{user_id}:{provider}
# Value: the raw access token string
# TTL: 300s (5 min) — short because tokens are sensitive. If a token is
#       revoked or refreshed, worst case the stale cached version is used
#       for at most 5 minutes. For OAuth tokens that expire in 1 hour,
# 

async def get_cached_token(user_id: str, provider: str) -> str | None:
    """
    Returns the cached access token, or None on miss.
    Caller is responsible for DB fallback on None.
    """
    redis = await get_redis()
    token = await redis.get(f"token:{user_id}:{provider}")
    if token:
        logger.debug(f"Token cache HIT for {provider}, user {user_id}")
    return token


async def cache_token(user_id: str, provider:str, token: str, ttl: int=300) -> None:
    """
    Cache an access token with TTL.

    Why 300s default: balances between reducing DB load (most chat sessions
    involve multiple tool calls within minutes) and limiting exposure window
    if a token is compromised. The token is encrypted at rest in Postgres
    anyway — this is just an ephemeral hot copy.
   """
    redis = await get_redis()
    await redis.set(f"token:{user_id}:{provider}", token, ex=ttl)
    logger.debug(f"Cached token for {provider}, user {user_id}, TTL={ttl}s")


async def invalidate_token(user_id: str, provider: str) -> None:
    """
    Delete a cached token. Called after:
    - OAuth disconnect (revoke)
    - OAuth reconnect (new token replaces old)
    - Token refresh (old token is now invalid)
    """
    redis = await get_redis()
    await redis.delete(f"token:{user_id}:{provider}")
    logger.debug(f"Invalidated token cache for {provider}, user {user_id}")

# ── Connection Status Cache ───────────────────────────────────────────────
# Key: user_conns:{user_id}
# Value: JSON-encoded list of connected provider names, e.g. ["github","google"]
# TTL: 300s
#
# Why cache this? The agent_node needs to know which providers are connected
# on EVERY turn to build the system prompt. Without caching, that's a
# Postgres SELECT on every single message. With caching, it's a Redis GET
# (~0.1ms) instead of a Postgres round-trip (~5-20ms to Neon Singapore).

async def get_user_connections(user_id:str) -> set[str] | None:
    """
    Returns set of connected provider names, or None on cache miss.

    Why a set? Membership checks ("is github connected?") are O(1) with sets.
    We JSON-serialize as a list because JSON has no set type, then convert
    back on read.
    """
    redis = await get_redis()
    raw =  await redis.get(f"user_conns:{user_id}")
    if raw is None:
        return None
    
    try:
        return set(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
            logger.warning(f"Corrupt connection cache for user {user_id}, returning None")
            return None

async def set_user_connections(user_id:str, providers: set[str], ttl: int=300) -> None:
    """
    Cache the set of connected providers for a user.
    Called after a DB fetch in agent_node (cache-aside pattern).
    """
    redis = await get_redis()
    await redis.set(
        f"user_conns:{user_id}", 
        json.dumps(sorted(providers)), # sorted for deterministic serialization
        ex=ttl,
    )

async def invalidate_user_connections(user_id: str) -> None:
    """
    Delete cached connection set. Called after:
    - OAuth connect (callback success)
    - OAuth disconnect (revoke)
    - API key stored
    Next agent_node call rebuilds from DB.
    """
    redis = await get_redis()
    await redis.delete(f"user_conns:{user_id}")
    logger.debug(f"Invalidated connection cache for user {user_id}")

# ── OAuth CSRF State ──────────────────────────────────────────────────────
# Key: oauth_state:{state_token}
# Value: user_id
# TTL: 600s (10 min)
#
# This REPLACES the _pending_states dict that was in oauth_engine.py.
# The old in-memory dict broke under multi-worker Gunicorn because
# Worker A generates the state, Worker B handles the callback — different
# memory spaces. Redis is shared across all workers.

async def store_oauth_state(state:str, user_id:str, session_id: str | None = None, ttl:int=600) -> None:
    """
    Store CSRF state for OAuth flow. 10 min TTL gives users enough time
    to authenticate with the provider without the state expiring.
    """
    redis = await get_redis()
    payload = json.dumps({"user_id": user_id, "session_id": session_id})
    await redis.set(f"oauth_state:{state}", payload, ex=ttl)

async def pop_oauth_state(state:str) -> str | None:
    """
    Atomically fetch AND delete the OAuth state.
    Why atomic (pipeline with transaction=True)?
    If two callbacks arrive simultaneously with the same state token
    (replay attack or double-click), only one should succeed. Without
    atomicity, both could GET the user_id before either DELETEs it.
    The pipeline ensures GET+DEL happen as one atomic unit.
    Returns user_id if valid, None if expired/not found.
    """
    redis = await get_redis()
    key = f"oauth_state:{state}"
    async with redis.pipeline(transaction=True) as pipe:
        await pipe.get(key)
        await pipe.delete(key)
        results = await pipe.execute()
    
    raw =  results[0]
    if not raw:
        return None, None
    
    try:
        data = json.loads(raw)
        return data.get("user_id"), data.get("session_id")
    except json.JSONDecodeError:
        return raw, None


#if neede in future for tool result or llm response
# async def cache_tool_result(user_id: str, cache_key: str, result: str, ttl: int = 120):
#     redis = await get_redis()
#     await redis.set(f"tool_result:{user_id}:{cache_key}", result, ex=ttl)


# ── MCP Config Cache ─────────────────────────────────────────────────────
# Three keys per user:
#   mcp_config_version:{user_id} — MD5 hash of config_json (no TTL, explicit invalidation)
#   mcp_config:{user_id}         — the raw config_json dict (TTL 3600s)
#   mcp_tool_schemas:{user_id}   — discovered tool schemas list (TTL 3600s)

async def get_mcp_config_version(user_id: str) -> str | None:
    """Returns the MD5 hash of the user's MCP config, or None if no config exists."""
    redis = await get_redis()
    return await redis.get(f"mcp_config_version:{user_id}")


async def set_mcp_config_version(user_id: str, version_hash: str) -> None:
    """Store the config version hash. No TTL — only cleared on config change or deletion."""
    redis = await get_redis()
    await redis.set(f"mcp_config_version:{user_id}", version_hash)
    logger.debug(f"Set MCP config version for user {user_id}")


async def invalidate_mcp_config(user_id: str) -> None:
    """Delete ALL MCP cache keys for a user. Called on config update or deletion."""
    redis = await get_redis()
    await redis.delete(
        f"mcp_config_version:{user_id}",
        f"mcp_config:{user_id}",
        f"mcp_tool_schemas:{user_id}",
    )
    from services.tool_retriever import tool_retriever_service
    tool_retriever_service.invalidate_user_mcp_cache(user_id)
    from services.vector_tool_retriever import vector_tool_retriever_service
    vector_tool_retriever_service.invalidate_user_cache(user_id)
    logger.debug(f"Invalidated all MCP cache for user {user_id}")


async def get_cached_mcp_config(user_id: str) -> dict | None:
    """Returns the cached config_json dict, or None on miss."""
    redis = await get_redis()
    raw = await redis.get(f"mcp_config:{user_id}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Corrupt MCP config cache for user {user_id}")
        return None


async def set_cached_mcp_config(user_id: str, config: dict, ttl: int | None = None) -> None:
    """Cache the config_json dict permanently (or with optional TTL)."""
    redis = await get_redis()
    if ttl:
        await redis.set(f"mcp_config:{user_id}", json.dumps(config), ex=ttl)
    else:
        await redis.set(f"mcp_config:{user_id}", json.dumps(config))


async def get_cached_mcp_tool_schemas(user_id: str) -> list[dict] | None:
    """Returns the cached tool schemas list, or None on miss."""
    redis = await get_redis()
    raw = await redis.get(f"mcp_tool_schemas:{user_id}")
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Corrupt MCP tool schemas cache for user {user_id}")
        return None


async def set_cached_mcp_tool_schemas(user_id: str, schemas: list[dict], ttl: int | None = None) -> None:
    """Cache the discovered tool schemas permanently (or with optional TTL)."""
    redis = await get_redis()
    if ttl:
        await redis.set(f"mcp_tool_schemas:{user_id}", json.dumps(schemas), ex=ttl)
    else:
        await redis.set(f"mcp_tool_schemas:{user_id}", json.dumps(schemas))
