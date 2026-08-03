"""
=======================
Stateless helpers used by tool_node and agent_node.
No graph logic here — pure async utility functions.
"""

from langchain_core.messages import ToolMessage
from sqlalchemy import select

from db.models import UserConnection, async_session_maker
from services.cache import cache_token, get_cached_token
from services.connections.oauth_engine import OAuthEngine
from utils.logger import get_logger

logger = get_logger(__name__)
oauth_engine = OAuthEngine()


async def fetch_connections_from_db(user_id: str) -> set[str]:
    """
    Direct DB query for connected provider names.
    Called when Redis cache misses in agent_node.
    Returns empty set on any failure — caller handles gracefully.
    """
    try:
        async with async_session_maker() as db:
            stmt = select(UserConnection.provider).where(
                UserConnection.user_id == user_id
            )
            result = await db.execute(stmt)
            return {row[0] for row in result.all()}
    except Exception as e:
        logger.error(f"DB connection fetch failed for user {user_id}: {e}")
        return set()


async def resolve_token(user_id: str, provider: str) -> str | None:
    """
    Cache-first token resolution. Never raises — returns None on any failure.

    Priority:
      1. Redis cache (300s TTL) — avoids DB round-trip on every tool call
      2. OAuthEngine.get_valid_token() — handles DB fetch + refresh if expired

    Why catch all exceptions and not just ValueError?
      OAuthEngine.get_valid_token() calls refresh_if_expired() which makes
      an httpx request. That can raise TimeoutError, ConnectError, or any
      httpx exception in addition to ValueError (no connection found).
      All failures have the same outcome here — no valid token, return None.
      Caller decides what to do (interrupt, error message, etc).
    """
    # 1. Redis
    try:
        token = await get_cached_token(user_id, provider)
        if token:
            return token
    except Exception as e:
        logger.warning(f"Cache read failed for {provider}/{user_id}: {e}")

    # 2. OAuthEngine
    try:
        token = await oauth_engine.get_valid_token(user_id, provider)
        try:
            await cache_token(user_id, provider, token)
        except Exception as e:
            # Cache write failure is non-fatal — token is still valid
            logger.warning(f"Cache write failed for {provider}/{user_id}: {e}")
        return token
    except ValueError:
        # No connection exists for this user+provider — expected path
        return None
    except Exception as e:
        # Network error, DB error, refresh failure
        logger.error(
            f"Token resolution failed for {provider}/{user_id}: "
            f"{type(e).__name__}: {e}"
        )
        return None


def cancel_all(tool_calls: list) -> dict:
    """
    Build cancel ToolMessages for every pending tool call.
    Called when user responds 'cancel' to an auth or rate limit interrupt.
    Agent receives these as tool results and responds naturally:
    'OK, I've cancelled that. Is there anything else?'
    """
    return {
        "messages": [
            ToolMessage(
                content="Action cancelled by user.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }