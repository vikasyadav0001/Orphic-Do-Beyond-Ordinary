"""
Two-layer rate limiting for Orphic.

Layer 1 — Tool/Provider Rate Limiting (per_minute + per_hour)
  Protects external API quotas (GitHub, Gmail, Notion, n8n).
  Consumed by tool_node before every MCP tool execution.
  Consumed by n8n_node (Phase 6) before every workflow trigger.

Layer 2 — Agent/Chat Rate Limiting (requests + tokens)
  Protects YOUR LLM costs and VM from runaway usage.
  Consumed by chat_router BEFORE graph invocation (requests).
  Consumed by chat_router AFTER stream completes (token logging).

Both layers use Redis sliding windows for accuracy.
No fixed-boundary reset problem. Atomic via Redis pipelines.
"""

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from services.cache import get_redis
from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LAYER 1 — Tool/Provider Rate Limit Configuration
# ---------------------------------------------------------------------------
# Defaults per provider, based on each provider's own API limits with headroom.
# GitHub: 5,000/hour authenticated → we allow 500 (10%)
# Gmail: 250 quota units/sec → we allow 10/min, 100/hour (conservative)
# Notion: 3 req/sec → we allow 20/min, 300/hour
# n8n: self-hosted, cap to prevent runaway workflows
#
# Override per-user via future user_rate_limits table (premium tiers).
# ---------------------------------------------------------------------------

DEFAULT_RATE_LIMITS: dict[str, dict[str, int]] = {
    "github":  {"per_minute": 30,  "per_hour": 500},
    "google":  {"per_minute": 10,  "per_hour": 100},
    "notion":  {"per_minute": 20,  "per_hour": 300},
    "slack":   {"per_minute": 20,  "per_hour": 200},
    "n8n":     {"per_minute": 10,  "per_hour": 100},
    "default": {"per_minute": 20,  "per_hour": 200},
}

WINDOW_SIZES = {
    "minute": 60,
    "hour":   3600,
    "day":    86400,
}


# ---------------------------------------------------------------------------
# LAYER 2 — Agent/Chat Rate Limit Configuration
# ---------------------------------------------------------------------------
# These protect YOUR costs, not provider APIs.
# Tier-based: free gets conservative limits, pro gets higher.
# Token limits are daily (simple counter with midnight TTL — sliding window
# is overkill for daily token budgets and adds unnecessary Redis complexity).
# Request limits are sliding window (same as tool limits).
#
# Limits as specified:
#   requests_per_hour: 10   (how many turns per hour)
#   requests_per_day:  30   (how many turns per day)
#   tokens_per_hour:   100,000
#   tokens_per_day:    100,000  (same — day budget is the real constraint)
# ---------------------------------------------------------------------------

DEFAULT_CHAT_LIMITS: dict[str, dict[str, int]] = {
    "free": {
        "requests_per_hour": 10,
        "requests_per_day":  30,
        "tokens_per_hour":   500000,
        "tokens_per_day":    1000000,
    },
    "pro": {
        "requests_per_hour": 30,
        "requests_per_day":  100,
        "tokens_per_hour":   5000000,
        "tokens_per_day":    5000000,
    },
    "default": "free",  # tier assigned to new users
}


# ---------------------------------------------------------------------------
# Result Schemas
# ---------------------------------------------------------------------------

@dataclass
class RateLimitResult:
    """
    Result from check_and_increment() — Layer 1 (tool/provider limits).

    Fields:
      allowed:        whether the request should proceed
      current_minute: requests in the current 60s window
      current_hour:   requests in the current 3600s window
      limit_minute:   configured per-minute cap
      limit_hour:     configured per-hour cap
      retry_after:    seconds until a slot opens (None if allowed)
    """
    allowed: bool
    current_minute: int
    current_hour: int
    limit_minute: int
    limit_hour: int
    retry_after: int | None


@dataclass
class ChatRateLimitResult:
    """
    Result from check_chat_rate() — Layer 2 (agent usage limits).

    Separate from RateLimitResult because it tracks different dimensions:
    requests (sliding window) AND tokens (daily counter).
    Both are shown on the frontend dashboard.

    Fields:
      allowed:            whether the chat turn should proceed
      denied_reason:      which limit was hit ('requests_hour', 'requests_day',
                          'tokens_hour', 'tokens_day') — None if allowed
      requests_this_hour: turn count in current hour window
      requests_today:     turn count today
      tokens_this_hour:   tokens burned this hour
      tokens_today:       tokens burned today
      limit_requests_hour: configured hourly request cap
      limit_requests_day:  configured daily request cap
      limit_tokens_hour:   configured hourly token cap
      limit_tokens_day:    configured daily token cap
      retry_after:        seconds until reset (None if allowed)
    """
    allowed: bool
    denied_reason: str | None
    requests_this_hour: int
    requests_today: int
    tokens_this_hour: int
    tokens_today: int
    limit_requests_hour: int
    limit_requests_day: int
    limit_tokens_hour: int
    limit_tokens_day: int
    retry_after: int | None


# ---------------------------------------------------------------------------
# LAYER 1 — Tool/Provider Rate Limiter
# ---------------------------------------------------------------------------

def _get_provider_limits(provider: str) -> dict[str, int]:
    """Resolve limits for a provider, fall back to default."""
    return DEFAULT_RATE_LIMITS.get(provider, DEFAULT_RATE_LIMITS["default"])


def _get_chat_limits(tier: str) -> dict[str, int]:
    """Resolve chat limits for a tier, fall back to free."""
    resolved = DEFAULT_CHAT_LIMITS.get(tier, DEFAULT_CHAT_LIMITS["default"])
    if isinstance(resolved, str):
        # Handle "default": "free" indirection
        resolved = DEFAULT_CHAT_LIMITS[resolved]
    return resolved


def _today_key_suffix() -> str:
    """
    Returns UTC date string for daily counter keys.
    Using UTC consistently — never local time — prevents midnight
    ambiguity across timezones and server restarts.
    Example: '2026-07-08'
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _hour_key_suffix() -> str:
    """
    Returns UTC year-month-day-hour string for hourly token counter keys.
    Example: '2026-07-08-14'
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")


async def check_and_increment(user_id: str, provider: str) -> RateLimitResult:
    """
    Atomically check + record a tool API call in both minute and hour windows.

    Pipeline ensures prune-count-add is indivisible — no race conditions
    under concurrent Gunicorn workers hitting the same user+provider.

    Returns RateLimitResult with allowed=True/False and usage stats.
    Consumed by tool_node (Phase 3) and n8n_node (Phase 6).
    """
    redis = await get_redis()
    limits = _get_provider_limits(provider)
    now = time.time()
    member = f"{now}:{uuid.uuid4().hex[:8]}"

    # ── Minute window ────────────────────────────────────────────────
    minute_key = f"ratelimit:{user_id}:{provider}:minute"
    minute_window_start = now - WINDOW_SIZES["minute"]

    async with redis.pipeline(transaction=True) as pipe:
        await pipe.zremrangebyscore(minute_key, 0, minute_window_start)
        await pipe.zcard(minute_key)
        minute_results = await pipe.execute()

    minute_count = minute_results[1]

    if minute_count >= limits["per_minute"]:
        oldest = await redis.zrange(minute_key, 0, 0, withscores=True)
        retry_after = int(WINDOW_SIZES["minute"] - (now - oldest[0][1])) + 1 if oldest else WINDOW_SIZES["minute"]
        logger.warning(f"Tool rate limit DENIED (minute): user={user_id} provider={provider} {minute_count}/{limits['per_minute']}")
        return RateLimitResult(
            allowed=False,
            current_minute=minute_count,
            current_hour=0,
            limit_minute=limits["per_minute"],
            limit_hour=limits["per_hour"],
            retry_after=max(retry_after, 1),
        )

    # ── Hour window ──────────────────────────────────────────────────
    hour_key = f"ratelimit:{user_id}:{provider}:hour"
    hour_window_start = now - WINDOW_SIZES["hour"]

    async with redis.pipeline(transaction=True) as pipe:
        await pipe.zremrangebyscore(hour_key, 0, hour_window_start)
        await pipe.zcard(hour_key)
        hour_results = await pipe.execute()

    hour_count = hour_results[1]

    if hour_count >= limits["per_hour"]:
        oldest = await redis.zrange(hour_key, 0, 0, withscores=True)
        retry_after = int(WINDOW_SIZES["hour"] - (now - oldest[0][1])) + 1 if oldest else WINDOW_SIZES["hour"]
        logger.warning(f"Tool rate limit DENIED (hour): user={user_id} provider={provider} {hour_count}/{limits['per_hour']}")
        return RateLimitResult(
            allowed=False,
            current_minute=minute_count,
            current_hour=hour_count,
            limit_minute=limits["per_minute"],
            limit_hour=limits["per_hour"],
            retry_after=max(retry_after, 1),
        )

    # ── Both OK — record the request ─────────────────────────────────
    async with redis.pipeline(transaction=True) as pipe:
        await pipe.zadd(minute_key, {member: now})
        await pipe.expire(minute_key, WINDOW_SIZES["minute"] + 10)
        await pipe.zadd(hour_key, {member: now})
        await pipe.expire(hour_key, WINDOW_SIZES["hour"] + 60)
        await pipe.execute()

    logger.debug(f"Tool rate limit OK: user={user_id} provider={provider} minute={minute_count+1}/{limits['per_minute']} hour={hour_count+1}/{limits['per_hour']}")

    return RateLimitResult(
        allowed=True,
        current_minute=minute_count + 1,
        current_hour=hour_count + 1,
        limit_minute=limits["per_minute"],
        limit_hour=limits["per_hour"],
        retry_after=None,
    )


async def get_usage(user_id: str, provider: str) -> dict:
    """
    Read-only usage stats for a provider. Safe to call anytime.
    Used for frontend dashboards and admin monitoring.
    """
    redis = await get_redis()
    limits = _get_provider_limits(provider)
    now = time.time()
    minute_key = f"ratelimit:{user_id}:{provider}:minute"
    hour_key = f"ratelimit:{user_id}:{provider}:hour"

    async with redis.pipeline(transaction=True) as pipe:
        await pipe.zremrangebyscore(minute_key, 0, now - WINDOW_SIZES["minute"])
        await pipe.zremrangebyscore(hour_key, 0, now - WINDOW_SIZES["hour"])
        await pipe.zcard(minute_key)
        await pipe.zcard(hour_key)
        results = await pipe.execute()

    return {
        "provider": provider,
        "minute": {"current": results[2], "limit": limits["per_minute"]},
        "hour":   {"current": results[3], "limit": limits["per_hour"]},
    }


async def reset(user_id: str, provider: str) -> None:
    """
    Admin-only reset of tool rate limit counters.
    NOT exposed as a user-facing API.
    """
    redis = await get_redis()
    await redis.delete(
        f"ratelimit:{user_id}:{provider}:minute",
        f"ratelimit:{user_id}:{provider}:hour",
    )
    logger.info(f"Tool rate limit reset: user={user_id} provider={provider}")


# ---------------------------------------------------------------------------
# LAYER 2 — Agent/Chat Rate Limiter
# ---------------------------------------------------------------------------

async def check_chat_rate(user_id: str, tier: str = "free") -> ChatRateLimitResult:
    """
    Check if user can make another chat turn.
    Called in chat_router BEFORE graph invocation — deny before spending tokens.

    Checks four limits in priority order:
      1. requests_per_hour (sliding window — accurate)
      2. requests_per_day  (sliding window — accurate)
      3. tokens_per_hour   (hourly counter — resets on the hour)
      4. tokens_per_day    (daily counter — resets at UTC midnight)

    Does NOT increment request counters here — only checks.
    Request increment happens in log_token_usage() AFTER the turn completes,
    so failed/errored turns don't count against the user.
    """
    redis = await get_redis()
    limits = _get_chat_limits(tier)
    now = time.time()

    req_hour_key = f"chat_req:{user_id}:hour"
    req_day_key  = f"chat_req:{user_id}:day:{_today_key_suffix()}"
    tok_hour_key = f"chat_tok:{user_id}:hour:{_hour_key_suffix()}"
    tok_day_key  = f"chat_tok:{user_id}:day:{_today_key_suffix()}"

    # ── Read current request counts (sliding windows) ─────────────────
    async with redis.pipeline(transaction=True) as pipe:
        # Prune and count hourly request window
        await pipe.zremrangebyscore(req_hour_key, 0, now - WINDOW_SIZES["hour"])
        await pipe.zcard(req_hour_key)
        # Prune and count daily request window
        await pipe.zremrangebyscore(req_day_key, 0, now - WINDOW_SIZES["day"])
        await pipe.zcard(req_day_key)
        results = await pipe.execute()

    req_hour_count = results[1]
    req_day_count  = results[3]

    # ── Read current token counts (simple counters) ────────────────────
    async with redis.pipeline(transaction=False) as pipe:
        await pipe.get(tok_hour_key)
        await pipe.get(tok_day_key)
        tok_results = await pipe.execute()

    tok_hour_count = int(tok_results[0] or 0)
    tok_day_count  = int(tok_results[1] or 0)

    # ── Check limits in priority order ────────────────────────────────

    if req_hour_count >= limits["requests_per_hour"]:
        oldest = await redis.zrange(req_hour_key, 0, 0, withscores=True)
        retry_after = int(WINDOW_SIZES["hour"] - (now - oldest[0][1])) + 1 if oldest else WINDOW_SIZES["hour"]
        logger.warning(f"Chat rate limit DENIED (req/hour): user={user_id} {req_hour_count}/{limits['requests_per_hour']}")
        return ChatRateLimitResult(
            allowed=False,
            denied_reason="requests_hour",
            requests_this_hour=req_hour_count,
            requests_today=req_day_count,
            tokens_this_hour=tok_hour_count,
            tokens_today=tok_day_count,
            limit_requests_hour=limits["requests_per_hour"],
            limit_requests_day=limits["requests_per_day"],
            limit_tokens_hour=limits["tokens_per_hour"],
            limit_tokens_day=limits["tokens_per_day"],
            retry_after=max(retry_after, 1),
        )

    if req_day_count >= limits["requests_per_day"]:
        # Day window resets at UTC midnight — compute seconds until then
        now_utc = datetime.now(timezone.utc)
        seconds_until_midnight = (
            86400
            - now_utc.hour * 3600
            - now_utc.minute * 60
            - now_utc.second
        )
        logger.warning(f"Chat rate limit DENIED (req/day): user={user_id} {req_day_count}/{limits['requests_per_day']}")
        return ChatRateLimitResult(
            allowed=False,
            denied_reason="requests_day",
            requests_this_hour=req_hour_count,
            requests_today=req_day_count,
            tokens_this_hour=tok_hour_count,
            tokens_today=tok_day_count,
            limit_requests_hour=limits["requests_per_hour"],
            limit_requests_day=limits["requests_per_day"],
            limit_tokens_hour=limits["tokens_per_hour"],
            limit_tokens_day=limits["tokens_per_day"],
            retry_after=seconds_until_midnight,
        )

    if tok_hour_count >= limits["tokens_per_hour"]:
        now_utc = datetime.now(timezone.utc)
        seconds_until_next_hour = 3600 - now_utc.minute * 60 - now_utc.second
        logger.warning(f"Chat rate limit DENIED (tok/hour): user={user_id} {tok_hour_count}/{limits['tokens_per_hour']}")
        return ChatRateLimitResult(
            allowed=False,
            denied_reason="tokens_hour",
            requests_this_hour=req_hour_count,
            requests_today=req_day_count,
            tokens_this_hour=tok_hour_count,
            tokens_today=tok_day_count,
            limit_requests_hour=limits["requests_per_hour"],
            limit_requests_day=limits["requests_per_day"],
            limit_tokens_hour=limits["tokens_per_hour"],
            limit_tokens_day=limits["tokens_per_day"],
            retry_after=seconds_until_next_hour,
        )

    if tok_day_count >= limits["tokens_per_day"]:
        now_utc = datetime.now(timezone.utc)
        seconds_until_midnight = (
            86400
            - now_utc.hour * 3600
            - now_utc.minute * 60
            - now_utc.second
        )
        logger.warning(f"Chat rate limit DENIED (tok/day): user={user_id} {tok_day_count}/{limits['tokens_per_day']}")
        return ChatRateLimitResult(
            allowed=False,
            denied_reason="tokens_day",
            requests_this_hour=req_hour_count,
            requests_today=req_day_count,
            tokens_this_hour=tok_hour_count,
            tokens_today=tok_day_count,
            limit_requests_hour=limits["requests_per_hour"],
            limit_requests_day=limits["requests_per_day"],
            limit_tokens_hour=limits["tokens_per_hour"],
            limit_tokens_day=limits["tokens_per_day"],
            retry_after=seconds_until_midnight,
        )

    # ── All checks passed ─────────────────────────────────────────────
    logger.debug(f"Chat rate limit OK: user={user_id} tier={tier} req_hour={req_hour_count+1}/{limits['requests_per_hour']} req_day={req_day_count+1}/{limits['requests_per_day']}")

    return ChatRateLimitResult(
        allowed=True,
        denied_reason=None,
        requests_this_hour=req_hour_count,
        requests_today=req_day_count,
        tokens_this_hour=tok_hour_count,
        tokens_today=tok_day_count,
        limit_requests_hour=limits["requests_per_hour"],
        limit_requests_day=limits["requests_per_day"],
        limit_tokens_hour=limits["tokens_per_hour"],
        limit_tokens_day=limits["tokens_per_day"],
        retry_after=None,
    )


async def log_token_usage(user_id: str, tokens_used: int) -> None:
    """
    Record tokens burned by a completed chat turn AND increment request counters.
    Called in chat_router AFTER the stream completes successfully.

    Why after, not before?
      - We don't know output token count until the stream finishes
      - Failed/errored turns shouldn't count against the user

    Request counters use sliding window (sorted sets).
    Token counters use simple INCRBY (hourly and daily buckets).
    TTL on token keys auto-expires them at the right boundary.
    """
    redis = await get_redis()
    now = time.time()
    member = f"{now}:{uuid.uuid4().hex[:8]}"

    req_hour_key = f"chat_req:{user_id}:hour"
    req_day_key  = f"chat_req:{user_id}:day:{_today_key_suffix()}"
    tok_hour_key = f"chat_tok:{user_id}:hour:{_hour_key_suffix()}"
    tok_day_key  = f"chat_tok:{user_id}:day:{_today_key_suffix()}"

    # Compute seconds until next hour boundary (for hourly token key TTL)
    now_utc = datetime.now(timezone.utc)
    seconds_until_next_hour = 3600 - now_utc.minute * 60 - now_utc.second
    seconds_until_midnight = (
        86400
        - now_utc.hour * 3600
        - now_utc.minute * 60
        - now_utc.second
    )

    async with redis.pipeline(transaction=True) as pipe:
        # Increment request sliding windows
        await pipe.zadd(req_hour_key, {member: now})
        await pipe.expire(req_hour_key, WINDOW_SIZES["hour"] + 60)
        await pipe.zadd(req_day_key, {member: now})
        await pipe.expire(req_day_key, WINDOW_SIZES["day"] + 300)

        # Increment token counters
        # INCRBY is atomic — safe under concurrent workers
        await pipe.incrby(tok_hour_key, tokens_used)
        await pipe.expire(tok_hour_key, seconds_until_next_hour + 60)  # expires after this hour
        await pipe.incrby(tok_day_key, tokens_used)
        await pipe.expire(tok_day_key, seconds_until_midnight + 300)   # expires at UTC midnight

        await pipe.execute()

    logger.info(f"Token usage logged: user={user_id} tokens={tokens_used}")


async def get_chat_usage(user_id: str, tier: str = "free") -> dict:
    """
    Read-only chat usage stats for the current user.
    Shown on the frontend dashboard: "You've used X/30 messages today"

    Returns all four counters with their limits — frontend decides what to display.
    """
    redis = await get_redis()
    limits = _get_chat_limits(tier)
    now = time.time()

    req_hour_key = f"chat_req:{user_id}:hour"
    req_day_key  = f"chat_req:{user_id}:day:{_today_key_suffix()}"
    tok_hour_key = f"chat_tok:{user_id}:hour:{_hour_key_suffix()}"
    tok_day_key  = f"chat_tok:{user_id}:day:{_today_key_suffix()}"

    async with redis.pipeline(transaction=False) as pipe:
        await pipe.zremrangebyscore(req_hour_key, 0, now - WINDOW_SIZES["hour"])
        await pipe.zcard(req_hour_key)
        await pipe.zremrangebyscore(req_day_key, 0, now - WINDOW_SIZES["day"])
        await pipe.zcard(req_day_key)
        await pipe.get(tok_hour_key)
        await pipe.get(tok_day_key)
        results = await pipe.execute()

    return {
        "requests": {
            "this_hour": {"current": results[1], "limit": limits["requests_per_hour"]},
            "today":     {"current": results[3], "limit": limits["requests_per_day"]},
        },
        "tokens": {
            "this_hour": {"current": int(results[4] or 0), "limit": limits["tokens_per_hour"]},
            "today":     {"current": int(results[5] or 0), "limit": limits["tokens_per_day"]},
        },
    }


async def reset_chat_limits(user_id: str) -> None:
    """
    Admin-only reset of all chat rate limit counters for a user.
    NOT exposed as a user-facing API.
    """
    redis = await get_redis()
    today = _today_key_suffix()
    hour = _hour_key_suffix()
    await redis.delete(
        f"chat_req:{user_id}:hour",
        f"chat_req:{user_id}:day:{today}",
        f"chat_tok:{user_id}:hour:{hour}",
        f"chat_tok:{user_id}:day:{today}",
    )
    logger.info(f"Chat rate limits reset: user={user_id}")


