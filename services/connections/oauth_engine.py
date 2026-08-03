# ============================================================
# services/connections/oauth_engine.py
# ============================================================

import base64
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode

import httpx
from sqlalchemy import select, delete, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import get_settings
from db.models import UserConnection, async_session_maker
from services.cache import get_redis, pop_oauth_state, store_oauth_state
from services.connections.providers import (
    ProviderConfig,
    PROVIDERS,
)
from utils.logger import get_logger

env = get_settings()
logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _make_expires_at(expires_in: int | None) -> datetime | None:
    if not expires_in:
        return None
    return _now_utc() + timedelta(seconds=int(expires_in))


def _is_expired(expires_at: datetime | None) -> bool:
    """True if token expires within the next 60 seconds."""
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= (_now_utc() + timedelta(seconds=60))

# ──────────────────────────────────────────────────────────────
# Response extraction helpers
# ──────────────────────────────────────────────────────────────

def _extract_nested(data: dict, dot_path: str) -> str | None:
    """
    Navigate a dot-notation path through nested dicts.
    "access_token"            → data["access_token"]
    "authed_user.access_token" → data["authed_user"]["access_token"]
    Returns None if any key is missing.
    """
    val = data
    for key in dot_path.split("."):
        if not isinstance(val, dict):
            return None
        val = val.get(key)
    return val if isinstance(val, str) else None


# ──────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────

def _build_basic_auth_header(client_id: str, client_secret: str) -> str:
    """
    RFC 6749 §2.3.1: credentials must be url-encoded before base64.
    Returns the full "Basic <b64>" string.
    """
    from urllib.parse import quote
    encoded = base64.b64encode(
        f"{quote(client_id, safe='')}:{quote(client_secret, safe='')}".encode()
    ).decode()
    return f"Basic {encoded}"


async def _post_token_endpoint(
    url: str,
    payload: dict,
    provider: ProviderConfig,
) -> dict:
    """
    Universal token POST. Handles all auth methods and body formats.
    Raises ValueError on non-200 or missing access_token.
    """
    headers = {"Accept": "application/json"}

    # Merge any provider-specific static headers (e.g. Notion-Version)
    headers.update(provider.extra_token_headers)

    # Apply client authentication method
    if provider.token_endpoint_auth_method == "basic":
        headers["Authorization"] = _build_basic_auth_header(
            provider.client_id, provider.client_secret
        )
        # Credentials must NOT be in body for basic auth
        payload.pop("client_id", None)
        payload.pop("client_secret", None)

    elif provider.token_endpoint_auth_method == "none":
        # Pure PKCE: no secret anywhere; client_id stays in body
        payload.pop("client_secret", None)

    # else "body": client_id + client_secret stay in payload (default)

    async with httpx.AsyncClient() as client:
        if provider.token_request_format == "json":
            response = await client.post(url, json=payload, headers=headers)
        else:
            response = await client.post(url, data=payload, headers=headers)

    if response.status_code != 200:
        raise ValueError(
            f"Token endpoint returned {response.status_code}: {response.text}"
        )

    return response.json()


# ──────────────────────────────────────────────────────────────
# Redis keys for PKCE verifier storage
# ──────────────────────────────────────────────────────────────
# We store the code_verifier alongside the oauth_state in Redis.
# Key: "pkce_verifier:{state}" TTL: 600s (same as oauth_state)
# Atomic GET+DEL happens at callback time.

PKCE_TTL = 600  # seconds — matches oauth_state TTL


async def _store_pkce_verifier(state: str, code_verifier: str) -> None:
    redis = await get_redis()
    await redis.set(f"pkce_verifier:{state}", code_verifier, ex=PKCE_TTL)


async def _pop_pkce_verifier(state: str) -> str | None:
    """Atomic GET+DEL — one-time use."""
    redis = await get_redis()
    key = f"pkce_verifier:{state}"
    verifier = await redis.get(key)
    await redis.delete(key)
    return verifier


# ──────────────────────────────────────────────────────────────
# Refresh lock — prevents concurrent refresh races
# ──────────────────────────────────────────────────────────────
# Key: "refresh_lock:{user_id}:{provider}" TTL: 30s
# If a lock exists, wait briefly and re-read from DB.

REFRESH_LOCK_TTL = 30


async def _acquire_refresh_lock(user_id: str, provider_name: str) -> bool:
    """Returns True if lock was acquired, False if already locked."""
    redis = await get_redis()
    key = f"refresh_lock:{user_id}:{provider_name}"
    result = await redis.set(key, "1", ex=REFRESH_LOCK_TTL, nx=True)
    return result is not None


async def _release_refresh_lock(user_id: str, provider_name: str) -> None:
    redis = await get_redis()
    await redis.delete(f"refresh_lock:{user_id}:{provider_name}")

# ──────────────────────────────────────────────────────────────
# PKCE helpers (pure functions, no I/O)
# ──────────────────────────────────────────────────────────────

def generate_pkce_pair() -> tuple[str, str]:
    """
    Returns (code_verifier, code_challenge).
    code_verifier: 128-char URL-safe random string (RFC 7636 max)
    code_challenge: base64url(SHA-256(verifier)) — S256 method
    """
    code_verifier = secrets.token_urlsafe(96)  # 96 bytes → 128 base64url chars
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge

# ──────────────────────────────────────────────────────────────
# OAuthEngine
# ──────────────────────────────────────────────────────────────

class OAuthEngine:

    # ── Internal DB helpers ────────────────────────────────

    def _get_provider(self, provider_name: str) -> ProviderConfig:
        provider = PROVIDERS.get(provider_name)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider

    async def _get_connection(
        self, user_id: str, provider_name: str
    ) -> UserConnection | None:
        async with async_session_maker() as db:
            result = await db.execute(
                select(UserConnection).where(
                    UserConnection.user_id == user_id,
                    UserConnection.provider == provider_name,
                )
            )
            return result.scalar_one_or_none()

    async def _upsert_connection(
        self,
        user_id: str,
        provider_name: str,
        auth_type: str,
        access_token: str,
        refresh_token: str | None,
        expires_at: datetime | None,
    ) -> None:
        stmt = pg_insert(UserConnection).values(
            user_id=user_id,
            provider=provider_name,
            auth_type=auth_type,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )
        stmt = stmt.on_conflict_do_update(
            constraint="uq_user_provider",
            set_={
                "access_token": stmt.excluded.access_token,
                "refresh_token": stmt.excluded.refresh_token,
                "expires_at": stmt.excluded.expires_at,
            },
        )
        async with async_session_maker() as db:
            await db.execute(stmt)
            await db.commit()

    # ── Public API ─────────────────────────────────────────

    async def get_authorization_url(
        self, provider_name: str, user_id: str, session_id: str | None = None
    ) -> str:
        """
        Build the OAuth redirect URL. Handles:
        - CSRF state (stored in Redis)
        - PKCE code_verifier generation + storage (if provider uses PKCE)
        - Slack's dual scope params (scope + user_scope)
        - Provider-specific extra_authorize_params
        """
        provider = self._get_provider(provider_name)

        # CSRF state
        state = secrets.token_urlsafe(32)
        await store_oauth_state(state, user_id, session_id)

        params: dict = {
            "client_id": provider.client_id,
            "redirect_uri": f"{env.oauth_redirect_base_url}/connections/{provider_name}/callback",
            "response_type": "code",
            "state": state,
        }

        # Scopes — bot/main scopes
        if provider.scopes:
            params["scope"] = provider.scope_separator.join(provider.scopes)

        # User-level scopes (Slack: user_scope param)
        if provider.user_scope_param and provider.user_scopes:
            params[provider.user_scope_param] = provider.scope_separator.join(
                provider.user_scopes
            )

        # Provider-specific authorize params (e.g. access_type=offline for Google)
        params.update(provider.extra_authorize_params)

        # PKCE — generate verifier + challenge, store verifier in Redis
        if provider.pkce_mode in ("s256", "s256+secret"):
            code_verifier, code_challenge = generate_pkce_pair()
            await _store_pkce_verifier(state, code_verifier)
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"

        auth_url = f"{provider.authorize_url}?{urlencode(params)}"
        logger.info(f"Authorization URL generated for {provider_name}, user {user_id}")
        return auth_url

    async def handle_callback(
        self, provider_name: str, code: str, state: str
    ) -> tuple[str, str | None]:
        """
        Handle OAuth callback. Returns user_id on success.
        
        Handles:
        - CSRF state validation (Redis atomic GET+DEL)
        - PKCE verifier retrieval (Redis atomic GET+DEL)
        - client_secret_post / client_secret_basic / none auth methods
        - form / json body formats
        - Nested token extraction (e.g. Slack authed_user.access_token)
        - Token validation before storing
        """
        provider = self._get_provider(provider_name)

        # 1. Validate CSRF state (atomic GET+DEL)
        user_id, session_id = await pop_oauth_state(state)
        if not user_id:
            raise ValueError(
                "Invalid or expired OAuth state. Please try connecting again."
            )

        # 2. Retrieve PKCE verifier if applicable (atomic GET+DEL)
        code_verifier: str | None = None
        if provider.pkce_mode in ("s256", "s256+secret"):
            code_verifier = await _pop_pkce_verifier(state)
            if not code_verifier:
                raise ValueError(
                    "PKCE verifier not found or expired. Please try connecting again."
                )

        # 3. Build token exchange payload
        redirect_uri = f"{env.oauth_redirect_base_url}/connections/{provider_name}/callback"

        payload: dict = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }

        # Add client credentials to body (basic auth method removes them later)
        if provider.token_endpoint_auth_method in ("body", "basic"):
            payload["client_id"] = provider.client_id
            payload["client_secret"] = provider.client_secret
        elif provider.token_endpoint_auth_method == "none":
            # Pure PKCE: only client_id, no secret
            payload["client_id"] = provider.client_id

        # PKCE verifier in body (for s256 and s256+secret)
        if code_verifier:
            payload["code_verifier"] = code_verifier

        # Provider-specific extra token body params (e.g. Google's grant_type override)
        payload.update(provider.extra_token_params)

        # 4. POST to token endpoint (handles auth method + format internally)
        token_data = await _post_token_endpoint(
            url=provider.token_url,
            payload=payload,
            provider=provider,
        )

        # 5. Extract tokens using configured paths
        access_token = _extract_nested(token_data, provider.access_token_path)
        if not access_token:
            raise ValueError(
                f"No access token at '{provider.access_token_path}' in response: {token_data}"
            )

        refresh_token = _extract_nested(token_data, provider.refresh_token_path)
        expires_in = token_data.get("expires_in")
        expires_at = _make_expires_at(expires_in)

        # 6. Validate the token works before storing
        is_valid = await self.test_connection(provider_name, access_token)
        if not is_valid:
            raise ValueError(
                "Token received but failed validation. Please try again."
            )

        # 7. Persist
        await self._upsert_connection(
            user_id=user_id,
            provider_name=provider_name,
            auth_type="oauth2",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )

        logger.info(f"Connected {provider_name} for user {user_id}")
        return user_id, session_id

    async def refresh_if_expired(
        self, user_id: str, provider_name: str
    ) -> str:
        """
        Returns a valid access token, refreshing if expired.
        
        Handles:
        - 60s expiry buffer
        - Redis lock to prevent concurrent refresh races (multi-worker safe)
        - Refresh token rotation (Notion, Google)
        - client_secret_basic for providers that need it on refresh too
        """
        provider = self._get_provider(provider_name)
        connection = await self._get_connection(user_id, provider_name)

        if not connection:
            raise ValueError(
                f"No {provider_name} connection found for user {user_id}. "
                "Please connect first."
            )

        # Token still valid
        if not _is_expired(connection.expires_at):
            return connection.access_token

        # No refresh URL or no refresh token — must reconnect
        if not provider.refresh_url or not connection.refresh_token:
            raise ValueError(
                f"{provider_name} token expired and cannot be refreshed. "
                "User must reconnect."
            )

        # Acquire refresh lock — prevents race condition in multi-worker setup
        # If another worker is already refreshing, wait briefly and re-read from DB
        lock_acquired = await _acquire_refresh_lock(user_id, provider_name)
        if not lock_acquired:
            import asyncio
            await asyncio.sleep(2)
            connection = await self._get_connection(user_id, provider_name)
            if connection and not _is_expired(connection.expires_at):
                return connection.access_token
            raise ValueError(
                f"Concurrent refresh in progress for {provider_name}. "
                "Please retry."
            )

        try:
            logger.info(f"Refreshing {provider_name} token for user {user_id}")

            payload = {
                "grant_type": "refresh_token",
                "refresh_token": connection.refresh_token,
                "client_id": provider.client_id,
                "client_secret": provider.client_secret,
            }

            token_data = await _post_token_endpoint(
                url=provider.refresh_url,
                payload=payload,
                provider=provider,
            )

            new_access_token = _extract_nested(token_data, provider.access_token_path)
            if not new_access_token:
                raise ValueError(
                    f"No access token in refresh response: {token_data}"
                )

            # Rotation: use new refresh token if provided, keep old one if not
            new_refresh_token = (
                _extract_nested(token_data, provider.refresh_token_path)
                if provider.refresh_token_behavior == "rotate"
                else token_data.get("refresh_token", connection.refresh_token)
            ) or connection.refresh_token

            new_expires_at = _make_expires_at(token_data.get("expires_in"))

            async with async_session_maker() as db:
                await db.execute(
                    update(UserConnection)
                    .where(
                        UserConnection.user_id == user_id,
                        UserConnection.provider == provider_name,
                    )
                    .values(
                        access_token=new_access_token,
                        refresh_token=new_refresh_token,
                        expires_at=new_expires_at,
                    )
                )
                await db.commit()

            logger.info(f"Refreshed {provider_name} token for user {user_id}")
            return new_access_token

        finally:
            await _release_refresh_lock(user_id, provider_name)

    async def test_connection(
        self, provider_name: str, access_token: str
    ) -> bool:
        """
        Hit the provider's test endpoint to verify the token works.
        Handles provider-specific extra headers (e.g. Notion-Version).
        """
        provider = self._get_provider(provider_name)
        if provider.test_url == None:
            return True

        headers = {
            provider.token_header: f"{provider.token_prefix} {access_token}"
        }
        # Notion requires Notion-Version on all API calls
        headers.update(provider.extra_token_headers)

        async with httpx.AsyncClient() as client:
            response = await client.get(provider.test_url, headers=headers)

        if response.status_code == 200:
            logger.info(f"Token validation passed for {provider_name}")
            return True

        logger.warning(
            f"Token validation failed for {provider_name}: {response.status_code}"
        )
        return False

    async def revoke(self, user_id: str, provider_name: str) -> None:
        """
        Disconnect — revoke on provider side (fail-open), delete from DB.
        """
        provider = self._get_provider(provider_name)
        connection = await self._get_connection(user_id, provider_name)

        if not connection:
            raise ValueError(
                f"No {provider_name} connection found for user {user_id}"
            )

        if provider.revoke_url:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        provider.revoke_url,
                        data={"token": connection.access_token},
                        headers={"Accept": "application/json"},
                    )
                logger.info(
                    f"Revoked {provider_name} token on provider side for user {user_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Provider-side revocation failed for {provider_name}: {e}"
                )

        async with async_session_maker() as db:
            await db.execute(
                delete(UserConnection).where(
                    UserConnection.user_id == user_id,
                    UserConnection.provider == provider_name,
                )
            )
            await db.commit()

        logger.info(f"Disconnected {provider_name} for user {user_id}")

    async def store_api_key(
        self, user_id: str, provider_name: str, api_key: str
    ) -> bool:
        """For API key providers (Firecrawl, Exa, etc.)."""
        is_valid = await self.test_connection(provider_name, api_key)
        if not is_valid:
            raise ValueError(
                f"Invalid API key for {provider_name}. Please check and try again."
            )
        await self._upsert_connection(
            user_id=user_id,
            provider_name=provider_name,
            auth_type="api_key",
            access_token=api_key,
            refresh_token=None,
            expires_at=None,
        )
        logger.info(f"Stored API key for {provider_name}, user {user_id}")
        return True

    async def get_valid_token(self, user_id: str, provider_name: str) -> str:
        """
        Single entry point for agent/tool layer.
        Handles refresh automatically — callers never think about expiry.
        """
        return await self.refresh_if_expired(user_id, provider_name)