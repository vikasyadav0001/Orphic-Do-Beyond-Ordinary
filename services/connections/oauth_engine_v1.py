from .providers import ProviderConfig, PROVIDERS
from db.models import UserConnection, async_session_maker
from sqlalchemy.dialects.postgresql import insert as pg_insert
from datetime import datetime, timezone, timedelta
from urllib.parse import urlencode
from sqlalchemy import select, delete, update
import httpx
import secrets
from config import get_settings
from utils.logger import get_logger
from services.cache import get_redis, pop_oauth_state, store_oauth_state

env = get_settings()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc() -> datetime:
    """Timezone-aware UTC now. Replaces deprecated datetime.utcnow()."""
    return datetime.now(timezone.utc)


def _make_expires_at(expires_in: int | None) -> datetime | None:
    if not expires_in:
        return None
    return _now_utc() + timedelta(seconds=int(expires_in))


def _is_expired(expires_at: datetime | None) -> bool:
    """
    Returns True if the token is expired or about to expire within 60 seconds.
    The 60s buffer prevents edge cases where token expires mid-request.
    expires_at may be naive (from DB) or aware — we handle both.
    """
    if expires_at is None:
        return False
    # If naive datetime from DB, treat as UTC
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= (_now_utc() + timedelta(seconds=60))


class OAuthEngine:

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _get_provider(self, provider_name: str) -> ProviderConfig:
        """Look up provider config, raise clearly if not found."""
        provider = PROVIDERS.get(provider_name)
        if not provider:
            logger.error(f"Unknown provider requested: {provider_name}")
            raise ValueError(f"Unknown provider: {provider_name}")
        return provider


    async def _get_connection(self, user_id: str, provider_name: str) -> UserConnection | None:
        """Fetch a user's connection from DB."""
        async with async_session_maker() as db:
            stmt = select(UserConnection).where(
                UserConnection.user_id == user_id,
                UserConnection.provider == provider_name,
            )
            result = await db.execute(stmt)
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
        """Insert or update a user connection — unified for OAuth and API keys."""
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

    def _build_token_request(
        self,
        provider: ProviderConfig,
        provider_name: str,
        code: str,
    ) -> dict:
        """
        Build the token exchange payload.
        Most providers use form-encoded body.
        Some (Notion) require JSON body — controlled by provider.token_request_format.
        extra_token_params handles provider-specific required fields
        e.g. Google requires grant_type="authorization_code".
        """
        base_params = {
            "client_id": provider.client_id,
            "client_secret": provider.client_secret,
            "code": code,
            "redirect_uri": f"{env.oauth_redirect_base_url}/connections/{provider_name}/callback",
        }
        # Merge any provider-specific extra params (e.g. grant_type for Google)
        if provider.extra_token_params:
            base_params.update(provider.extra_token_params)
        return base_params

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def get_authorization_url(self, provider_name: str, user_id: str) -> str:
        """
        Generate the OAuth redirect URL for a provider.
        Now async because we write the CSRF state to Redis.
        """
        provider = self._get_provider(provider_name)

        state = secrets.token_urlsafe(32)
        await store_oauth_state(state, user_id)

        params = {
            "client_id": provider.client_id,
            "redirect_uri": f"{env.oauth_redirect_base_url}/connections/{provider_name}/callback",
            "scope": provider.scope_separator.join(provider.scopes or []),
            "response_type": "code",
            "state": state,
        }

        # Some providers need extra authorize params (e.g. Google needs access_type=offline
        # to return a refresh_token)
        if provider.extra_authorize_params:
            params.update(provider.extra_authorize_params)

        auth_url = f"{provider.authorize_url}?{urlencode(params)}"
        logger.info(f"Generated authorization URL for {provider_name}, user {user_id}")
        return auth_url

    async def handle_callback(self, provider_name: str, code: str, state: str) -> bool:
        """
        Handle the OAuth callback from the provider.
        Validates CSRF state, exchanges code for tokens, tests and stores them.
        """
        provider = self._get_provider(provider_name)

        # Step 1: Validate CSRF state from Redis (atomic GET+DEL)
        user_id = await pop_oauth_state(state)
        if not user_id:
            logger.error(f"Invalid or expired OAuth state for {provider_name}")
            raise ValueError("Invalid or expired OAuth state. Please try connecting again.")

        # Step 2: Exchange code for tokens
        # Provider config controls form vs JSON body format
        payload = self._build_token_request(provider, provider_name, code)

        async with httpx.AsyncClient() as client:
            if provider.token_request_format == "json":
                response = await client.post(
                    provider.token_url,
                    json=payload,
                    headers={"Accept": "application/json"},
                )
            else:
                # Default: form-encoded (GitHub, Google, most providers)
                response = await client.post(
                    provider.token_url,
                    data=payload,
                    headers={"Accept": "application/json"},
                )

        if response.status_code != 200:
            logger.error(f"Token exchange failed for {provider_name}: {response.text}")
            raise ValueError(f"Token exchange failed: {response.text}")

        token_data = response.json()
        access_token = token_data.get("access_token")
        refresh_token = token_data.get("refresh_token")
        expires_in = token_data.get("expires_in")

        if not access_token:
            raise ValueError(f"No access token in response: {token_data}")

        expires_at = _make_expires_at(expires_in)

        # Step 3: Validate the token actually works before storing
        is_valid = await self.test_connection(provider_name, access_token)
        if not is_valid:
            raise ValueError("Token received but failed validation. Please try again.")

        # Step 4: Upsert into DB
        await self._upsert_connection(
            user_id=user_id,
            provider_name=provider_name,
            auth_type="oauth2",
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
        )

        logger.info(f"Successfully connected {provider_name} for user {user_id}")
        return user_id

    async def refresh_if_expired(self, user_id: str, provider_name: str) -> str:
        """
        Returns a valid access token, refreshing silently if expired.
        Call this before every MCP tool call or n8n credential injection.
        Has a 60s expiry buffer — prevents mid-request token expiry.
        """
        provider = self._get_provider(provider_name)
        connection = await self._get_connection(user_id, provider_name)

        if not connection:
            raise ValueError(f"No {provider_name} connection found for user {user_id}. Please connect first.")

        # Token still valid — return immediately
        if not _is_expired(connection.expires_at):
            return connection.access_token

        # Token expired — need to refresh
        if not provider.refresh_url or not connection.refresh_token:
            raise ValueError(
                f"{provider_name} token expired and cannot be refreshed automatically. "
                f"User must reconnect."
            )

        logger.info(f"Refreshing expired {provider_name} token for user {user_id}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                provider.refresh_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": connection.refresh_token,
                    "client_id": provider.client_id,
                    "client_secret": provider.client_secret,
                },
                headers={"Accept": "application/json"},
            )

        if response.status_code != 200:
            logger.error(f"Token refresh failed for {provider_name}: {response.text}")
            raise ValueError(f"Token refresh failed: {response.text}")

        token_data = response.json()
        new_access_token = token_data.get("access_token")
        # Some providers (Google) rotate refresh tokens — use new one if provided
        new_refresh_token = token_data.get("refresh_token", connection.refresh_token)
        new_expires_at = _make_expires_at(token_data.get("expires_in"))

        async with async_session_maker() as db:
            stmt = update(UserConnection).where(
                UserConnection.user_id == user_id,
                UserConnection.provider == provider_name,
            ).values(
                access_token=new_access_token,
                refresh_token=new_refresh_token,
                expires_at=new_expires_at,
            )
            await db.execute(stmt)
            await db.commit()

        logger.info(f"Successfully refreshed {provider_name} token for user {user_id}")
        return new_access_token

    async def test_connection(self, provider_name: str, access_token: str) -> bool:
        """
        Hit the provider's test endpoint with the token to verify it works.
        Called after token exchange and can be called standalone for health checks.
        """
        provider = self._get_provider(provider_name)
        headers = {
            provider.token_header: f"{provider.token_prefix} {access_token}"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(provider.test_url, headers=headers)

        if response.status_code == 200:
            logger.info(f"Token validation passed for {provider_name}")
            return True

        logger.warning(f"Token validation failed for {provider_name}: {response.status_code}")
        return False

    async def revoke(self, user_id: str, provider_name: str) -> None:
        """
        Disconnect a provider — revoke on provider side, delete from DB.
        Fail-open: if provider revocation fails, we still clean up locally.
        """
        provider = self._get_provider(provider_name)
        connection = await self._get_connection(user_id, provider_name)

        if not connection:
            raise ValueError(f"No {provider_name} connection found for user {user_id}")

        if provider.revoke_url:
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        provider.revoke_url,
                        data={"token": connection.access_token},
                        headers={"Accept": "application/json"},
                    )
                logger.info(f"Revoked {provider_name} token on provider side for user {user_id}")
            except Exception as e:
                # Fail-open: log and continue — token will expire naturally
                logger.warning(f"Provider-side revocation failed for {provider_name}: {e}")

        async with async_session_maker() as db:
            stmt = delete(UserConnection).where(
                UserConnection.user_id == user_id,
                UserConnection.provider == provider_name,
            )
            await db.execute(stmt)
            await db.commit()

        logger.info(f"Disconnected {provider_name} for user {user_id}")

    async def store_api_key(self, user_id: str, provider_name: str, api_key: str) -> bool:
        """
        For API key providers (Firecrawl, Exa, Serper etc.)
        User pastes key → we test it → store if valid.
        Same DB table as OAuth connections, auth_type='api_key'.
        """
        is_valid = await self.test_connection(provider_name, api_key)
        if not is_valid:
            raise ValueError(f"Invalid API key for {provider_name}. Please check and try again.")

        await self._upsert_connection(
            user_id=user_id,
            provider_name=provider_name,
            auth_type="api_key",
            access_token=api_key,
            refresh_token=None,
            expires_at=None,
        )

        logger.info(f"Stored API key for {provider_name} for user {user_id}")
        return True

    async def get_valid_token(self, user_id: str, provider_name: str) -> str:
        """
        Single entry point for agent/n8n to get a valid token.
        Handles refresh automatically — agent never needs to think about expiry.
        Call this before every MCP tool call or n8n credential injection.
        """
        return await self.refresh_if_expired(user_id, provider_name)





if __name__ == "__main__":
    import asyncio    
    conn = OAuthEngine()    
    print(asyncio.run(conn.get_authorization_url("github", "default_user")))