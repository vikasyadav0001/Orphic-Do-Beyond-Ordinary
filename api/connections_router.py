from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from services.connections.oauth_engine import OAuthEngine
from services.connections.providers import PROVIDERS
from db.models import User, UserConnection, async_session_maker
from api.auth import current_active_user
from api.scalekit_auth import current_scalekit_active_user
from sqlalchemy import select
from services.cache import invalidate_user_connections, invalidate_token
from utils.logger import get_logger
from config import get_settings

env = get_settings()
logger = get_logger(__name__)

router = APIRouter(prefix="/connections", tags=["Connections"])
engine = OAuthEngine()


class APIKeyRequest(BaseModel):
    api_key: str


@router.get("/")
async def get_user_connections(user: User = Depends(current_scalekit_active_user)):
    """All providers with this user's connection status."""
    async with async_session_maker() as db:
        stmt = select(UserConnection.provider).where(
            UserConnection.user_id == user.id
        )
        result = await db.execute(stmt)
        connected_providers = {row[0] for row in result.all()}

    return [
        {
            "name": config.name,
            "label": config.label,
            "auth_type": config.auth_type,
            "connected": name in connected_providers,
        }
        for name, config in PROVIDERS.items()
    ]


@router.get("/connected")
async def get_connected_providers(user: User = Depends(current_scalekit_active_user)):
    """
    Returns only providers this user has active connections for.
    Used by the agent before building workflows — knows what tools are available.
    """
    async with async_session_maker() as db:
        stmt = select(UserConnection.provider, UserConnection.auth_type).where(
            UserConnection.user_id == user.id
        )
        result = await db.execute(stmt)
        rows = result.all()

    return [{"provider": row[0], "auth_type": row[1]} for row in rows]


@router.get("/{provider_name}/authorize")
async def authorize_user(
    provider_name: str,
    session_id: str= Query(None),
    user: User = Depends(current_scalekit_active_user)
):
    """Start OAuth flow — returns the provider authorization URL."""
    try:
        auth_url = await engine.get_authorization_url(provider_name, str(user.id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"authorization_url": auth_url}


@router.get("/{provider_name}/callback")
async def oauth_callback(
    provider_name: str,
    code: str = Query(...),
    state: str = Query(...),
):
    """
    OAuth callback — provider redirects browser here after user authenticates.
    Not JWT-protected — browser redirects cannot carry Authorization headers.
    user_id is resolved from the Redis state token set during authorize step.
    """
    session_id = None
    try:
        user_id, session_id = await engine.handle_callback(provider_name, code, state)
        
        # Invalidate stale cache so agent_node rebuilds from DB on next turn
        await invalidate_user_connections(user_id)
        await invalidate_token(user_id, provider_name)

        if session_id:
            return RedirectResponse(
                url=f"{env.frontend_url}/chat/{session_id}?oauth=success&provider={provider_name}&session_id={session_id}"
            )
        else:
            return RedirectResponse(
                url=f"{env.frontend_url}/connectors?status=success&provider={provider_name}"
            )

    except Exception as e:
        logger.error(f"OAuth callback failed for {provider_name}: {e}")
        if session_id:
            return RedirectResponse(
                url=f"{env.frontend_url}/chat/{session_id}?oauth=error&provider={provider_name}&session_id={session_id}"
            )

        return RedirectResponse(
            url=f"{env.frontend_url}/connectors?status=error&provider={provider_name}"
        )



@router.post("/{provider_name}/apikey")
async def store_api_key_for_user(
    provider_name: str,
    body: APIKeyRequest,
    user: User = Depends(current_scalekit_active_user),
):
    """Store an API key connection (Firecrawl, Exa, Serper etc.)"""
    try:
        await engine.store_api_key(str(user.id), provider_name, body.api_key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # API key is a new connection — invalidate so agent sees it
    await invalidate_user_connections(str(user.id))

    return {"status": "connected", "provider": provider_name}


@router.delete("/{provider_name}")
async def delete_connection(
    provider_name: str,
    user: User = Depends(current_scalekit_active_user),
):
    """Disconnect a provider — revoke token and remove from DB."""
    try:
        await engine.revoke(str(user.id), provider_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Clear cached token and connection list — agent sees disconnection immediately
    await invalidate_user_connections(str(user.id))
    await invalidate_token(str(user.id), provider_name)

    return {"status": "disconnected", "provider": provider_name}



@router.get("/{provider_name}/test")
async def test_user_connection(
    provider_name: str,
    user: User = Depends(current_scalekit_active_user),
):
    """
    Test if a user's connection is valid.
    Uses get_valid_token — handles refresh automatically before testing.
    """
    try:
        token = await engine.get_valid_token(str(user.id), provider_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    is_valid = await engine.test_connection(provider_name, token)
    if is_valid:
        return {"status": "valid", "provider": provider_name}

    raise HTTPException(
        status_code=401,
        detail=f"{provider_name} token is invalid or expired. Please reconnect."
    )