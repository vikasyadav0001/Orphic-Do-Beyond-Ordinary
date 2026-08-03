"""
Authentication using Scalekit.
Validates bearer tokens issued by Scalekit and maps them to
the existing fastapi-users User model via JIT provisioning.
"""

import uuid
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from scalekit import ScalekitClient

from db.models import User, get_async_session
from config import get_settings
from utils.logger import get_logger

env = get_settings()
logger = get_logger(__name__)

# ── Scalekit client ────────────────────────────────────────────────────────────
# Create the client lazily so importing the app never performs network auth.
_scalekit_client: ScalekitClient | None = None


def get_scalekit_client() -> ScalekitClient:
    global _scalekit_client

    if _scalekit_client is not None:
        return _scalekit_client

    if not env.scalekit_env_url or not env.scalekit_client_id or not env.scalekit_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scalekit auth is not configured",
        )

    _scalekit_client = ScalekitClient(
        env_url=env.scalekit_env_url,
        client_id=env.scalekit_client_id,
        client_secret=env.scalekit_client_secret
    )
    return _scalekit_client

oauth2_scheme = HTTPBearer()

async def current_scalekit_active_user(
    token: HTTPAuthorizationCredentials = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_async_session)
) -> User:
    try:
        scalekit_client = get_scalekit_client()

        # 1. Verify token — raises internally if expired or signature invalid
        token_claims = scalekit_client.validate_token(token.credentials)

        if not token_claims:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

        # 2. Extract email from claims
        email = token_claims.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing email claim"
            )

        # 3. Lookup user in DB
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalars().first()

        # 4. Active check for existing users
        if user:
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account is inactive"
                )
            return user

        # 5. JIT provisioning — race-safe upsert
        # hashed_password is a NOT NULL legacy field from fastapi-users schema;
        # "!" is a bcrypt-impossible value so it can never match a real password check
        stmt = (
            pg_insert(User)
            .values(
                id=uuid.uuid4(),
                email=email,
                hashed_password="!",   # unusable — external auth only
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
            .on_conflict_do_nothing(index_elements=["email"])
        )
        await db.execute(stmt)
        await db.commit()

        # Re-fetch after upsert (handles race where another worker inserted first)
        result = await db.execute(select(User).where(User.email == email))
        user = result.scalars().first()

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scalekit auth error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
