"""
Authentication setup using fastapi-users.
Handles JWT issuance, password hashing, and user validation.
"""

import uuid
from typing import Optional

from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, UUIDIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport, 
    JWTStrategy,
)

from fastapi_users import schemas

class UserRead(schemas.BaseUser[uuid.UUID]):
    pass

class UserCreate(schemas.BaseUserCreate):
    pass

from utils.logger import get_logger
from config import get_settings
from db.models import User, get_user_db

env = get_settings()
logger = get_logger(__name__)

SECRET = env.jwt_secret

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy() -> JWTStrategy:

    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy
)

class UserManager(UUIDIDMixin, BaseUserManager[User, uuid.UUID]):
    """Handles logic for user creation, verification, and password resets."""

    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        logger.info(f"User {user.id} has registered.")

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info(f"User {user.id} has forgot their password.")
        # TODO: Send an email to user.email containing the reset token


    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        logger.info(f"Verification requested for user {user.id}.")
        # TODO: Send an email to user.email containing the verification token


async def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

fastapi_users = FastAPIUsers[User, uuid.UUID](
    get_user_manager,
    [auth_backend],
)

# 4. Dependency to use in our routes to protect endpoints!
current_active_user = fastapi_users.current_user(active=True)