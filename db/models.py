"""
Database models for authentication and conversation
"""

from typing import AsyncGenerator
from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from  sqlalchemy import String, ForeignKey, DateTime, func
import uuid

from config import get_settings
env = get_settings()

DB_URL = env.db_url

# Only do Postgres-specific rewriting when we're actually talking to Postgres.
# This keeps the module importable for SQLite-backed test suites.
if DB_URL.startswith("postgresql://"):
    DB_URL = DB_URL.replace("postgresql://", "postgresql+asyncpg://")
    if "sslmode=require" in DB_URL:
        DB_URL = DB_URL.replace("sslmode=require", "ssl=require")

    from urllib.parse import parse_qs, urlparse, urlencode, urlunparse
    parsed = urlparse(DB_URL)
    query_params = parse_qs(parsed.query)
    query_params.pop("channel_binding", None)

    flat_query = {k: v[0] for k, v in query_params.items()}

    if flat_query.get("sslmode") == "require":
        flat_query.pop("sslmode")
        flat_query["ssl"] = "require"

    new_query = urlencode(flat_query)

    parsed = parsed._replace(query=new_query)
    DB_URL = urlunparse(parsed)

engine = create_async_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args={"timeout": 60}
)

async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class User(SQLAlchemyBaseUserTableUUID, Base):
    """User table holding auth credentials. """

    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    connections: Mapped[list["UserConnection"]] = relationship(back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    """Maps a langgraph thread id to specific user"""

    __tablename__="conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False, default="New Chat")
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="conversations")

class UserConnection(Base):
    """Tracks which external providers(tools) a user has successfully connected."""

    __tablename__ = "user_connections"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False) # e.g., "github", "notion"
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="connections")


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
    
async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)