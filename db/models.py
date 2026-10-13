"""
Database models for authentication and conversation
"""

from typing import AsyncGenerator
from fastapi import Depends
from fastapi_users.db import SQLAlchemyBaseUserTableUUID, SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from  sqlalchemy import String, ForeignKey, DateTime, func, Text, UniqueConstraint, Boolean, ARRAY, JSON
from typing import Optional,  List
from datetime import datetime
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
    pool_size=5,
    max_overflow=2,
    pool_recycle=300,
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
    workflows: Mapped[list["Workflows"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    mcp_config: Mapped[Optional["UserMCPConfig"]] = relationship(back_populates="user", uselist=False, cascade="all, delete-orphan")

class Conversation(Base):
    """Maps a langgraph thread id to specific user"""

    __tablename__="conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False, default="New Chat")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="conversations")

class ChatMessage(Base):
    """Stores every user/assistant/system message for UI history display.
    Independent of LangGraph checkpoints — never summarized or truncated."""

    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id: Mapped[str] = mapped_column(
        String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String, nullable=False)  # "user", "assistant", "system"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    conversation: Mapped["Conversation"] = relationship()


class UserConnection(Base):
    """Tracks which external providers(tools) a user has successfully connected."""

    __tablename__ = "user_connections"
    __table_args__ = (
        UniqueConstraint('user_id', 'provider', name='uq_user_provider'),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("user.id"), nullable=False)
    provider: Mapped[str] = mapped_column(String, nullable=False) # e.g., "github", "notion"
    auth_type: Mapped[str] = mapped_column(String)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="connections")

class UserMCPConfig(Base):
    """Stores per-user MCP server configuration for dynamic tool registration."""

    __tablename__ = "user_mcp_configs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="mcp_config")

class Workflows(Base):
    __tablename__ = "user_workflows"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    n8n_workflow_id: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    providers: Mapped[list] = mapped_column(
        ARRAY(String).with_variant(JSON, "sqlite"),
        default=list,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    trigger_summary: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source_conversation_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="workflows")

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
    
async def get_user_db(session: AsyncSession = Depends(get_async_session)):
    yield SQLAlchemyUserDatabase(session, User)
