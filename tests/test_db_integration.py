"""
test_db_integration.py
======================
LAYER 1: Database Integration Tests.

Verifies that the ORM models can be created, read, updated and deleted
through the actual SQLAlchemy async session used by the app — without
going through HTTP. This catches:
  - schema errors (missing columns, wrong FK targets)
  - async driver issues (asyncpg / aiosqlite)
  - cascade behavior on relationships

Each test creates a fresh isolated schema (see conftest.test_engine fixture).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db.models import Base, User, Conversation, UserConnection

pytestmark = pytest.mark.db


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA SANITY
# ─────────────────────────────────────────────────────────────────────────────
class TestSchemaSanity:
    """Confirm the metadata actually contains what app.py expects."""

    def test_all_tables_present(self, test_engine):
        """The four tables the app relies on must be defined."""
        tables = set(Base.metadata.tables.keys())
        assert "user" in tables
        assert "conversations" in tables
        assert "user_connections" in tables

    def test_user_columns(self):
        """User (fastapi-users base table) needs email, hashed_password, is_active."""
        user_cols = set(User.__table__.columns.keys())
        assert "email" in user_cols
        assert "hashed_password" in user_cols
        assert "is_active" in user_cols
        assert "is_superuser" in user_cols
        assert "is_verified" in user_cols

    def test_conversation_columns(self):
        conv_cols = set(Conversation.__table__.columns.keys())
        assert "id" in conv_cols
        assert "user_id" in conv_cols
        assert "title" in conv_cols
        assert "created_at" in conv_cols

    def test_user_connection_columns(self):
        conn_cols = set(UserConnection.__table__.columns.keys())
        assert "id" in conn_cols
        assert "user_id" in conn_cols
        assert "provider" in conn_cols


# ─────────────────────────────────────────────────────────────────────────────
# CRUD — User
# ─────────────────────────────────────────────────────────────────────────────
class TestUserCRUD:
    """Round-trip a user through the database."""

    @pytest.mark.asyncio
    async def test_create_and_read_user(self, db_session):
        user_id = uuid.uuid4()
        user = User(
            id=user_id,
            email=f"db-{user_id.hex[:8]}@orphic.local",
            hashed_password="$2b$12$fakefakefakefakefakefakefakefakefakefake",
            is_active=True,
            is_superuser=False,
            is_verified=True,
        )
        db_session.add(user)
        await db_session.commit()

        # Read back
        result = await db_session.execute(select(User).where(User.id == user_id))
        fetched = result.scalar_one()

        assert fetched.id == user_id
        assert fetched.is_active is True
        assert fetched.email.startswith("db-")

    @pytest.mark.asyncio
    async def test_user_email_uniqueness(self, db_session):
        """A duplicate email must violate the unique constraint."""
        email = "duplicate@orphic.local"
        u1 = User(
            id=uuid.uuid4(),
            email=email,
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=False,
        )
        db_session.add(u1)
        await db_session.commit()

        u2 = User(
            id=uuid.uuid4(),
            email=email,
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=False,
        )
        db_session.add(u2)
        with pytest.raises(IntegrityError):
            await db_session.commit()

    @pytest.mark.asyncio
    async def test_delete_user_cascades_conversations(self, db_session):
        """Deleting a user should remove their conversations (cascade)."""
        user = User(
            id=uuid.uuid4(),
            email="cascade@orphic.local",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=False,
        )
        db_session.add(user)
        await db_session.flush()

        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Should be cascaded",
        )
        db_session.add(conv)
        await db_session.commit()

        # Sanity: conversation is there
        result = await db_session.execute(
            select(Conversation).where(Conversation.user_id == user.id)
        )
        assert len(result.scalars().all()) == 1

        # Delete the user — cascade should nuke the conversation
        await db_session.delete(user)
        await db_session.commit()

        result = await db_session.execute(
            select(Conversation).where(Conversation.user_id == user.id)
        )
        assert result.scalars().all() == []


# ─────────────────────────────────────────────────────────────────────────────
# CRUD — Conversation
# ─────────────────────────────────────────────────────────────────────────────
class TestConversationCRUD:
    @pytest.mark.asyncio
    async def test_create_conversation(self, db_session):
        user = User(
            id=uuid.uuid4(),
            email="conv@orphic.local",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=False,
        )
        db_session.add(user)
        await db_session.flush()

        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Test thread",
        )
        db_session.add(conv)
        await db_session.commit()

        result = await db_session.execute(
            select(Conversation).where(Conversation.user_id == user.id)
        )
        found = result.scalar_one()
        assert found.title == "Test thread"
        assert found.created_at is not None  # server_default populated

    @pytest.mark.asyncio
    async def test_rename_conversation(self, db_session):
        user = User(
            id=uuid.uuid4(),
            email="rename@orphic.local",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=False,
        )
        db_session.add(user)
        await db_session.flush()

        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=user.id,
            title="Old title",
        )
        db_session.add(conv)
        await db_session.commit()

        conv.title = "New title"
        await db_session.commit()
        await db_session.refresh(conv)

        assert conv.title == "New title"

    @pytest.mark.asyncio
    async def test_conversation_query_filters_by_user(self, db_session):
        """Make sure user isolation works at the query level."""
        u1 = User(id=uuid.uuid4(), email="u1@o.local", hashed_password="x",
                  is_active=True, is_superuser=False, is_verified=False)
        u2 = User(id=uuid.uuid4(), email="u2@o.local", hashed_password="x",
                  is_active=True, is_superuser=False, is_verified=False)
        db_session.add_all([u1, u2])
        await db_session.flush()

        db_session.add_all([
            Conversation(id=str(uuid.uuid4()), user_id=u1.id, title="u1-thread"),
            Conversation(id=str(uuid.uuid4()), user_id=u2.id, title="u2-thread"),
        ])
        await db_session.commit()

        u1_threads = (await db_session.execute(
            select(Conversation).where(Conversation.user_id == u1.id)
        )).scalars().all()

        assert len(u1_threads) == 1
        assert u1_threads[0].title == "u1-thread"


# ─────────────────────────────────────────────────────────────────────────────
# CRUD — UserConnection
# ─────────────────────────────────────────────────────────────────────────────
class TestUserConnectionCRUD:
    @pytest.mark.asyncio
    async def test_create_user_connection(self, db_session):
        user = User(
            id=uuid.uuid4(),
            email="conn@orphic.local",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=False,
        )
        db_session.add(user)
        await db_session.flush()

        connection = UserConnection(
            id=str(uuid.uuid4()),
            user_id=user.id,
            provider="github",
        )
        db_session.add(connection)
        await db_session.commit()

        result = await db_session.execute(
            select(UserConnection).where(UserConnection.user_id == user.id)
        )
        rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].provider == "github"

    @pytest.mark.asyncio
    async def test_user_can_have_multiple_connections(self, db_session):
        """A single user might connect to GitHub + Notion + etc."""
        user = User(
            id=uuid.uuid4(),
            email="multi@orphic.local",
            hashed_password="x",
            is_active=True,
            is_superuser=False,
            is_verified=False,
        )
        db_session.add(user)
        await db_session.flush()

        db_session.add_all([
            UserConnection(id=str(uuid.uuid4()), user_id=user.id, provider="github"),
            UserConnection(id=str(uuid.uuid4()), user_id=user.id, provider="notion"),
        ])
        await db_session.commit()

        result = await db_session.execute(
            select(UserConnection).where(UserConnection.user_id == user.id)
        )
        providers = {r.provider for r in result.scalars().all()}
        assert providers == {"github", "notion"}


# ─────────────────────────────────────────────────────────────────────────────
# get_async_session dependency
# ─────────────────────────────────────────────────────────────────────────────
class TestGetAsyncSessionDependency:
    @pytest.mark.asyncio
    async def test_session_can_be_used_as_async_generator(self, test_engine):
        """The get_async_session() generator must yield a working session."""
        from db.models import get_async_session

        agen = get_async_session()
        session = await agen.__anext__()
        try:
            # A trivial no-op query — proves the connection works
            result = await session.execute(select(1))
            assert result.scalar() == 1
        finally:
            await agen.aclose()