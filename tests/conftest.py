"""
conftest.py
===========
Shared pytest fixtures for the Orphic test suite.
"""

# ─── MUST BE FIRST: sets os.environ before any app/config import ────────────
from __future__ import annotations

import os

# Side-effect import: sets OPENAI_API_KEY, DB_URL, JWT_SECRET, etc.
# MUST run before `from config import get_settings`.
from tests import _test_env  # noqa: F401

import asyncio
import sys
import uuid
import tempfile
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EVENT LOOP — pytest-asyncio session-scoped
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Single event loop for the whole test session (asyncpg + AsyncSession)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEST DATABASE — isolated SQLite (or override URL)
# ─────────────────────────────────────────────────────────────────────────────
TEST_DB_FILE = os.path.join(tempfile.gettempdir(), f"orphic_test_{uuid.uuid4().hex}.db")
TEST_DB_URL = f"sqlite+aiosqlite:///{TEST_DB_FILE}"


@pytest_asyncio.fixture
async def test_engine():
    """
    Create a fresh async engine + tables for each test, drop them after.
    Default backend is SQLite for hermetic CI; override via TEST_DATABASE_URL env.
    """
    url = os.getenv("TEST_DATABASE_URL", TEST_DB_URL)

    # Import models inside the fixture so conftest can be imported without DB
    from db.models import Base

    engine = create_async_engine(url, echo=False, future=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield engine
    finally:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await engine.dispose()
        # Best-effort cleanup of the sqlite file
        if url.startswith("sqlite") and os.path.exists(TEST_DB_FILE):
            try:
                os.remove(TEST_DB_FILE)
            except OSError:
                pass


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """A single AsyncSession bound to the test engine."""
    SessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session


# ─────────────────────────────────────────────────────────────────────────────
# 3. APP + TestClient (sync) and AsyncClient
# ─────────────────────────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def app_with_test_db(test_engine):
    """
    Build the FastAPI app with:
      - the test engine + session wired in via dependency_overrides
      - the orchestrator (get_bot, stream_response) mocked
    """
    # Import after env vars are set so config.get_settings() doesn't crash
    from db.models import get_async_session, Base
    from app import app

    SessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)

    async def _override_get_session():
        async with SessionLocal() as session:
            yield session

    app.dependency_overrides[get_async_session] = _override_get_session

    # Mock the orchestrator at the module level used by the routers
    mock_bot = MagicMock()
    mock_bot.aupdate_state = AsyncMock(return_value=None)
    mock_bot.aget_state = AsyncMock(
        return_value=MagicMock(values={"messages": []})
    )

    async def fake_stream_response(user_message, thread_id, user_id):
        # Deterministic token stream — lets us assert on the SSE output
        for token in ["Hello", " ", "from", " ", "mocked", " ", "bot."]:
            yield token

    with patch("orchestrators.graph.get_bot", new=AsyncMock(return_value=mock_bot)), \
         patch("api.chat_router.get_bot", new=AsyncMock(return_value=mock_bot)), \
         patch("api.conversations.get_bot", new=AsyncMock(return_value=mock_bot)), \
         patch("orchestrators.graph.stream_response", side_effect=fake_stream_response), \
         patch("api.chat_router.stream_response", side_effect=fake_stream_response):
        # Also mock the document/image pipeline so chat_router doesn't try to load files
        with patch("api.chat_router.doc_pipeline", new=MagicMock(
            ainvoke=AsyncMock(return_value={
                "response": "mocked image description",
                "image_cached": False,
                "vector_store_ready": True,
                "error": "",
            })
        )):
            yield app

    app.dependency_overrides.clear()


@pytest.fixture
def client(app_with_test_db) -> Generator[TestClient, None, None]:
    """Synchronous TestClient for tests that don't need async streaming."""
    # httpx>=0.28 requires ASGITransport for ASGI apps in some configs
    with TestClient(app_with_test_db) as c:
        yield c


@pytest_asyncio.fixture
async def async_client(app_with_test_db) -> AsyncGenerator[AsyncClient, None]:
    """Async client — useful for SSE streaming assertions."""
    transport = ASGITransport(app=app_with_test_db)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# 4. AUTH HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def registered_user(client, app_with_test_db) -> dict:
    """
    Register a fresh user via the /auth/register endpoint, then return
    ``{"email", "password", "user_id", "token", "headers"}``.
    """
    # NOTE: pydantic's email-validator rejects `.local` TLD as reserved.
    # Use example.com — a real reserved-for-testing domain.
    email = f"user-{uuid.uuid4().hex[:8]}@example.com"
    password = "TestPassword123!"

    resp = client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )
    assert resp.status_code == 201, f"register failed: {resp.text}"

    # /auth/register returns the user (id, email). Now log in to get a JWT.
    login = client.post(
        "/auth/jwt/login",
        data={"username": email, "password": password},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    assert login.status_code == 200, f"login failed: {login.text}"
    token = login.json()["access_token"]

    return {
        "email": email,
        "password": password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest_asyncio.fixture
async def another_user(client) -> dict:
    """A second registered user — used to verify authorization isolation."""
    email = f"other-{uuid.uuid4().hex[:8]}@example.com"
    password = "OtherPass456!"
    client.post("/auth/register", json={"email": email, "password": password})
    login = client.post(
        "/auth/jwt/login",
        data={"username": email, "password": password},
        headers={"content-type": "application/x-www-form-urlencoded"},
    )
    token = login.json()["access_token"]
    return {
        "email": email,
        "password": password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MOCKED ORCHESTRATOR (direct, without HTTP)
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mock_bot():
    """Bare MagicMock stand-in for the LangGraph agent."""
    bot = MagicMock()
    bot.aupdate_state = AsyncMock(return_value=None)
    bot.aget_state = AsyncMock(return_value=MagicMock(values={"messages": []}))
    bot.astream = MagicMock(return_value=iter([]))  # used in some endpoints
    return bot


# ─────────────────────────────────────────────────────────────────────────────
# 6. PYTEST CONFIG (markers)
# ─────────────────────────────────────────────────────────────────────────────
def pytest_configure(config: pytest.Config):
    """Register custom markers."""
    config.addinivalue_line("markers", "db: database integration tests")
    config.addinivalue_line("markers", "auth: authentication / authorization tests")
    config.addinivalue_line("markers", "api: HTTP endpoint tests")
    config.addinivalue_line("markers", "orchestrator: LangGraph agent / pipeline tests")
    config.addinivalue_line("markers", "slow: long-running integration tests")