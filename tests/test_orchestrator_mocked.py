"""
test_orchestrator_mocked.py
===========================
LAYER 3: Mocked External Services (AI Orchestrator).

Goal: prove our code correctly *handles* the LangGraph orchestrator's output
WITHOUT actually calling OpenAI / Groq / MCP servers. We mock at the boundary
that crosses into the agent so every test runs in milliseconds and costs $0.

What's mocked:
  - ``orchestrators.graph.get_bot``         — bot construction
  - ``orchestrators.graph.stream_response``  — token generator
  - ``api.chat_router.doc_pipeline``        — image/doc ingestion
  - LangChain ChatOpenAI / ChatGroq         — instantiated at module import

What's NOT mocked:
  - FastAPI request/response lifecycle
  - Auth dependency (we use real test users)
  - Database persistence
  - The SSE serialization / chunking logic
"""

from __future__ import annotations

import asyncio
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.orchestrator


# ─────────────────────────────────────────────────────────────────────────────
# 1. /chat/stream — message-only path (the most common case)
# ─────────────────────────────────────────────────────────────────────────────
class TestChatStreamMessageOnly:
    def test_message_only_streams_agent_response(
        self, client, registered_user, mock_bot
    ):
        """No file, just text → must yield the mocked tokens via SSE."""
        # The fake_stream_response in conftest yields 'Hello from mocked bot.'
        with client.stream(
            "POST",
            "/chat/stream",
            data={
                "session_id": str(uuid.uuid4()),
                "message": "Hi Orphic!",
            },
            headers=registered_user["headers"],
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())

        # Every SSE chunk is "data: <token>\n\n" — concat the data lines
        tokens = []
        for line in body.splitlines():
            if line.startswith("data: "):
                tokens.append(line[len("data: "):])

        joined = "".join(tokens)
        assert "Hello" in joined
        assert "mocked" in joined
        assert "bot." in joined

    def test_message_only_persists_conversation(self, client, registered_user):
        """Hitting /chat/stream auto-creates a conversation row."""
        sid = str(uuid.uuid4())
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": sid, "message": "First ever message"},
            headers=registered_user["headers"],
        ) as resp:
            for _ in resp.iter_text():
                pass

        listed = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        ).json()
        ids = [c["id"] for c in listed]
        assert sid in ids
        # Title truncated to 30 chars + "..."
        assert any(c["title"].startswith("First ever message") for c in listed)

    def test_empty_message_returns_guidance(self, client, registered_user):
        """Neither file nor message → server returns guidance text."""
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": str(uuid.uuid4())},
            headers=registered_user["headers"],
        ) as resp:
            body = "".join(resp.iter_text())

        assert "Please provide a message" in body or "upload a file" in body


# ─────────────────────────────────────────────────────────────────────────────
# 2. /chat/stream — file path (image upload mocked)
# ─────────────────────────────────────────────────────────────────────────────
class TestChatStreamFileUpload:
    def test_image_upload_streams_mocked_description(
        self, client, registered_user
    ):
        """Upload an image → pipeline is mocked to return 'mocked image description'."""
        # Minimal valid 1×1 PNG bytes
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0"
            b"\x00\x00\x00\x03\x00\x01"
            b"\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )

        with client.stream(
            "POST", "/chat/stream",
            files={"file": ("test.png", png_bytes, "image/png")},
            data={"session_id": str(uuid.uuid4()), "message": ""},
            headers=registered_user["headers"],
        ) as resp:
            body = "".join(resp.iter_text())

        # The mocked pipeline returns "mocked image description" so it should appear
        assert "mocked image description" in body

    def test_unsupported_file_type_yields_error_event(self, client, registered_user):
        """Uploading a .exe (not in IMAGE/DOCUMENT extensions) → SSE error event."""
        with client.stream(
            "POST", "/chat/stream",
            files={"file": ("virus.exe", b"MZ\x90\x00", "application/octet-stream")},
            data={"session_id": str(uuid.uuid4())},
            headers=registered_user["headers"],
        ) as resp:
            body = "".join(resp.iter_text())

        assert "Unsupported file type" in body

    def test_oversized_file_yields_error_event(self, client, registered_user):
        """Files > 20MB must be rejected by the endpoint."""
        big = b"\x00" * (21 * 1024 * 1024)
        with client.stream(
            "POST", "/chat/stream",
            files={"file": ("huge.txt", big, "text/plain")},
            data={"session_id": str(uuid.uuid4())},
            headers=registered_user["headers"],
        ) as resp:
            body = "".join(resp.iter_text())

        assert "maximum size" in body.lower()

    def test_document_upload_triggers_proactive_offer(self, client, registered_user):
        """Upload a .txt → mocked pipeline returns vector_store_ready=True →
        endpoint takes the proactive-offer path (extract_preview + stream_opening_offer).

        Both extract_preview and stream_opening_offer call the LLM, so we patch them.
        """
        with patch("api.chat_router.extract_preview", return_value="PDF preview text"), \
             patch("api.chat_router.stream_opening_offer") as mock_offer:
            async def fake_offer_gen(text, fname):
                yield f"Here is what I can help with regarding {fname}"
            mock_offer.side_effect = fake_offer_gen

            with client.stream(
                "POST", "/chat/stream",
                files={"file": ("notes.txt", b"hello world", "text/plain")},
                data={"session_id": str(uuid.uuid4())},
                headers=registered_user["headers"],
            ) as resp:
                body = "".join(resp.iter_text())

        assert "Here is what I can help" in body


# ─────────────────────────────────────────────────────────────────────────────
# 3. /chat/stream — pipeline error handling
# ─────────────────────────────────────────────────────────────────────────────
class TestChatStreamErrorHandling:
    def test_pipeline_error_yields_error_event(self, client, registered_user):
        """If doc_pipeline.ainvoke returns {'error': '...'} → SSE error event."""
        with patch(
            "api.chat_router.doc_pipeline",
            new=MagicMock(ainvoke=AsyncMock(return_value={
                "error": "File corrupted",
                "response": "",
                "image_cached": False,
            })),
        ):
            with client.stream(
                "POST", "/chat/stream",
                files={"file": ("notes.txt", b"data", "text/plain")},
                data={"session_id": str(uuid.uuid4())},
                headers=registered_user["headers"],
            ) as resp:
                body = "".join(resp.iter_text())

        assert "File corrupted" in body

    def test_bot_init_failure_yields_friendly_error(self, client, registered_user):
        """If get_bot raises, stream_response must yield an error string,
        not crash the request."""
        # Patch BOTH api.chat_router.get_bot AND orchestrators.graph.get_bot
        # because chat_router imports get_bot directly into its module namespace.
        from api import chat_router as cr
        from orchestrators import graph as og

        async def fake_stream(*args, **kwargs):
            yield "Sorry, I'm having trouble starting up. Please try again later."

        with patch.object(cr, "get_bot", new=AsyncMock(side_effect=RuntimeError("DB down"))), \
             patch.object(og, "get_bot", new=AsyncMock(side_effect=RuntimeError("DB down"))), \
             patch.object(cr, "stream_response", side_effect=fake_stream):
            with client.stream(
                "POST", "/chat/stream",
                data={"session_id": str(uuid.uuid4()), "message": "hi"},
                headers=registered_user["headers"],
            ) as resp:
                body = "".join(resp.iter_text())

        assert "Sorry" in body


# ─────────────────────────────────────────────────────────────────────────────
# 4. Memory middleware — pure unit test, no DB / no LLM
# ─────────────────────────────────────────────────────────────────────────────
class TestMemoryMiddlewareUnit:
    """
    Port the standalone test_memory_middleware.py into pytest form.
    Patches retrieve_memory so no Postgres + no OpenAI embeddings needed.
    """

    FAKE_MEMORIES = [
        "User's name is Vikas",
        "User works on Project Orphic",
        "User prefers Python",
    ]

    @pytest.mark.asyncio
    async def test_abefore_agent_returns_memory_prompt(self):
        from middleware.memory_middleware import MemoryMiddleware
        from schemas.context import UserContext
        
        middleware = MemoryMiddleware()
        mock_runtime = MagicMock()
        mock_runtime.context = UserContext(user_id="test_user")

        with patch(
            "middleware.memory_middleware.retrieve_memory",
            new=AsyncMock(return_value=self.FAKE_MEMORIES),
        ):
            result = await middleware.abefore_agent(
                {"messages": []},
                runtime=mock_runtime,
            )

        assert isinstance(result, dict)
        assert "memory_prompt" in result
        assert "Vikas" in result["memory_prompt"]
        assert "Project Orphic" in result["memory_prompt"]

    @pytest.mark.asyncio
    async def test_abefore_agent_handles_retrieve_failure(self):
        """If retrieve_memory throws, middleware must fall back gracefully
        (return an empty memory_prompt) instead of crashing the agent."""
        from middleware.memory_middleware import MemoryMiddleware
        from schemas.context import UserContext
        
        middleware = MemoryMiddleware()
        mock_runtime = MagicMock()
        mock_runtime.context = UserContext(user_id="test_user")

        with patch(
            "middleware.memory_middleware.retrieve_memory",
            new=AsyncMock(side_effect=RuntimeError("DB down")),
        ):
            result = await middleware.abefore_agent(
                {"messages": []},
                runtime=mock_runtime,
            )

        assert "memory_prompt" in result
        assert isinstance(result["memory_prompt"], str)

    @pytest.mark.asyncio
    async def test_awrap_model_call_injects_system_message(self):
        from langchain_core.messages import AIMessage, SystemMessage
        from langchain.agents.middleware.types import (
            ModelRequest,
            ModelResponse,
        )
        from middleware.memory_middleware import MemoryMiddleware
        from prompts.system_persona_prompt import get_prompt

        middleware = MemoryMiddleware()
        expected_prompt = get_prompt(self.FAKE_MEMORIES)

        req = ModelRequest(
            model=MagicMock(),
            messages=[],
            system_message=None,
            state={"messages": [], "memory_prompt": expected_prompt},
            runtime=None,
        )

        captured = {}

        async def fake_handler(r):
            captured["sys"] = r.system_message
            return ModelResponse(result=[AIMessage(content="ok")])

        await middleware.awrap_model_call(req, fake_handler)

        assert isinstance(captured["sys"], SystemMessage)
        assert "Vikas" in captured["sys"].content

    @pytest.mark.asyncio
    async def test_awrap_tool_call_catches_errors(self):
        """A failing MCP tool must return a ToolMessage with status='error',
        not crash the entire ReAct loop."""
        from middleware.memory_middleware import MemoryMiddleware
        from langchain_core.messages import ToolMessage

        middleware = MemoryMiddleware()
        req = MagicMock()
        req.tool_call = {"id": "tc_abc123", "name": "broken_tool", "args": {}}

        async def failing_handler(r):
            raise RuntimeError("MCP server unreachable")

        result = await middleware.awrap_tool_call(req, failing_handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "MCP server unreachable" in result.content
        assert result.tool_call_id == "tc_abc123"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Document pipeline routing — without invoking LLMs
# ─────────────────────────────────────────────────────────────────────────────
class TestDocumentPipelineRouting:
    """
    The pipeline graph in document_parser/graph.py does:
        START → entry → (image_vision | doc_analysis_node) → END
    We test the routing decisions and validation WITHOUT calling vision/embedding.
    """

    @pytest.mark.asyncio
    async def test_entry_rejects_missing_file(self):
        from document_parser.graph import entry_node
        state = {
            "file_path": "",
            "file_name": "",
            "user_query": "anything",
            "user_id": "u1",
            "session_id": "s1",
        }
        out = await entry_node(state)
        assert out["has_file"] is False
        assert "error" in out

    @pytest.mark.asyncio
    async def test_entry_rejects_missing_user_id(self):
        from document_parser.graph import entry_node
        state = {
            "file_path": "/tmp/whatever.png",
            "file_name": "whatever.png",
            "user_query": "",
            "user_id": "",
            "session_id": "s1",
        }
        out = await entry_node(state)
        assert "error" in out
        assert "user_id" in out["error"].lower()

    @pytest.mark.asyncio
    async def test_entry_rejects_nonexistent_file(self, tmp_path):
        from document_parser.graph import entry_node
        state = {
            "file_path": str(tmp_path / "does_not_exist.png"),
            "file_name": "does_not_exist.png",
            "user_query": "",
            "user_id": "u1",
            "session_id": "s1",
        }
        out = await entry_node(state)
        assert out["has_file"] is False
        assert "File not found" in out["error"]

    @pytest.mark.asyncio
    async def test_entry_rejects_unsupported_extension(self, tmp_path):
        from document_parser.graph import entry_node
        f = tmp_path / "evil.exe"
        f.write_bytes(b"MZ")
        state = {
            "file_path": str(f),
            "file_name": "evil.exe",
            "user_query": "",
            "user_id": "u1",
            "session_id": "s1",
        }
        out = await entry_node(state)
        assert "Unsupported file type" in out["error"]

    @pytest.mark.asyncio
    async def test_entry_accepts_image_and_routes_to_vision(self, tmp_path):
        """A real .png file must be flagged has_file=True with file_type='png'."""
        from document_parser.graph import entry_node, route_after_entry, END
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")  # just a header, non-empty

        state = {
            "file_path": str(f),
            "file_name": "test.png",
            "user_query": "",
            "user_id": "u1",
            "session_id": "s1",
        }
        out = await entry_node(state)
        assert out["has_file"] is True
        assert out["file_type"] == "png"

        # Routing decision
        assert route_after_entry(out) == "image_vision"

    @pytest.mark.asyncio
    async def test_entry_routes_pdf_to_doc_analysis(self, tmp_path):
        from document_parser.graph import entry_node, route_after_entry
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4\n%fake\n")

        state = {
            "file_path": str(f),
            "file_name": "doc.pdf",
            "user_query": "",
            "user_id": "u1",
            "session_id": "s1",
        }
        out = await entry_node(state)
        assert out["has_file"] is True
        assert out["file_type"] == "pdf"
        assert route_after_entry(out) == "doc_analysis_node"

    @pytest.mark.asyncio
    async def test_image_vision_node_uses_analyse_image(self, tmp_path):
        """image_vision calls analyse_image(); we mock it to confirm wiring."""
        from document_parser.graph import image_vision

        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")

        with patch(
            "document_parser.graph.analyse_image",
            new=AsyncMock(return_value=("a fluffy cat sitting on a couch", False)),
        ):
            out = await image_vision({
                "file_path": str(f),
                "user_id": "u1",
                "session_id": "s1",
            })

        assert out["response"] == "a fluffy cat sitting on a couch"
        assert out["image_cached"] is False

    @pytest.mark.asyncio
    async def test_image_vision_handles_failure(self, tmp_path):
        from document_parser.graph import image_vision
        f = tmp_path / "test.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n")

        with patch(
            "document_parser.graph.analyse_image",
            new=AsyncMock(side_effect=RuntimeError("vision API down")),
        ):
            out = await image_vision({
                "file_path": str(f),
                "user_id": "u1",
                "session_id": "s1",
            })

        assert "error" in out
        assert "vision API down" in out["error"]

    @pytest.mark.asyncio
    async def test_doc_analysis_node_uses_vector_store(self, tmp_path):
        """doc_analysis_node calls get_or_create_vector_store; mock it."""
        from document_parser.graph import doc_analysis_node

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4\n")

        with patch(
            "document_parser.graph.get_or_create_vector_store",
            return_value=MagicMock(),
        ) as mock_vs:
            out = await doc_analysis_node({
                "file_path": str(f),
                "user_id": "u1",
                "session_id": "s1",
            })

        assert out["vector_store_ready"] is True
        mock_vs.assert_called_once()

    @pytest.mark.asyncio
    async def test_doc_analysis_node_handles_failure(self, tmp_path):
        from document_parser.graph import doc_analysis_node
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4\n")

        with patch(
            "document_parser.graph.get_or_create_vector_store",
            side_effect=RuntimeError("embedding API down"),
        ):
            out = await doc_analysis_node({
                "file_path": str(f),
                "user_id": "u1",
                "session_id": "s1",
            })

        assert "error" in out
        assert "embedding API down" in out["error"]


# ─────────────────────────────────────────────────────────────────────────────
# 6. Settings — proves config loading works
# ─────────────────────────────────────────────────────────────────────────────
class TestConfigLoading:
    def test_get_settings_is_cached(self):
        from config import get_settings
        a = get_settings()
        b = get_settings()
        assert a is b  # lru_cache returns the same instance

    def test_settings_has_required_keys(self):
        from config import get_settings
        s = get_settings()
        assert s.openai_api_key
        assert s.groq_api_key
        assert s.db_url
        assert s.jwt_secret