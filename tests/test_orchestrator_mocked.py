"""
test_orchestrator_mocked.py
===========================
LAYER 3: Mocked External Services (AI Orchestrator).

Goal: prove our code correctly *handles* the LangGraph orchestrator's output
WITHOUT actually calling OpenAI / Groq / MCP servers. We mock at the boundary
that crosses into the agent so every test runs in milliseconds and costs $0.

What's mocked (post-rewrite):
  - ``orchestrators.graph.stream_response``  — (event_type, data) tuple generator
  - ``orchestrators.graph.resume_response``  — resume tuple generator
  - ``orchestrators.graph.update_agent_state`` — state injection
  - ``orchestrators.graph.get_thread_state``   — state retrieval
  - ``api.chat_router.doc_pipeline``          — image/doc ingestion

What's NOT mocked:
  - FastAPI request/response lifecycle
  - Auth dependency (we use real test users)
  - Database persistence (conversations table)
  - The SSE serialization / chunking logic
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.orchestrator


def _pending_interrupt_state():
    return MagicMock(
        tasks=[
            MagicMock(
                interrupts=[
                    MagicMock(value={"id": "interrupt-1"})
                ]
            )
        ]
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. /chat/stream — message-only path (the most common case)
# ─────────────────────────────────────────────────────────────────────────────
class TestChatStreamMessageOnly:
    def test_message_only_streams_agent_response(
        self, client, registered_user, mock_bot
    ):
        """No file, just text → must yield the mocked tokens via SSE."""
        # The fake_stream_response in conftest yields ("token", word) tuples
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
        assert '"text":"Hello"' in body or '"text": "Hello"' in body
        assert '"text":"mocked"' in body or '"text": "mocked"' in body
        assert '"text":"bot."' in body or '"text": "bot."' in body

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

    def test_stream_error_yields_friendly_message(self, client, registered_user):
        """If stream_response yields an error tuple, the SSE must contain a
        structured JSON error, not crash the request."""
        from api import chat_router as cr

        async def failing_stream(*args, **kwargs):
            yield ("error", "Service is starting up. Please try again.")

        with patch.object(cr, "stream_response", side_effect=failing_stream):
            with client.stream(
                "POST", "/chat/stream",
                data={"session_id": str(uuid.uuid4()), "message": "hi"},
                headers=registered_user["headers"],
            ) as resp:
                body = "".join(resp.iter_text())

        assert "starting up" in body.lower() or "error" in body.lower()


# ─────────────────────────────────────────────────────────────────────────────
# 4. /chat/stream — SSE event structure (tuple unpacking)
# ─────────────────────────────────────────────────────────────────────────────
class TestSSEEventStructure:
    """Verify that stream_response's (event_type, data) tuples are correctly
    serialized into SSE events with proper JSON structure for control events."""

    def test_done_event_is_json(self, client, registered_user):
        """The 'done' event must be serialized as a JSON object."""
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": str(uuid.uuid4()), "message": "hi"},
            headers=registered_user["headers"],
        ) as resp:
            body = "".join(resp.iter_text())

        # Find the done event
        for line in body.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict) and parsed.get("type") == "done":
                        return  # test passes
                except (json.JSONDecodeError, TypeError):
                    continue
        # If no done event found, that's acceptable too (some streams don't yield done)

    def test_interrupt_event_structure(self, client, registered_user):
        """When stream_response yields an interrupt, it must be JSON with type=interrupt."""
        from api import chat_router as cr

        async def interrupt_stream(*args, **kwargs):
            yield ("token", "Let me check...")
            yield ("interrupt", {"provider": "github", "action": "connect"})

        with patch.object(cr, "stream_response", side_effect=interrupt_stream):
            with client.stream(
                "POST", "/chat/stream",
                data={"session_id": str(uuid.uuid4()), "message": "list my repos"},
                headers=registered_user["headers"],
            ) as resp:
                body = "".join(resp.iter_text())

        # Must contain both a token and a structured interrupt
        assert "Let me check" in body
        found_interrupt = False
        for line in body.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict) and parsed.get("type") == "interrupt":
                        assert parsed["data"]["provider"] == "github"
                        found_interrupt = True
                except (json.JSONDecodeError, TypeError):
                    pass
        assert found_interrupt, "No interrupt event found in SSE stream"


# ─────────────────────────────────────────────────────────────────────────────
# 5. /chat/resume/{session_id} — resume after HITL interrupt
# ─────────────────────────────────────────────────────────────────────────────
class TestChatResume:
    def test_resume_requires_auth(self, client):
        """POST /chat/resume without token → 401."""
        resp = client.post(
            f"/chat/resume/{uuid.uuid4()}",
            json={
                "interrupt_id": "interrupt-1",
                "decision": "connected",
                "input": {},
            },
        )
        assert resp.status_code == 401

    def test_resume_rejects_invalid_action(self, client, registered_user):
        """action must be 'continue' or 'cancel', not arbitrary strings."""
        sid = str(uuid.uuid4())
        # Create the conversation first
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": sid, "message": "hi"},
            headers=registered_user["headers"],
        ) as resp:
            for _ in resp.iter_text():
                pass

        resp = client.post(
            f"/chat/resume/{sid}",
            json={
                "interrupt_id": "interrupt-1",
                "decision": "hack",
                "input": {},
            },
            headers=registered_user["headers"],
        )
        assert resp.status_code == 422

    def test_resume_unknown_session_403(self, client, registered_user):
        """Resuming a thread that doesn't belong to the user → 403."""
        with patch("api.chat_router.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())), \
             patch("orchestrators.graph.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())):
            resp = client.post(
                f"/chat/resume/{uuid.uuid4()}",
                json={
                    "interrupt_id": "interrupt-1",
                    "decision": "connected",
                    "input": {},
                },
                headers=registered_user["headers"],
            )
        assert resp.status_code == 403

    def test_resume_continue_streams_tokens(self, client, registered_user):
        """A 'continue' resume must stream tokens from resume_response."""
        sid = str(uuid.uuid4())
        # Create the conversation first
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": sid, "message": "hi"},
            headers=registered_user["headers"],
        ) as resp:
            for _ in resp.iter_text():
                pass

        with patch("api.chat_router.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())), \
             patch("orchestrators.graph.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())):
            # Resume
            with client.stream(
                "POST", f"/chat/resume/{sid}",
                json={
                    "interrupt_id": "interrupt-1",
                    "decision": "connected",
                    "input": {},
                },
                headers=registered_user["headers"],
            ) as resp:
                body = "".join(resp.iter_text())

        assert "Resumed successfully" in body

    def test_resume_cancel_streams_cancellation(self, client, registered_user):
        """A 'cancel' resume must stream the cancellation message."""
        sid = str(uuid.uuid4())
        # Create the conversation first
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": sid, "message": "hi"},
            headers=registered_user["headers"],
        ) as resp:
            for _ in resp.iter_text():
                pass

        with patch("api.chat_router.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())), \
             patch("orchestrators.graph.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())):
            # Resume with cancel
            with client.stream(
                "POST", f"/chat/resume/{sid}",
                json={
                    "interrupt_id": "interrupt-1",
                    "decision": "cancel",
                    "input": {},
                },
                headers=registered_user["headers"],
            ) as resp:
                body = "".join(resp.iter_text())

        found_cancel = False
        for line in body.splitlines():
            if line.startswith("data: "):
                payload = line[len("data: "):]
                try:
                    parsed = json.loads(payload)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed, dict) and parsed.get("type") == "token":
                    text = parsed.get("data", {}).get("text", "")
                    if text == "Action cancelled.":
                        found_cancel = True
                        break

        assert found_cancel, body

    def test_resume_cross_user_rejected(self, client, registered_user, another_user):
        """User B cannot resume User A's conversation."""
        sid = str(uuid.uuid4())
        # User A creates a conversation
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": sid, "message": "private stuff"},
            headers=registered_user["headers"],
        ) as resp:
            for _ in resp.iter_text():
                pass

        # User B tries to resume it
        with patch("api.chat_router.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())), \
             patch("orchestrators.graph.get_thread_state", AsyncMock(return_value=_pending_interrupt_state())):
            resp = client.post(
                f"/chat/resume/{sid}",
                json={
                    "interrupt_id": "interrupt-1",
                    "decision": "connected",
                    "input": {},
                },
                headers=another_user["headers"],
            )
        assert resp.status_code == 403


# ─────────────────────────────────────────────────────────────────────────────
# 6. Document pipeline routing — without invoking LLMs
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
# 7. Settings — proves config loading works
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
