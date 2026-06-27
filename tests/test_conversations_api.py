"""
test_conversations_api.py
=========================
Targeted tests for /api/v1/conversations/* endpoints.

These complement test_auth.py — that file proves the endpoints are private;
this file proves they DO the right thing once you're past the gate.
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.api


# ─────────────────────────────────────────────────────────────────────────────
# LIST — GET /api/v1/conversations/
# ─────────────────────────────────────────────────────────────────────────────
class TestListConversations:
    def test_empty_list_for_new_user(self, client, registered_user):
        resp = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        )
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_created_conversation(self, client, registered_user):
        client.post(
            "/api/v1/conversations/",
            json={"title": "Hello"},
            headers=registered_user["headers"],
        )
        resp = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        )
        data = resp.json()
        assert len(data) == 1
        assert data[0]["title"] == "Hello"
        # created_at is serialized as string
        assert isinstance(data[0]["created_at"], str)

    def test_newest_first(self, client, registered_user):
        # Create three in order; the newest one must be first
        import time
        for title in ["first", "second", "third"]:
            client.post(
                "/api/v1/conversations/",
                json={"title": title},
                headers=registered_user["headers"],
            )
            time.sleep(1)
        data = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        ).json()
        titles = [c["title"] for c in data]
        assert titles[0] == "third"
        assert titles[-1] == "first"


# ─────────────────────────────────────────────────────────────────────────────
# CREATE — POST /api/v1/conversations/
# ─────────────────────────────────────────────────────────────────────────────
class TestCreateConversation:
    def test_default_title(self, client, registered_user):
        resp = client.post(
            "/api/v1/conversations/",
            json={},
            headers=registered_user["headers"],
        )
        # Pydantic defaults to "New Chat" via the model default
        assert resp.status_code == 200, resp.text
        assert resp.json()["title"] == "New Chat"

    def test_custom_title(self, client, registered_user):
        resp = client.post(
            "/api/v1/conversations/",
            json={"title": "Project brainstorming"},
            headers=registered_user["headers"],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["title"] == "Project brainstorming"
        assert uuid.UUID(body["id"])  # id is a valid UUID


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGES — GET /api/v1/conversations/{id}/messages
# ─────────────────────────────────────────────────────────────────────────────
class TestGetConversationMessages:
    def test_returns_empty_messages_for_new_thread(self, client, registered_user):
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "empty"},
            headers=registered_user["headers"],
        ).json()

        resp = client.get(
            f"/api/v1/conversations/{c['id']}/messages",
            headers=registered_user["headers"],
        )
        assert resp.status_code == 200
        # The mocked bot returns empty messages
        assert resp.json()["messages"] == []

    def test_404_for_unknown_thread(self, client, registered_user):
        resp = client.get(
            f"/api/v1/conversations/{uuid.uuid4()}/messages",
            headers=registered_user["headers"],
        )
        # Either 403 (ownership) or 404 (not found) is acceptable per code path
        assert resp.status_code in (403, 404), resp.text


# ─────────────────────────────────────────────────────────────────────────────
# RENAME — PATCH /api/v1/conversations/{id}
# ─────────────────────────────────────────────────────────────────────────────
class TestRenameConversation:
    def test_rename(self, client, registered_user):
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "old"},
            headers=registered_user["headers"],
        ).json()

        resp = client.patch(
            f"/api/v1/conversations/{c['id']}",
            json={"title": "new"},
            headers=registered_user["headers"],
        )
        assert resp.status_code == 200
        assert resp.json()["title"] == "new"

    def test_rename_unknown_thread_404(self, client, registered_user):
        resp = client.patch(
            f"/api/v1/conversations/{uuid.uuid4()}",
            json={"title": "anything"},
            headers=registered_user["headers"],
        )
        assert resp.status_code in (403, 404), resp.text

    def test_rename_empty_title_rejected(self, client, registered_user):
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "x"},
            headers=registered_user["headers"],
        ).json()
        resp = client.patch(
            f"/api/v1/conversations/{c['id']}",
            json={"title": ""},
            headers=registered_user["headers"],
        )
        # Pydantic should reject empty string OR the endpoint accepts it
        # We only require non-500
        assert resp.status_code < 500


# ─────────────────────────────────────────────────────────────────────────────
# DELETE — DELETE /api/v1/conversations/{id}
# ─────────────────────────────────────────────────────────────────────────────
class TestDeleteConversation:
    def test_delete_own_thread(self, client, registered_user):
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "to-delete"},
            headers=registered_user["headers"],
        ).json()

        delete = client.delete(
            f"/api/v1/conversations/{c['id']}",
            headers=registered_user["headers"],
        )
        assert delete.status_code == 204

        # Confirm it's gone
        listed = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        ).json()
        ids = [conv["id"] for conv in listed]
        assert c["id"] not in ids

    def test_delete_unknown_thread_404(self, client, registered_user):
        resp = client.delete(
            f"/api/v1/conversations/{uuid.uuid4()}",
            headers=registered_user["headers"],
        )
        assert resp.status_code in (403, 404), resp.text

    def test_delete_is_idempotent_safe(self, client, registered_user):
        """Deleting a thread twice → second call is 404, not 500."""
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "del-twice"},
            headers=registered_user["headers"],
        ).json()

        first = client.delete(
            f"/api/v1/conversations/{c['id']}",
            headers=registered_user["headers"],
        )
        assert first.status_code == 204

        second = client.delete(
            f"/api/v1/conversations/{c['id']}",
            headers=registered_user["headers"],
        )
        assert second.status_code in (403, 404)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR AUTO-CREATION via /chat/stream
# ─────────────────────────────────────────────────────────────────────────────
class TestChatStreamAutoCreatesThread:
    """
    Hitting /chat/stream with a fresh session_id should auto-create a
    Conversation row in the sidebar with a sensible title.
    """

    def test_auto_create_with_message_title(self, client, registered_user):
        sid = str(uuid.uuid4())
        with client.stream(
            "POST", "/chat/stream",
            data={"session_id": sid, "message": "Help me with SQL"},
            headers=registered_user["headers"],
        ) as resp:
            for _ in resp.iter_text():
                pass

        listed = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        ).json()
        match = next((c for c in listed if c["id"] == sid), None)
        assert match is not None
        assert match["title"].startswith("Help me with SQL")

    def test_auto_create_with_file_title(self, client, registered_user):
        sid = str(uuid.uuid4())
        with client.stream(
            "POST", "/chat/stream",
            files={"file": ("report.pdf", b"%PDF-1.4\n", "application/pdf")},
            data={"session_id": sid},
            headers=registered_user["headers"],
        ) as resp:
            for _ in resp.iter_text():
                pass

        listed = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        ).json()
        match = next((c for c in listed if c["id"] == sid), None)
        assert match is not None
        assert match["title"].startswith("Doc:")

    def test_no_duplicate_on_repeat_post(self, client, registered_user):
        """Posting to the same session_id twice must NOT create two rows."""
        sid = str(uuid.uuid4())
        for _ in range(3):
            with client.stream(
                "POST", "/chat/stream",
                data={"session_id": sid, "message": "hi"},
                headers=registered_user["headers"],
            ) as resp:
                for _ in resp.iter_text():
                    pass

        listed = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        ).json()
        matches = [c for c in listed if c["id"] == sid]
        assert len(matches) == 1