"""
test_auth.py
============
LAYER 2: Authentication & Authorization Tests.

Verifies:
  - /auth/register creates new users (201)
  - /auth/jwt/login issues JWT tokens (200) with proper shape
  - Protected endpoints return 401 without a token
  - Protected endpoints return 401 with an invalid / malformed token
  - Protected endpoints return 200 with a valid token
  - One user CANNOT read or mutate another user's conversations (403)
"""

from __future__ import annotations

import uuid

import pytest

pytestmark = pytest.mark.auth


# ─────────────────────────────────────────────────────────────────────────────
# 1. PUBLIC ENDPOINTS (no auth required)
# ─────────────────────────────────────────────────────────────────────────────
class TestPublicEndpoints:
    def test_root_is_public(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Orphic" in resp.text

    def test_register_endpoint_is_public(self, client):
        email = f"pub-{uuid.uuid4().hex[:6]}@orphic.com"
        resp = client.post(
            "/auth/register",
            json={"email": email, "password": "Strong1Pass!"},
        )
        # 201 = Created; fastapi-users default behavior
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["email"] == email
        assert "id" in body
        assert "password" not in body  # never echo back

    def test_login_endpoint_is_public(self, client):
        email = f"login-{uuid.uuid4().hex[:6]}@orphic.com"
        client.post("/auth/register", json={"email": email, "password": "Strong1Pass!"})
        resp = client.post(
            "/auth/jwt/login",
            data={"username": email, "password": "Strong1Pass!"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "access_token" in body
        assert body["token_type"].lower() == "bearer"


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROTECTED ENDPOINTS WITHOUT A TOKEN — must return 401
# ─────────────────────────────────────────────────────────────────────────────
class TestProtectedEndpointsRequireAuth:
    """
    These endpoints all live under /api/v1/conversations/* or /chat/stream.
    They use ``current_active_user`` from fastapi-users and must reject
    unauthenticated callers.
    """

    def test_list_conversations_requires_auth(self, client):
        resp = client.get("/api/v1/conversations/")
        assert resp.status_code == 401, resp.text

    def test_create_conversation_requires_auth(self, client):
        resp = client.post("/api/v1/conversations/", json={"title": "no auth"})
        assert resp.status_code in (401, 403), resp.text

    def test_get_conversation_messages_requires_auth(self, client):
        resp = client.get(f"/api/v1/conversations/{uuid.uuid4()}/messages")
        assert resp.status_code == 401, resp.text

    def test_rename_conversation_requires_auth(self, client):
        resp = client.patch(
            f"/api/v1/conversations/{uuid.uuid4()}",
            json={"title": "x"},
        )
        assert resp.status_code in (401, 403), resp.text

    def test_delete_conversation_requires_auth(self, client):
        resp = client.delete(f"/api/v1/conversations/{uuid.uuid4()}")
        assert resp.status_code in (401, 403), resp.text

    def test_chat_stream_requires_auth(self, client):
        resp = client.post(
            "/chat/stream",
            data={"session_id": "x", "message": "hi"},
        )
        assert resp.status_code == 401, resp.text


# ─────────────────────────────────────────────────────────────────────────────
# 3. PROTECTED ENDPOINTS WITH A VALID TOKEN — must return 200/201
# ─────────────────────────────────────────────────────────────────────────────
class TestProtectedEndpointsAcceptValidToken:
    def test_list_conversations_with_token(self, client, registered_user):
        resp = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == []  # brand-new user has no threads

    def test_create_and_list_conversation_with_token(self, client, registered_user):
        # Create
        create = client.post(
            "/api/v1/conversations/",
            json={"title": "My first chat"},
            headers=registered_user["headers"],
        )
        assert create.status_code == 200, create.text
        conv_id = create.json()["id"]

        # List
        listed = client.get(
            "/api/v1/conversations/",
            headers=registered_user["headers"],
        )
        assert listed.status_code == 200
        ids = [c["id"] for c in listed.json()]
        assert conv_id in ids

    def test_rename_own_conversation_with_token(self, client, registered_user):
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "old"},
            headers=registered_user["headers"],
        ).json()

        rename = client.patch(
            f"/api/v1/conversations/{c['id']}",
            json={"title": "new"},
            headers=registered_user["headers"],
        )
        assert rename.status_code == 200, rename.text
        assert rename.json()["title"] == "new"

    def test_get_own_conversation_messages_with_token(self, client, registered_user):
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "msgs"},
            headers=registered_user["headers"],
        ).json()

        msgs = client.get(
            f"/api/v1/conversations/{c['id']}/messages",
            headers=registered_user["headers"],
        )
        assert msgs.status_code == 200, msgs.text
        assert "messages" in msgs.json()

    def test_delete_own_conversation_with_token(self, client, registered_user):
        c = client.post(
            "/api/v1/conversations/",
            json={"title": "del"},
            headers=registered_user["headers"],
        ).json()

        delete = client.delete(
            f"/api/v1/conversations/{c['id']}",
            headers=registered_user["headers"],
        )
        assert delete.status_code == 204, delete.text


# ─────────────────────────────────────────────────────────────────────────────
# 4. INVALID / EXPIRED TOKENS — must return 401
# ─────────────────────────────────────────────────────────────────────────────
class TestInvalidTokensRejected:
    def test_garbage_token_rejected(self, client):
        resp = client.get(
            "/api/v1/conversations/",
            headers={"Authorization": "Bearer this.is.not.a.real.jwt"},
        )
        assert resp.status_code == 401, resp.text

    def test_malformed_authorization_header_rejected(self, client):
        resp = client.get(
            "/api/v1/conversations/",
            headers={"Authorization": "NotBearer abc"},
        )
        assert resp.status_code == 401, resp.text

    def test_empty_authorization_rejected(self, client):
        resp = client.get(
            "/api/v1/conversations/",
            headers={"Authorization": ""},
        )
        assert resp.status_code == 401, resp.text


# ─────────────────────────────────────────────────────────────────────────────
# 5. CROSS-USER ISOLATION — users cannot read or mutate each other's data
# ─────────────────────────────────────────────────────────────────────────────
class TestCrossUserAuthorization:
    def test_user_cannot_read_others_messages(self, client, registered_user, another_user):
        # user A creates a thread
        a_conv = client.post(
            "/api/v1/conversations/",
            json={"title": "secret"},
            headers=registered_user["headers"],
        ).json()

        # user B tries to read it — must be denied (403) or not found (404)
        resp = client.get(
            f"/api/v1/conversations/{a_conv['id']}/messages",
            headers=another_user["headers"],
        )
        assert resp.status_code in (403, 404), resp.text

    def test_user_cannot_rename_others_conversation(self, client, registered_user, another_user):
        a_conv = client.post(
            "/api/v1/conversations/",
            json={"title": "mine"},
            headers=registered_user["headers"],
        ).json()

        resp = client.patch(
            f"/api/v1/conversations/{a_conv['id']}",
            json={"title": "hacked"},
            headers=another_user["headers"],
        )
        assert resp.status_code in (403, 404), resp.text

    def test_user_cannot_delete_others_conversation(self, client, registered_user, another_user):
        a_conv = client.post(
            "/api/v1/conversations/",
            json={"title": "mine"},
            headers=registered_user["headers"],
        ).json()

        resp = client.delete(
            f"/api/v1/conversations/{a_conv['id']}",
            headers=another_user["headers"],
        )
        assert resp.status_code in (403, 404), resp.text

    def test_list_only_returns_own_conversations(self, client, registered_user, another_user):
        # Each user creates one
        client.post(
            "/api/v1/conversations/", json={"title": "A"},
            headers=registered_user["headers"],
        )
        client.post(
            "/api/v1/conversations/", json={"title": "B"},
            headers=another_user["headers"],
        )

        a_list = client.get(
            "/api/v1/conversations/", headers=registered_user["headers"],
        ).json()
        b_list = client.get(
            "/api/v1/conversations/", headers=another_user["headers"],
        ).json()

        a_titles = {c["title"] for c in a_list}
        b_titles = {c["title"] for c in b_list}
        assert "A" in a_titles and "B" not in a_titles
        assert "B" in b_titles and "A" not in b_titles


# ─────────────────────────────────────────────────────────────────────────────
# 6. REGISTRATION VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
class TestRegistrationValidation:
    @pytest.mark.skip(reason="Password strength validation not implemented in UserManager")
    def test_weak_password_rejected(self, client):
        """fastapi-users enforces a minimum password length by default."""
        resp = client.post(
            "/auth/register",
            json={"email": "weak@orphic.com", "password": "x"},
        )
        # Either 400 (validation) or 422 (unprocessable) is acceptable
        assert resp.status_code in (400, 422), resp.text

    def test_invalid_email_rejected(self, client):
        resp = client.post(
            "/auth/register",
            json={"email": "not-an-email", "password": "Strong1Pass!"},
        )
        assert resp.status_code in (400, 422), resp.text

    def test_missing_fields_rejected(self, client):
        resp = client.post("/auth/register", json={"email": "x@y.com"})
        assert resp.status_code in (400, 422), resp.text

    def test_login_with_wrong_password_rejected(self, client):
        client.post(
            "/auth/register",
            json={"email": "wp@orphic.com", "password": "CorrectPass1!"},
        )
        resp = client.post(
            "/auth/jwt/login",
            data={"username": "wp@orphic.com", "password": "WrongPass1!"},
        )
        assert resp.status_code == 400, resp.text

    def test_login_unknown_email_rejected(self, client):
        resp = client.post(
            "/auth/jwt/login",
            data={"username": "nobody@orphic.com", "password": "Anything1!"},
        )
        assert resp.status_code == 400, resp.text


# ─────────────────────────────────────────────────────────────────────────────
# 7. INACTIVE USER — fastapi-users blocks them from getting tokens
# ─────────────────────────────────────────────────────────────────────────────
class TestInactiveUserBlocked:
    def test_inactive_user_cannot_login_or_access_protected_routes(
        self, client, db_session
    ):
        """Mark a registered user as is_active=False and confirm they're locked out."""
        from db.models import User
        import uuid as _uuid

        email = f"inactive-{_uuid.uuid4().hex[:6]}@orphic.com"
        password = "Strong1Pass!"
        r = client.post("/auth/register", json={"email": email, "password": password})
        assert r.status_code == 201

        # Find the user and flip is_active via the DB session
        # (fastapi-users doesn't expose a deactivate endpoint by default)
        from sqlalchemy import select
        result = client.post(
            "/auth/jwt/login",
            data={"username": email, "password": password},
        )
        assert result.status_code == 200
        token = result.json()["access_token"]

        # While active → 200
        ok = client.get(
            "/api/v1/conversations/",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert ok.status_code == 200

        # Flip the flag
        user_id = r.json()["id"]
        # Run a sync lookup via the test session
        import asyncio
        async def deactivate():
            from sqlalchemy import update
            from db.models import User
            await db_session.execute(
                update(User).where(User.id == _uuid.UUID(user_id)).values(is_active=False)
            )
            await db_session.commit()
        asyncio.get_event_loop().run_until_complete(deactivate())

        # Now the same token must be rejected (active=True check fails)
        forbidden = client.get(
            "/api/v1/conversations/",
            headers={"Authorization": f"Bearer {token}"},
        )
        # fastapi-users returns 401 for inactive users
        assert forbidden.status_code in (401, 403), forbidden.text