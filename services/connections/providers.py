# ============================================================
# services/connections/providers.py
# ============================================================

import base64
import hashlib
import os
import secrets
from typing import Optional, List
from pydantic import BaseModel
from config import get_settings

settings = get_settings()


class ProviderConfig(BaseModel):
    name: str
    label: str
    auth_type: str  # "oauth2" | "api_key"

    # OAuth credentials
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    # OAuth endpoints
    authorize_url: Optional[str] = None
    token_url: Optional[str] = None
    refresh_url: Optional[str] = None
    revoke_url: Optional[str] = None
    test_url: Optional[str] =  None

    # MCP
    mcp_url: str
    mcp_transport: str  # "http" | "sse"

    # How the token is sent in API calls
    token_header: str = "Authorization"
    token_prefix: str = "Bearer"

    # ── Token exchange config ──────────────────────────────
    # Where client credentials go during token exchange:
    #   "body"  → client_id + client_secret in POST body (default, most providers)
    #   "basic" → Authorization: Basic base64(id:secret) header (Notion)
    #   "none"  → no credentials sent (pure PKCE, no secret)
    token_endpoint_auth_method: str = "body"

    # POST body encoding:
    #   "form" → application/x-www-form-urlencoded (default)
    #   "json" → application/json (Notion)
    token_request_format: str = "form"

    # Extra static headers on EVERY token exchange / refresh call
    # e.g. {"Notion-Version": "2022-06-28"}
    extra_token_headers: dict = {}

    # Extra params merged into token exchange body
    # e.g. {"grant_type": "authorization_code"} for Google
    extra_token_params: dict = {}

    # Extra params merged into the authorize URL
    # e.g. {"access_type": "offline", "prompt": "consent"} for Google
    extra_authorize_params: dict = {}

    # ── Scope config ──────────────────────────────────────
    scopes: Optional[List[str]] = None
    scope_separator: str = " "

    # Some providers use a different param name for user-level scopes
    # e.g. Slack uses "user_scope" for user tokens (separate from bot "scope")
    user_scope_param: Optional[str] = None   # "user_scope" for Slack
    user_scopes: Optional[List[str]] = None  # scopes that go into user_scope

    # ── Token response extraction ─────────────────────────
    # Dot-notation path to access_token in the response JSON
    # Default: "access_token" (top-level)
    # Slack user token: "authed_user.access_token"
    access_token_path: str = "access_token"

    # Dot-notation path to refresh_token (top-level for most)
    refresh_token_path: str = "refresh_token"

    # ── PKCE config ───────────────────────────────────────
    # Whether to generate and send PKCE parameters
    # "none"     → no PKCE (default, confidential clients with client_secret)
    # "s256"     → PKCE with SHA-256 (S256), replaces client_secret at token exchange
    # "s256+secret" → PKCE *and* client_secret (some providers want both)
    pkce_mode: str = "none"

    # ── Refresh token behavior ────────────────────────────
    # "reuse"   → provider returns same refresh_token (most)
    # "rotate"  → provider issues new refresh_token on every refresh (Notion, Google)
    refresh_token_behavior: str = "reuse"

    # ── Local MCP server fallback ─────────────────────────
    # Used when mcp_url is empty or remote is unavailable
    # command: executable to run ("npx", "uvx", "node", "python")
    # local_args: args to the command
    # env_token_key: which env var name receives the user's access token
    # local_env: static env vars the server always needs
    local_command: Optional[str] = None
    local_args: List[str] = []
    env_token_key: Optional[str] = None
    local_env: dict = {}

# ──────────────────────────────────────────────────────────────
# Provider registry
# ──────────────────────────────────────────────────────────────

PROVIDERS: dict[str, ProviderConfig] = {

    "github": ProviderConfig(
        name="github",
        label="GitHub",
        auth_type="oauth2",
        client_id=settings.github_client_id,
        client_secret=settings.github_client_secret,
        aupermanentlythorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        refresh_url=None,
        test_url="https://api.github.com/user",
        scopes=["repo", "read:user"],
        mcp_url="https://api.githubcopilot.com/mcp",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        # GitHub: client_secret_post (body), form-encoded, no PKCE
        token_endpoint_auth_method="body",
        token_request_format="form",

        
    ),

    "google_gmail": ProviderConfig(
        name="google_gmail",
        label="Gmail",
        auth_type="oauth2",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        test_url="https://www.googleapis.com/oauth2/v1/userinfo",
        scopes=["openid", "email", "profile", "https://www.googleapis.com/auth/gmail.send", "https://www.googleapis.com/auth/gmail.metadata"],
        mcp_url="https://gmailmcp.googleapis.com/mcp/v1",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="body",
        token_request_format="form",
        extra_token_params={"grant_type": "authorization_code"},
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
        # Google rotates refresh tokens when prompt=consent is used
        refresh_token_behavior="rotate",
    ),

    "google_drive": ProviderConfig(
        name="google_drive",
        label="Drive",
        auth_type="oauth2",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        test_url="https://www.googleapis.com/oauth2/v1/userinfo",
        scopes=["openid", "email", "profile", "https://www.googleapis.com/auth/drive.file"],
        mcp_url="https://drivemcp.googleapis.com/mcp/v1",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="body",
        token_request_format="form",
        extra_token_params={"grant_type": "authorization_code"},
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
        # Google rotates refresh tokens when prompt=consent is used
        refresh_token_behavior="rotate",
    ),

    "google_sheets": ProviderConfig(
        name="google_sheets",
        label="Sheets",
        auth_type="oauth2",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        test_url="https://www.googleapis.com/oauth2/v1/userinfo",
        scopes=["openid", "email", "profile", "https://www.googleapis.com/auth/spreadsheets"],
        mcp_url="https://sheetsmcp.googleapis.com/mcp/v1",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="body",
        token_request_format="form",
        extra_token_params={"grant_type": "authorization_code"},
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
        # Google rotates refresh tokens when prompt=consent is used
        refresh_token_behavior="rotate",
    ),

    "google_docs": ProviderConfig(
        name="google_docs",
        label="Docs",
        auth_type="oauth2",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        test_url="https://www.googleapis.com/oauth2/v1/userinfo",
        scopes=["openid", "email", "profile", "https://www.googleapis.com/auth/documents"],
        mcp_url="https://docsmcp.googleapis.com/mcp/v1",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="body",
        token_request_format="form",
        extra_token_params={"grant_type": "authorization_code"},
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
        # Google rotates refresh tokens when prompt=consent is used
        refresh_token_behavior="rotate",
    ),

    "google_calendar": ProviderConfig(
        name="google_calendar",
        label="Calendar",
        auth_type="oauth2",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        test_url="https://www.googleapis.com/oauth2/v1/userinfo",

        scopes=["openid", "email", "profile", "https://www.googleapis.com/auth/calendar"],
        mcp_url="https://calendarmcp.googleapis.com/mcp/v1",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="body",
        token_request_format="form",
        extra_token_params={"grant_type": "authorization_code"},
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
        # Google rotates refresh tokens when prompt=consent is used
        refresh_token_behavior="rotate",
    ),

    "notion": ProviderConfig(
        name="notion",
        label="Notion",
        auth_type="oauth2",
        client_id=settings.notion_client_id,   # from dynamic registration
        client_secret=None,                     # public client, no secret
        authorize_url="https://mcp.notion.com/authorize",
        token_url="https://mcp.notion.com/token",
        refresh_url="https://mcp.notion.com/token",
        test_url=None,                          # REST API rejects MCP tokens
        scopes=[],
        mcp_url="https://mcp.notion.com/mcp",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="none",      # public client
        token_request_format="form",
        extra_token_headers={},
        extra_authorize_params={},              # remove owner=user
        refresh_token_behavior="rotate",
        pkce_mode="s256"
    ),


    "slack": ProviderConfig(
        name="slack",
        label="Slack",
        auth_type="oauth2",
        client_id=settings.slack_client_id,
        client_secret=settings.slack_client_secret,
        authorize_url="https://slack.com/oauth/v2/authorize",
        token_url="https://slack.com/api/oauth.v2.access",
        refresh_url=None,  # token rotation is opt-in; keeping simple
        test_url="https://slack.com/api/auth.test",
        # Bot scopes go in "scope", user scopes go in "user_scope"
        scopes=[],  # MUST BE EMPTY for localhost redirects
        user_scope_param="user_scope",
        user_scopes=[
            "channels:history", "channels:read",
            "groups:history", "groups:read",
            "im:history", "im:read", "im:write",
            "mpim:history", "mpim:read",
            "users:read", "chat:write", "search:read",
        ],
        mcp_url="https://mcp.slack.com/mcp",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="body",
        token_request_format="form",
        # Slack user token lives at authed_user.access_token, not top-level
        access_token_path="authed_user.access_token",
        pkce_mode="none"
    ),
    
    "atlassian": ProviderConfig(
        name="atlassian",
        label="Atlassian",
        auth_type="oauth2",
        client_id=settings.atlassian_client_id,  # from DCR, not dev console
        client_secret=None,                            # public client, no secret
        authorize_url="https://mcp.atlassian.com/v1/authorize",
        token_url="https://cf.mcp.atlassian.com/v1/token",
        refresh_url="https://cf.mcp.atlassian.com/v1/token",
        revoke_url="https://cf.mcp.atlassian.com/v1/token",
        test_url=None,                                 # MCP tokens don't work on REST APIs
        scopes=[],
        mcp_url="https://mcp.atlassian.com/v1/mcp",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        token_endpoint_auth_method="none",             # public client, PKCE only
        token_request_format="form",
        refresh_token_behavior="rotate",
        pkce_mode="s256",
    ),
}