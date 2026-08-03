from pydantic import BaseModel
from typing import List, Optional
from config import get_settings
settings = get_settings()

class ProviderConfig(BaseModel):
    name: str
    label: str
    auth_type: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    scopes: Optional[List[str]] = None
    authorize_url: Optional[str] = None
    token_url: Optional[str] = None
    refresh_url: Optional[str] = None
    revoke_url: Optional[str] = None
    test_url: str
    mcp_url: str
    mcp_transport: str
    
    token_header: str
    token_prefix: str

    # Multi-provider compatibility fields
    token_request_format: str = "form"        # "form" or "json" (Notion uses json)
    scope_separator: str = " "                # " " for most, "," for some
    extra_token_params: dict = {}             # e.g. {"grant_type": "authorization_code"} for Google
    extra_authorize_params: dict = {}         # e.g. {"access_type": "offline"} for Google


PROVIDERS = { 

    
    "github": ProviderConfig( 
        name="github",
        label="Github",
        auth_type="oauth2",
        client_id=settings.github_client_id,
        client_secret=settings.github_client_secret,
        authorize_url ="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        refresh_url=None,
        test_url="https://api.github.com/user",
        scopes=["repo", "read:user"],
        mcp_url="https://api.githubcopilot.com/mcp/",
        mcp_transport="http",
        token_header="Authorization", 
        token_prefix="Bearer",
    ),

    "google": ProviderConfig(
        name="google",
        label="Google",
        auth_type="oauth2",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        refresh_url="https://oauth2.googleapis.com/token",
        test_url="https://www.googleapis.com/oauth2/v1/userinfo",
        scopes=[
            "openid",
            "email", 
            "profile",

            # Gmail
            "https://www.googleapis.com/auth/gmail.modify",

            # Google Calendar
            "https://www.googleapis.com/auth/calendar",

            # Google Drive
            "https://www.googleapis.com/auth/drive",

            # Google Docs
            "https://www.googleapis.com/auth/documents",

            # Google Sheets
            "https://www.googleapis.com/auth/spreadsheets",

            # Google Tasks
            "https://www.googleapis.com/auth/tasks",

            # Google Meet (limited — mostly read via Calendar API)
            "https://www.googleapis.com/auth/calendar.events",
        ],
        mcp_url="...",
        mcp_transport="http",
        token_header="Authorization",
        token_prefix="Bearer",
        extra_token_params={"grant_type": "authorization_code"},
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
    ),

    "notion": ProviderConfig(
        name="notion",
        label="Notion",
        auth_type="oauth2",
        client_id=settings.notion_client_id,
        client_secret=settings.notion_client_secret,
        authorize_url="https://api.notion.com/v1/oauth/authorize",
        token_url="https://api.notion.com/v1/oauth/token",
        refresh_url=None, 
        test_url="https://api.notion.com/v1/users/me",
        scopes=[], 
        mcp_url="https://mcp.notion.com/sse",
        mcp_transport="sse",
        token_header="Authorization",
        token_prefix="Bearer",
        token_request_format="json", 
        extra_authorize_params={"owner": "user"},
    ),

    "slack": ProviderConfig(
        name="slack",
        label="Slack",
        auth_type="oauth2",
        client_id=settings.slack_client_id,
        client_secret=settings.slack_client_secret,
        authorize_url="https://slack.com/oauth/v2/authorize",
        token_url="https://slack.com/api/oauth.v2.access",
        refresh_url=None,
        test_url="https://slack.com/api/auth.test",
        scopes=["channels:read", "chat:write", "users:read"],
        mcp_url="https://mcp.slack.com/sse",
        mcp_transport="sse",
        token_header="Authorization",
        token_prefix="Bearer",
        token_request_format="form",
    ),
}