from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from typing import Optional

class Settings(BaseSettings):
    """
    Validated app configuration.
    Pydantic BaseSettings automatically reads environment variables.
    Variables defined without a default value are REQUIRED and will cause
    the application to crash at startup if missing.
    """
    # Required Variables
    openai_api_key: str
    db_url: str
    groq_api_key: str
    redis_url: str = "redis://localhost:6379"
    mistral_api_key:str

    azure_storage_connection_string: Optional[str] = None
    azure_storage_access_key: Optional[str] = None
    applicationinsights_connection_string:Optional[str] = None

    # Optional Variables (with defaults)
    frontend_url: str = "http://localhost:3000"
    oauth_redirect_base_url: str = "http://localhost:8000"
    n8n_base_url: str = "http://localhost:5678"

    github_token: Optional[str] = None
    notion_token: Optional[str] = None
    slack_bot_token: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    exa_api_key: Optional[str] = None
    n8n_token: Optional[str] = None
    n8n_mcp_token: Optional[str] = None
    browserless_api_key: Optional[str] = None

    #----------------------------OAuth Credentials----------------------
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None

    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None

    notion_client_id: Optional[str] = None
    notion_client_secret: Optional[str] = None

    slack_client_id: Optional[str] = None
    slack_client_secret: Optional[str] = None

    atlassian_client_id: Optional[str] = None
    atlassian_client_secret: Optional[str] = None

    #----------------------------Observability---------------
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = "orphic-project"
    langsmith_tracing: Optional[bool] = False

    composio_api_key: Optional[str] = None
    scalekit_env_url: Optional[str] = None
    scalekit_client_id: Optional[str] = None
    scalekit_client_secret: Optional[str] = None

    #auth settings
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 1440

    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    """Cached singleton — avoids re-reading .env from disk on every call."""
    return Settings()
