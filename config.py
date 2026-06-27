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

    # Optional Variables (with defaults)
    github_token: Optional[str] = None
    notion_token: Optional[str] = None
    firecrawl_api_key: Optional[str] = None
    exa_api_key: Optional[str] = None

    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = "orphic-project"
    langsmith_tracing: Optional[bool] = False

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
