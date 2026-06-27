"""
_test_env.py
============
Sets environment variables for the test process BEFORE any app module
is imported. Imported at the top of conftest.py.

The trick: pydantic-settings' BaseSettings reads .env at instantiation time,
NOT at import time. So if we mutate os.environ BEFORE any `from app import ...`
or `from config import get_settings` runs, our values win.

For tests, we want:
  - All required secrets to exist (so Settings() doesn't ValidationError)
  - DB_URL to point to an isolated SQLite file (no Postgres needed)
  - JWT_SECRET to be deterministic
"""

import os

# ── Secrets (test-only, never used in production) ────────────────────────────
os.environ["OPENAI_API_KEY"] = "test-openai-key-not-real"
os.environ["GROQ_API_KEY"] = "test-groq-key-not-real"
os.environ["JWT_SECRET"] = "test-jwt-secret-for-pytest-only-do-not-use-in-prod"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["JWT_EXPIRY_MINUTES"] = "60"

# ── Point at SQLite so we don't need Postgres for the test suite ─────────────
# (the engine rewrite in db/models.py converts postgresql+asyncpg → fine,
#  but aiosqlite is simpler and works without a server)
os.environ["DB_URL"] = "sqlite+aiosqlite:///./.test_orphic.db"

# ── Disable LangSmith tracing ────────────────────────────────────────────────
os.environ["LANGSMITH_TRACING"] = "false"

# ── Optional: allow real Postgres override via env var ───────────────────────
# If you want to run integration tests against real NeonDB, set:
#   export TEST_DATABASE_URL="postgresql+asyncpg://user:pass@host/db"
# and conftest will use it instead of SQLite.