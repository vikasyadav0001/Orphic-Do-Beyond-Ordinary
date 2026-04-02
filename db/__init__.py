# db/
# Database infrastructure layer — raw database plumbing.
# Manages connections, ORM models, and schema migrations.
# Responsibilities:
#   - connection.py  → PostgreSQL async connection pool (psycopg + asyncpg)
#   - models.py      → SQLAlchemy ORM models (Users, Sessions, Threads, Messages)
#   - migrations/    → Alembic migration scripts for schema versioning
# This layer is NOT agent-aware — it only deals with raw DB operations.
# Agent-level memory (LangGraph checkpointer) lives in memory/ and uses this layer.
