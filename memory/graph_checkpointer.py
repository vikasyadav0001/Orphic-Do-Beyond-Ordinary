import os
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
load_dotenv()


DB_URI = os.getenv("DATABASE_URL")

pool = AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    open=False,
    kwargs={
        "row_factory": dict_row, 
        "autocommit": True
    }
)

async def setup():
    """Call ONCE at app startup — opens pool and creates DB schema."""
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return checkpointer