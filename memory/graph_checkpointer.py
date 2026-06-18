from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from config import get_settings
from utils.logger import get_logger

settings = get_settings()
DB_URI = settings.db_url

logger = get_logger(__name__)

pool = AsyncConnectionPool(
    conninfo=DB_URI,
    min_size=0,          # NeonDB: no pre-opened connections (serverless auto-suspend)
    max_size=5,
    open=False,      # NeonDB free tier: limited concurrent connections    # NeonDB: allow time for cold-start wake-up
    kwargs={
        "autocommit": True,
        "prepare_threshold": 0
    }
)

async def setup_db():
    """Call ONCE at app startup — opens pool and creates DB schema."""
    try:
        await pool.open()
        logger.info("Database pool opened successfully")

        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        logger.info("Database schema setup complete")
        return checkpointer
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        raise RuntimeError(f"Database setup failed: {e}") from e