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

# import ssl
# from psycopg_pool import AsyncConnectionPool

# # 1. Define your Aiven DB URI (remove '?sslmode=require' from the text string)
# DB_URI = settings.db_url

# # 2. Build an explicit Python SSL Context for psycopg3
# ssl_context = ssl.create_default_context()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_REQUIRED

# pool = AsyncConnectionPool(
#     conninfo=DB_URI,
#     min_size=2,          # Aiven is ALWAYS-ON. Pre-opening a few connections eliminates "cold start" query lag.
#     max_size=15,         # Safe buffer: Aiven free tier hard caps at 20 max connections. Leave 5 slots open for dev work/PG Studio.
#     open=False,          # Keep False as required by psycopg3 async pool life cycles
#     kwargs={
#         "autocommit": True,
#         "sslcontext": ssl_context,  # Injects the mandatory secure SSL connection layer explicitly
#         "prepare_threshold": 5      # Re-enable prepared statements (value > 0) to optimize queries on your dedicated Aiven CPU!
#     }
# )


async def setup_db():
    """Call ONCE at app startup — opens pool and creates DB schema."""
    
    #check if pool is already open to prevent duplicate calls
    if getattr(pool, "_opened", False):
        logger.info("Database pool already open, skipping setup")
        return AsyncPostgresSaver(pool)

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