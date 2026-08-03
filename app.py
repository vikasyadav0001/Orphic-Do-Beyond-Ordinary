# main.py (at project root)
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from azure.monitor.opentelemetry import configure_azure_monitor
from api.chat_router import router
from api.conversations import router as conversations_router
from api.connections_router import router as connections_router
from api.workflows_router import router as workflows_router


from api.auth import fastapi_users, auth_backend, UserRead, UserCreate, current_active_user
from db.models import engine, Base, User

from config import get_settings
from utils.logger import get_logger
logger = get_logger(__name__)
settings = get_settings()

#configure azure monitoring
if settings.applicationinsights_connection_string:
    configure_azure_monitor(
        connection_string=settings.applicationinsights_connection_string,
        sampling_ratio=0.2,
    )
    logger.info("Azure Application Insights telemetry configured.")
else:
    logger.info("Azure Application Insights connection string not provided — telemetry disabled.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle — fail-fast startup, graceful shutdown.
    Startup order matters:
      1. Redis first — fastest, needed by everything else
      2. Postgres pool — checkpointer schema creation
      3. Memory store — depends on Postgres pool being open
    If any dependency fails, the app crashes at boot. Docker's
    restart policy retries automatically. Better than serving
    500s from a half-initialized app.
    """
    from services.cache import get_redis
    from memory.graph_checkpointer import setup_db
    from memory.long_term_memory import setup_memory_store
    from orchestrators.graph import initialize_graph

    # ── Startup ───────────────────────────────────────────────

    try:
        redis = await get_redis()
        await redis.ping()
        logger.info("Redis connected and healthy")
    except Exception as e:
        logger.critical(f"Redis connection failed: {e}")
        raise  # crash the app — don't serve without Redis

    try:
        await setup_db()
        logger.info("Postgres pool opened, checkpointer schema ready")
    except Exception as e:
        logger.critical(f"Postgres setup failed: {e}")
        raise

    try:
        await setup_memory_store()
        logger.info("Memory store initialized")
    except Exception as e:
        logger.critical(f"Memory store setup failed: {e}")
        raise

    try:
        await initialize_graph()
        logger.info("Main Agent graph initialized")
    except Exception as e:
        logger.critical(f"Failed to initialize graph: {e}")
        raise


    logger.info("All systems ready — accepting requests")
    
    logger.info("Database ready (migrations handled by Alembic).")
    logger.info("App is ready and running...")

    yield
    # ── Shutdown ──────────────────────────────────────────────
    try:
        redis = await get_redis()
        await redis.aclose()
        logger.info("Redis connection closed")
    except Exception as e:
        logger.warning(f"Redis shutdown error (non-fatal): {e}")

    logger.info("Shutting down the server...")
    logger.info("Application shutdown complete")



app = FastAPI(title="Project-Orphic", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex="https://.*\\.azurestaticapps\\.net|http://localhost:\\d+|http://127\\.0\\.0\\.1:\\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=['auth'],
)

app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=['auth'],
)

app.include_router(workflows_router)
app.include_router(router)
app.include_router(conversations_router)
app.include_router(connections_router)

@app.get("/")
def welcome():
    """Welcome Function.
    """
    return "Welcome to the project Orphic."


@app.get("/health")
async def health_check():
    """Deep health check — verifies app + all critical dependencies."""
    from datetime import datetime
    from fastapi.responses import JSONResponse
    from services.cache import get_redis
    import os

    health = {
        "status": "healthy",
        "service": "orphic-backend",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }

    try:
        async with engine.connect() as conn:
            from sqlalchemy import text
            await conn.execute(text("SELECT 1"))
        health["checks"]["database"] = "connected"
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["database"] = f"error: {str(e)}"

    try:
        import shutil
        disk = shutil.disk_usage("/app/uploads" if os.path.exists("/app/uploads") else "./uploads")
        free_gb = round(disk.free / (1024**3), 2)
        health["checks"]["disk_free_gb"] = free_gb
        if free_gb < 1:
            health["status"] = "degraded"
    except Exception:
        health["checks"]["disk_free_gb"] = "unknown"

    status_code = 200 if health["status"] != "unhealthy" else 503
    return JSONResponse(content=health, status_code=status_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", "127.0.0.1", 8000)
