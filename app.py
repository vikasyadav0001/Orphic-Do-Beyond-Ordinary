# main.py (at project root)
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from api.chat_router import router
from api.conversations import router as conversations_router

from api.auth import fastapi_users, auth_backend, UserRead, UserCreate, current_active_user
from db.models import engine, Base, User

from utils.logger import get_logger
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Database ready (migrations handled by Alembic).")

    logger.info("App is ready and running...")
    yield
    logger.info("Shutting down the server...")


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


app.include_router(router)
app.include_router(conversations_router)

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
    from sqlalchemy import text
    import shutil

    health = {
        "status": "healthy",
        "service": "orphic-backend",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {}
    }

    # Check database connectivity
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health["checks"]["database"] = "connected"
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["database"] = f"error: {str(e)}"

    # Check if LangGraph agent is initialized
    try:
        from orchestrators.graph import _bot
        health["checks"]["agent"] = "ready" if _bot is not None else "not_initialized"
    except Exception:
        health["checks"]["agent"] = "import_error"

    # Check disk space for uploads
    try:
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