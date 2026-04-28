import os
from psycopg_pool import AsyncConnectionPool
from langgraph.store.postgres import AsyncPostgresStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from memory.graph_checkpointer import pool
from uuid import uuid4
from datetime import datetime
from utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

try:
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
except Exception as e:
    logger.error(f"Failed to initialize embedding model: {e}")
    raise

# Module-level store — None until setup_memory_store() is called
store: AsyncPostgresStore | None = None


async def setup_memory_store():
    """Initialize the memory store with connection pooling."""
    global store

    try:
        store = AsyncPostgresStore(
            pool,
            index={"embed": embedding_model, "dims": 1536}
        )
        await store.setup()
        logger.info("Memory store initialized successfully")
        return store
    except Exception as e:
        logger.error(f"Failed to setup memory store: {e}")
        raise RuntimeError(f"Memory store setup failed: {e}") from e


async def store_memory(user_id: str, content: str):
    """Stores a semantic memory fact for a user in the vector DB."""
    if store is None:
        logger.error("Memory store not initialized")
        raise RuntimeError("Memory store not initialized. Call setup_memory_store() first")

    try:
        await store.aput(
            ("memories", user_id),
            str(uuid4()),
            {"content": content, "timestamp": datetime.utcnow().isoformat()}
        )
        logger.debug(f"Stored memory for user {user_id}: {content[:50]}...")
    except Exception as e:
        logger.error(f"Failed to store memory for user {user_id}: {e}")
        # Don't raise - memory storage shouldn't break the chat


async def retrieve_memory(user_id: str, query: str, top_k: int = 5):
    """Retrieves the top-K most semantically relevant memories for a user."""
    if store is None:
        logger.error("Memory store not initialized")
        return []

    try:
        results = await store.asearch(
            ("memories", user_id),
            query=query,
            limit=top_k
        )
        memories = [r.value["content"] for r in results]
        # logger.debug(f"Retrieved {len(memories)} memories for user {user_id}")
        return memories
    except Exception as e:
        logger.error(f"Failed to retrieve memories for user {user_id}: {e}")
        return []
