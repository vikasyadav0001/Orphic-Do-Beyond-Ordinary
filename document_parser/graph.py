import os
import asyncio
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from modalities.vision.vision_model import analyse_image
from .doc_parser_rag import get_or_create_vector_store
from utils.logger import get_logger

logger = get_logger(__name__)


class PipelineState(TypedDict):
    """Graph state for image/doc ingestion pipeline."""
    file_path: Optional[str]
    file_name: Optional[str]
    file_type: Optional[str]
    user_id: str
    session_id: str
    user_query: str
    has_file: bool
    has_query: bool
    image_cached: bool
    response: str
    error: str
    vector_store_ready: bool


IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DOCUMENT_EXTENSIONS = {"pdf", "csv", "xlsx", "xls", "txt", "md", "py", "json", "js", "html", "css"}

async def entry_node(state: PipelineState) -> dict:
    logger.info("=== Entry Node ===")

    file_path = state.get("file_path", "").strip()
    user_query = state.get("user_query", "").strip()
    user_id = state.get("user_id", "").strip()
    session_id = state.get("session_id", "").strip()

    # ── VALIDATE REQUIRED FIELDS ──
    if not user_id or not session_id:
        logger.error("Missing user_id or session_id.")
        return {"error": "Missing user_id or session_id.", "has_file": False, "has_query": False}

    # ── VALIDATE FILE ──
    if not file_path:
        logger.warning("No file_path provided. Nothing to process.")
        return {"has_file": False, "has_query": bool(user_query), "error": "No file provided."}

    if not os.path.exists(file_path):
        logger.warning(f"File not found on disk: {file_path!r}")
        return {"has_file": False, "has_query": bool(user_query), "error": f"File not found: {file_path!r}"}

    if not os.path.isfile(file_path):
        logger.warning(f"Path exists but is not a file: {file_path!r}")
        return {"has_file": False, "has_query": bool(user_query), "error": f"Path is not a file: {file_path!r}"}

    if os.path.getsize(file_path) == 0:
        logger.warning(f"File is empty: {file_path!r}")
        return {"has_file": False, "has_query": bool(user_query), "error": f"File is empty: {file_path!r}"}

    # ── DETECT FILE TYPE ──
    file_name = os.path.basename(file_path)
    _, ext = os.path.splitext(file_name)
    file_ext = ext.lstrip(".").lower()

    if not file_ext:
        logger.warning(f"Could not determine file extension for: {file_name!r}")
        return {"has_file": False, "has_query": bool(user_query), "error": f"Unknown file type: {file_name!r}"}

    if file_ext not in IMAGE_EXTENSIONS and file_ext not in DOCUMENT_EXTENSIONS:
        logger.warning(f"Unsupported file type: {file_ext!r}")
        return {"has_file": False, "has_query": bool(user_query), "error": f"Unsupported file type: '.{file_ext}'"}

    logger.info(f"Entry complete. file={file_name!r}, type={file_ext!r}, has_query={bool(user_query)}")
    return {
        "has_file": True,
        "has_query": bool(user_query),
        "file_name": file_name,
        "file_type": file_ext,
        "error": "",
    }


def route_after_entry(state: PipelineState) -> str:
    if not state.get("has_file"):
        logger.info("Route: END (no valid file)")
        return END

    file_type = state.get("file_type", "")

    if file_type in IMAGE_EXTENSIONS:
        logger.info("Route: IMAGE path")
        return "image_vision"

    logger.info("Route: INGEST path")
    return "doc_analysis_node"


async def image_vision(state: PipelineState) -> dict:
    logger.info("=== Image Vision Node ===")
    file_path = state.get("file_path")

    try:
        response, is_cached = await analyse_image(file_path)
        return {"response": response, "image_cached": is_cached}
    except Exception as e:
        logger.error(f"Image analysis failed: {e}", exc_info=True)
        return {"error": f"Image analysis failed: {str(e)}"}


async def doc_analysis_node(state: PipelineState) -> dict:
    logger.info("=== Ingestion Node ===")
    file_path = state.get("file_path")
    session_id = state.get("session_id")
    user_id = state.get("user_id")

    try:
        logger.info("Running full ingestion.")
        get_or_create_vector_store(
            file_path=file_path,
            session_id=session_id,
            user_id=user_id,
        )
        logger.info("Ingestion complete.")
        return {"vector_store_ready": True}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        return {"error": f"Ingestion failed: {str(e)}", "vector_store_ready": False}


def build_pipeline_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("entry", entry_node)
    graph.add_node("image_vision", image_vision)
    graph.add_node("doc_analysis_node", doc_analysis_node)

    graph.add_edge(START, "entry")

    graph.add_conditional_edges(
        "entry",
        route_after_entry,
        {
            "image_vision": "image_vision",
            "doc_analysis_node": "doc_analysis_node",
            END: END,
        }
    )

    graph.add_edge("image_vision", END)
    graph.add_edge("doc_analysis_node", END)

    return graph.compile()


pipeline = build_pipeline_graph()


if __name__ == "__main__":
    input_state = {
        "file_path": "/home/dell/Desktop/Orphic-Do-Beyond-Ordinary/uploads/unnamed.jpg",
        "file_name": "unnamed.jpg",
        "user_query": "",
        "user_id": "default_user",
        "session_id": "test-thread-3",
    }

    result = asyncio.run(pipeline.ainvoke(input_state))
    print(f"Analysis Response: {result.get('response')}")
    print(f"Vector Store Ready: {result.get('vector_store_ready')}")
    print(f"Error: {result.get('error')}")















