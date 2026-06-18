import os
import shutil
import asyncio
from typing import Optional
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage

from document_parser.graph import pipeline as doc_pipeline
from document_parser.proactive_analyzer import extract_preview, stream_opening_offer
from orchestrators.graph import get_bot, stream_response
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

UPLOAD_DIR = "./uploads"
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt", "md", "csv", "xlsx"}

os.makedirs(UPLOAD_DIR, exist_ok=True)


def _safe_save_path(filename: str) -> str:
    """Strips path traversal and returns a safe absolute save path."""
    safe_name = os.path.basename(filename)
    return os.path.join(UPLOAD_DIR, safe_name)


def _get_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext.lstrip(".").lower()


@router.post("/chat/stream")
async def chat_stream(
    file: Optional[UploadFile] = File(None),
    message: Optional[str] = Form(None),
    session_id: str = Form(...),
    user_id: str = Form(...),
):
    """
    Unified streaming endpoint for documents, images, and text queries.

    Scenarios:
        A. File only       → ingest/analyze, then stream proactive offer or image analysis
        B. File + message  → ingest/analyze, then stream agent response
        C. Message only    → stream agent response directly
        D. Nothing         → return guidance message
    """
    async def sse_generator():
        file_path = None
        pipeline_result = {}
        is_image = False

        # ── STEP 1: Handle file upload ──────────────────────────────────────
        if file and file.filename:
            ext = _get_extension(file.filename)

            # Validate extension before touching disk
            if ext not in IMAGE_EXTENSIONS and ext not in DOCUMENT_EXTENSIONS:
                yield f"data: [Error: Unsupported file type '.{ext}']\n\n"
                return

            file_path = _safe_save_path(file.filename)
            is_image = ext in IMAGE_EXTENSIONS

            # Enforce file size limit
            content = await file.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                yield f"data: [Error: File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB]\n\n"
                return

            try:
                with open(file_path, "wb") as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Failed to save file: {e}", exc_info=True)
                yield f"data: [Error: Failed to save file: {str(e)}]\n\n"
                return

            # ── Run ingestion / vision pipeline ──
            pipeline_result = await doc_pipeline.ainvoke({
                "file_path": file_path,
                "file_name": file.filename,
                "user_query": message or "",
                "user_id": user_id,
                "session_id": session_id,
            })

            if pipeline_result.get("error"):
                logger.error(f"Pipeline error: {pipeline_result['error']}")
                yield f"data: [Error: {pipeline_result['error']}]\n\n"
                return

            # ── Commit upload event to agent checkpointer ──
            try:
                bot = await get_bot()
                config = {"configurable": {"thread_id": session_id, "user_id": user_id}}

                if is_image:
                    image_cached = pipeline_result.get("image_cached", False)
                    if image_cached:
                        system_event = SystemMessage(
                            content=f"[Event: Image Uploaded] Name: {file.filename} | Already analyzed, use previous description."
                        )
                    else:
                        image_description = pipeline_result.get("response", "No description available.")
                        system_event = SystemMessage(
                            content=f"[Event: Image Uploaded] Name: {file.filename} | Description: {image_description}"
                        )
                else:
                    system_event = SystemMessage(
                        content=f"[Event: File Uploaded] Name: {file.filename} | Path: {file_path}"
                    )

                await bot.aupdate_state(config, {"messages": [system_event]}, as_node="__start__")
            except Exception as e:
                # Non-fatal: agent state update failing shouldn't abort the stream
                logger.warning(f"Failed to update agent state: {e}", exc_info=True)

        # ── STEP 2: Route to correct response strategy ────────────────────
        if file and file.filename and not message:
            if is_image:
                image_cached = pipeline_result.get("image_cached", False)
                if image_cached:
                    yield "data: Image already analyzed. Ask me anything about it.\n\n"
                else:
                    analysis_text = pipeline_result.get("response", "")
                    words = analysis_text.split()
                    for i in range(0, len(words), 5):
                        chunk = " ".join(words[i:i + 5]) + " "
                        yield f"data: {chunk}\n\n"
                        await asyncio.sleep(0.05)
            else:
                preview_text = extract_preview(file_path, max_tokens=1000)
                async for token in stream_opening_offer(preview_text, file.filename):
                    yield f"data: {token}\n\n"

        elif message:
            # ── CASE B + C: Message present (with or without file) ──
            async for token in stream_response(message, session_id, user_id):
                yield f"data: {token}\n\n"
            logger.info("Agent has responded.")

        else:
            # ── CASE D: Nothing provided ──
            yield "data: Please provide a message or upload a file.\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")



