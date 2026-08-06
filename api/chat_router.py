import os
import json
import shutil
import asyncio
from typing import Optional, Any, Literal
from fastapi import FastAPI, APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse
from langchain_core.messages import SystemMessage
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from memory.memory_extractor import extract_memory

from document_parser.graph import pipeline as doc_pipeline
from document_parser.proactive_analyzer import extract_preview, stream_opening_offer
from api.scalekit_auth import current_scalekit_active_user

from services.rate_limiter import check_chat_rate, log_token_usage
from orchestrators.graph import stream_response, resume_response, update_agent_state, get_thread_state
from api.auth import current_active_user
from db.models import User, Conversation, ChatMessage, get_async_session
from schemas.stream_events import StreamEvent
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

#sse helper

def sse(event_name: str, session_id: str, data: dict | None = None) -> str:
    payload = StreamEvent(
        type=event_name,
        session_id=session_id,
        data=data or {}
    )
    return f"event: {event_name}\ndata: {payload.model_dump_json()}\n\n"

@router.post("/chat/stream")
async def chat_stream(
    file: Optional[UploadFile] = File(None),
    message: Optional[str] = Form(None),
    session_id: str = Form(...),
    model: str = Form("gpt-5-nano"),
    user: User = Depends(current_scalekit_active_user),
    db_session: AsyncSession = Depends(get_async_session),
):
    """
    Unified streaming endpoint for documents, images, and text queries.

    Scenarios:
        A. File only       → ingest/analyze, then stream proactive offer or image analysis
        B. File + message  → ingest/analyze, then stream agent response
        C. Message only    → stream agent response directly
        D. Nothing         → return guidance message
    """
    rate_check = await check_chat_rate(str(user.id), getattr(user, "tier", "free"))
    if not rate_check.allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "message": f"You have exhausted your current limit: {rate_check.denied_reason}",
                "denied_reason": rate_check.denied_reason,
                "retry_after": rate_check.retry_after
            }
        )

    try:
        stmt = select(Conversation).where(
            Conversation.id == session_id,
            Conversation.user_id == user.id
        )
        result = await db_session.execute(stmt)
        existing_conv = result.scalar_one_or_none()

        if not existing_conv:
            if message:
                title = message[:30] + ("..." if len(message) > 30 else "")
            elif file and file.filename:
                title = f"Doc: {file.filename}"
            else:
                title = "New Chat"

            new_conv = Conversation(
                id=session_id,
                user_id=user.id,
                title=title
            )
            db_session.add(new_conv)
            await db_session.commit()
            logger.info(f"Auto-created sidebar thread {session_id} for user {user.id} with title: '{title}'")
    except Exception as e:
        logger.error(f"Failed to auto-create conversation: {e}", exc_info=True)

    async def sse_generator():
        file_path = None
        pipeline_result = {}
        is_image = False
        user_id = str(user.id)

        # ── STEP 1: Handle file upload ──────────────────────────────────────
        if file and file.filename:
            ext = _get_extension(file.filename)

            # Validate extension before touching disk
            if ext not in IMAGE_EXTENSIONS and ext not in DOCUMENT_EXTENSIONS:
                yield sse("error", session_id, {"message": f"Unsupported file type '.{ext}'"})
                return

            file_path = _safe_save_path(file.filename)
            is_image = ext in IMAGE_EXTENSIONS

            # Enforce file size limit
            content = await file.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                yield sse("error", session_id, {"message": f"File exceeds maximum size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB"})
                return

            try:
                with open(file_path, "wb") as f:
                    f.write(content)
            except Exception as e:
                logger.error(f"Failed to save file: {e}", exc_info=True)
                yield sse("error", session_id, {"message": f"Failed to save file: {str(e)}"})
                return

            # Upload file to Azure Blob Storage
            blob_url = None
            try:
                from services.azure_blob import upload_file_to_azure_blob
                blob_url = await upload_file_to_azure_blob(file.filename, content, file.content_type)
                logger.info(f"File saved to azure blob: {file.filename}")
            except Exception as blob_err:
                logger.warning(f"Azure Blob upload attempt failed: {blob_err}")

            # ── Run ingestion / vision pipeline ──
            pipeline_result = await doc_pipeline.ainvoke({
                "file_path": file_path,
                "file_name": file.filename,
                "user_query": message or "",
                "user_id": user_id,
                "session_id": session_id,
                "blob_url": blob_url,
            })

            if pipeline_result.get("error"):
                logger.error(f"Pipeline error: {pipeline_result['error']}")
                yield sse("error", session_id, {"message": pipeline_result['error']})
                return

            # ── Commit upload event to agent checkpointer ──
            try:
                url_str = f" | URL: {blob_url}" if blob_url else ""
                if is_image:
                    image_cached = pipeline_result.get("image_cached", False)
                    if image_cached:
                        system_event = SystemMessage(
                            content=f"[Event: Image Uploaded] Name: {file.filename}{url_str} | Already analyzed, use previous description."
                        )
                    else:
                        image_description = pipeline_result.get("response", "No description available.")
                        system_event = SystemMessage(
                            content=f"[Event: Image Uploaded] Name: {file.filename}{url_str} | Description: {image_description}"
                        )
                else:
                    system_event = SystemMessage(
                        content=f"[Event: File Uploaded] Name: {file.filename}{url_str} | Path: {file_path}"
                    )

                await update_agent_state(session_id, user_id, [system_event])

                # ── Also persist to chat_messages for UI history ──
                # Use a lightweight version without Description to avoid
                # duplication — the assistant response row stores the full text.
                try:
                    if is_image:
                        ui_event_content = f"[Event: Image Uploaded] Name: {file.filename}{url_str}"
                    else:
                        ui_event_content = system_event.content
                    db_session.add(ChatMessage(
                        conversation_id=session_id, role="system",
                        content=ui_event_content,
                    ))
                    await db_session.commit()
                except Exception as db_err:
                    logger.warning(f"Failed to persist system event to chat_messages: {db_err}")

            except Exception as e:
                # Non-fatal: agent state update failing shouldn't abort the stream
                logger.warning(f"Failed to update agent state: {e}", exc_info=True)

        # ── STEP 2: Route to correct response strategy ────────────────────
        if file and file.filename and not message:
            response_text = ""
            if is_image:
                image_cached = pipeline_result.get("image_cached", False)
                if image_cached:
                    response_text = "Image already analyzed. Ask me anything about it."
                    yield sse("token", session_id, {"text": response_text})
                else:
                    analysis_text = pipeline_result.get("response", "")
                    response_text = analysis_text
                    words = analysis_text.split()
                    for i in range(0, len(words), 5):
                        chunk = " ".join(words[i:i + 5]) + " "
                        yield sse("token", session_id, {"text" : chunk})
                        await asyncio.sleep(0.05)
            else:
                preview_text = extract_preview(file_path, max_tokens=1000)
                chunks = []
                async for token in stream_opening_offer(preview_text, file.filename):
                    chunks.append(token)
                    yield sse("token", session_id, {"text" : token})
                response_text = "".join(chunks)

            # Persist the AI response so it survives page reloads
            if response_text:
                try:
                    from langchain_core.messages import AIMessage
                    await update_agent_state(session_id, str(user.id), [AIMessage(content=response_text)])
                except Exception as e:
                    logger.warning(f"Failed to persist AI response for file upload: {e}", exc_info=True)

                # ── Also persist to chat_messages for UI history ──
                try:
                    db_session.add(ChatMessage(
                        conversation_id=session_id, role="assistant",
                        content=response_text,
                    ))
                    await db_session.commit()
                except Exception as db_err:
                    logger.warning(f"Failed to persist file assistant response to chat_messages: {db_err}")


        elif message:
            # ── CASE B + C: Message present (with or without file) ──

            # ── Persist user message to chat_messages BEFORE agent runs (crash-safe) ──
            try:
                db_session.add(ChatMessage(
                    conversation_id=session_id, role="user",
                    content=message,
                ))
                await db_session.commit()
            except Exception as db_err:
                logger.warning(f"Failed to persist user message to chat_messages: {db_err}")

            accumulated_text = []
            accumulated_reasoning = []

            async for event_type, data in stream_response(message, session_id, user_id, model):
                if event_type == "token":
                    yield sse("token", session_id, {"text": data})
                    # Accumulate text + reasoning for post-stream persistence
                    # chunk.content can be: str, dict, or list of dicts
                    if isinstance(data, str):
                        accumulated_text.append(data)
                    elif isinstance(data, list):
                        for part in data:
                            if isinstance(part, dict):
                                if part.get("type") == "text" and isinstance(part.get("text"), str):
                                    accumulated_text.append(part["text"])
                                elif part.get("type") == "reasoning":
                                    summary = part.get("summary")
                                    if isinstance(summary, list):
                                        for s in summary:
                                            if isinstance(s, dict) and isinstance(s.get("text"), str):
                                                accumulated_reasoning.append(s["text"])
                                    elif isinstance(part.get("text"), str):
                                        accumulated_reasoning.append(part["text"])
                    elif isinstance(data, dict) and isinstance(data.get("text"), str):
                        accumulated_text.append(data["text"])
                elif event_type == "activity":
                    yield sse("activity", session_id, data)
                elif event_type == "interrupt":
                    yield sse("interrupt", session_id, data)
                elif event_type == "error":
                    yield sse("error", session_id, {"message": data})
                elif event_type == "done":
                    token_used = data.get("total_tokens", 0) if isinstance(data, dict) else 0
                    yield sse("done", session_id)

                    if token_used > 0:
                        await log_token_usage(user_id, tokens_used=token_used)

            logger.info("Agent stream completed.")

            # ── Persist assistant response to chat_messages AFTER stream ends ──
            full_response = "".join(accumulated_text)
            full_reasoning = "".join(accumulated_reasoning)


            if full_response or full_reasoning:
                try:
                    db_session.add(ChatMessage(
                        conversation_id=session_id, role="assistant",
                        content=full_response,
                        reasoning=full_reasoning or None,
                    ))
                    await db_session.commit()
                except Exception as db_err:
                    logger.warning(f"Failed to persist assistant response to chat_messages: {db_err}")

            #strart memory
            asyncio.create_task(extract_memory(message, user_id))

        else:
            # ── CASE D: Nothing provided ──
            yield sse("error", session_id, {"message": "Please provide a message or upload a file."})

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })


#-------------------------Chat Resume Endpoint----------------------------
class ResumeRequest(BaseModel):
    interrupt_id: str
    decision: Literal[
        "connected",
        "approve",
        "reject",
        "cancel",
        "skip",
        "submit",       
    ]
    input:  dict[str, Any] = Field(default_factory=dict)


@router.post("/chat/resume/{session_id}")
async def resume_chat(
    session_id: str,
    body: ResumeRequest,
    user: User = Depends(current_scalekit_active_user),
    db_session: AsyncSession = Depends(get_async_session),
):
    """
    Resume a paused graph after an HITL interrupt.

    The frontend calls this after the user:
      - Connects a provider (action="continue")
      - Confirms a destructive action (action="continue")
      - Cancels the operation (action="cancel")

    session_id is the LangGraph thread_id from the original /chat/stream request.
    The checkpointer restores the exact graph state where execution paused.
    """
    user_id = str(user.id)

    # Verify the user owns this conversation
    stmt = select(Conversation).where(
        Conversation.id == session_id, Conversation.user_id == user.id,
    )
    result = await db_session.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=403,
            detail="Conversation not found or unauthorized.",
        )
    
    state = await get_thread_state(session_id, user_id)
    if not state or not state.tasks or not getattr(state.tasks[0], "interrupts", None):
        raise HTTPException(status_code=400, detail="No pending interrupt found.")

    expected_interrupt = state.tasks[0].interrupts[0].value
    if expected_interrupt.get("id") != body.interrupt_id:
        raise HTTPException(status_code=400, detail="Invalid interrupt ID.Please refresh the page")

    resume_value = {
        "interrupt_id": body.interrupt_id,
        "decision": body.decision,
        "input": body.input
    }

    async def sse_generator():
        accumulated_text = []
        accumulated_reasoning = []
        try:
            async for event_type, data in resume_response(session_id, user_id, resume_value):
                if event_type == "token":
                    yield sse("token", session_id, {"text": data})
                    if isinstance(data, dict):
                        parts = data.get("content", [])
                        if isinstance(parts, list):
                            for part in parts:
                                if isinstance(part, dict):
                                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                                        accumulated_text.append(part["text"])
                                    elif part.get("type") == "reasoning":
                                        summary = part.get("summary")
                                        if isinstance(summary, list):
                                            for s in summary:
                                                if isinstance(s, dict) and isinstance(s.get("text"), str):
                                                    accumulated_reasoning.append(s["text"])
                                        elif isinstance(part.get("text"), str):
                                            accumulated_reasoning.append(part["text"])
                        elif isinstance(data.get("text"), str):
                            accumulated_text.append(data["text"])
                    elif isinstance(data, str):
                        accumulated_text.append(data)
                elif event_type == "activity":
                    yield sse("activity", session_id, data)
                elif event_type == "interrupt":
                    yield sse("interrupt", session_id, data)
                elif event_type == "error":
                    yield sse("error", session_id, {"message": data})
                elif event_type == "done":
                    yield sse("done", session_id)

            # Persist post-resume assistant message to chat_messages DB
            full_response = "".join(accumulated_text)
            full_reasoning = "".join(accumulated_reasoning)
            if full_response or full_reasoning:
                try:
                    db_session.add(ChatMessage(
                        conversation_id=session_id, role="assistant",
                        content=full_response,
                        reasoning=full_reasoning or None,
                    ))
                    await db_session.commit()
                except Exception as db_err:
                    logger.warning(f"Failed to persist resumed assistant response to chat_messages: {db_err}")

        except Exception as e:
            logger.error(
                f"Resume SSE error session={session_id}: {type(e).__name__}: {e}",
                exc_info=True,
            )
            yield sse("error", session_id, {"message": "Resume failed. Please try again."})

    return StreamingResponse(
        sse_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        })


from services.rate_limiter import get_chat_usage

@router.get("/user/usage")
async def get_user_rate_usage(user: User = Depends(current_scalekit_active_user)):
    """
    Returns current usage stats for the logged-in user.
    Frontend can render progress bars: "Used 12 / 50 turns today".
    """
    usage = await get_chat_usage(str(user.id), getattr(user, "tier", "free"))
    return usage

#user usage router response format
"""
{
  "requests": {
    "this_hour": { "current": 3, "limit": 10 },
    "today":     { "current": 14, "limit": 50 }
  },
  "tokens": {
    "this_hour": { "current": 12450, "limit": 100000 },
    "today":     { "current": 45100, "limit": 100000 }
  }
}
"""