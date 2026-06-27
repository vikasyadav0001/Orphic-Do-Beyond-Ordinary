"""
Endpoints for managing user conversations (the sidebar chats).
Maps LangGraph thread_ids to users and titles.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, update
from typing import List
from pydantic import BaseModel
import uuid

from db.models import Conversation, User, get_async_session
from api.auth import current_active_user
from utils.logger import get_logger
from orchestrators.graph import get_bot
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/conversations", tags=["conversations"])

# --- Schemas ---
class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str

    class Config:
        from_attributes = True

class ConversationCreate(BaseModel):
    title: str = "New Chat"

class ConversationUpdate(BaseModel):
    title: str


# --- Endpoints ---
@router.get("/", response_model=List[ConversationResponse])
async def get_user_conversations(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Fetch all conversations for the logged-in user to populate the sidebar."""
    logger.info(f"Fetching conversations for user {user.id}")
    
    # Query the database for conversations belonging to this user, newest first
    stmt = select(Conversation).where(Conversation.user_id == user.id).order_by(Conversation.created_at.desc())
    result = await session.execute(stmt)
    conversations = result.scalars().all()
    
    # We must convert datetime to string for the Pydantic response
    response = []
    for conv in conversations:
        response.append(
            ConversationResponse(
                id=conv.id,
                title=conv.title,
                created_at=str(conv.created_at)
            )
        )
    return response


@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    data: ConversationCreate,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Create a new conversation (New Chat button)."""
    logger.info(f"Creating new conversation for user {user.id}")
    
    new_conv = Conversation(
        user_id=user.id,
        title=data.title
    )
    
    session.add(new_conv)
    await session.commit()
    await session.refresh(new_conv)
    
    return ConversationResponse(
        id=new_conv.id,
        title=new_conv.title,
        created_at=str(new_conv.created_at)
    )

@router.get("/{thread_id}/messages")
async def get_conversation_messages(
    thread_id: str,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Fetch the chat history for a specific conversation from LangGraph."""
    logger.info(f"Fetching messages for thread {thread_id} (user {user.id})")
    
    # 1. Verify ownership
    stmt = select(Conversation).where(
        Conversation.id == thread_id,
        Conversation.user_id == user.id
    )
    result = await session.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=403, detail="Conversation not found or unauthorized")
        
    # 2. Get the bot and load state
    bot = await get_bot()
    config = {"configurable": {"thread_id": thread_id, "user_id": str(user.id)}}
    
    try:
        state = await bot.aget_state(config)
        messages = []
        if state and state.values and "messages" in state.values:
            for msg in state.values["messages"]:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    messages.append({"role": "assistant", "content": msg.content})
                elif isinstance(msg, SystemMessage):
                    messages.append({"role": "system", "content": msg.content})
        
        return {"messages": messages}
    except Exception as e:
        logger.error(f"Error fetching state for thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")

@router.patch("/{thread_id}", response_model=ConversationResponse)
async def rename_conversation(
    thread_id: str,
    data: ConversationUpdate,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Rename a conversation's title."""
    logger.info(f"Renaming conversation {thread_id} for user {user.id} to '{data.title}'")
    
    # Verify ownership
    stmt = select(Conversation).where(
        Conversation.id == thread_id,
        Conversation.user_id == user.id
    )
    result = await session.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")
        
    conv.title = data.title
    await session.commit()
    await session.refresh(conv)
    
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        created_at=str(conv.created_at)
    )

@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    thread_id: str,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session)
):
    """Delete a conversation and its corresponding checkpoints in LangGraph."""
    logger.info(f"Deleting conversation {thread_id} for user {user.id}")
    
    # 1. Verify ownership
    stmt = select(Conversation).where(
        Conversation.id == thread_id,
        Conversation.user_id == user.id
    )
    result = await session.execute(stmt)
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found or unauthorized")
        
    # 2. Delete the conversation row
    await session.delete(conv)
    
    # 3. Clean up LangGraph checkpoint tables (checkpoints, checkpoint_blobs, checkpoint_writes)
    try:
        from sqlalchemy import text
        await session.execute(text("DELETE FROM checkpoint_writes WHERE thread_id = :tid"), {"tid": thread_id})
        await session.execute(text("DELETE FROM checkpoint_blobs WHERE thread_id = :tid"), {"tid": thread_id})
        await session.execute(text("DELETE FROM checkpoints WHERE thread_id = :tid"), {"tid": thread_id})
    except Exception as e:
        logger.warning(f"Failed to clear LangGraph checkpoints for thread {thread_id}: {e}")

    await session.commit()
    return
