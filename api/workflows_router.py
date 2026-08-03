from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import User, Workflows, get_async_session
from api.scalekit_auth import current_scalekit_active_user
from config import get_settings
settings = get_settings()

router = APIRouter(prefix="/workflows", tags=["workflows"])

@router.get("/")
async def list_workflows(user: User = Depends(current_scalekit_active_user), session: AsyncSession = Depends(get_async_session)):
    result = await session.execute(
        select(Workflows).where(Workflows.user_id == user.id).order_by(Workflows.created_at.desc())
    )

    return result.scalars().all()


@router.get("/{workflow_id}")
async def get_workflow(workflow_id: str, user: User = Depends(current_scalekit_active_user), session=Depends(get_async_session)):
    result = await session.execute(
        select(Workflows)
        .where(Workflows.n8n_workflow_id == workflow_id, Workflows.user_id == user.id)
    )
    workflow = result.scalar_one_or_none()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return workflow


@router.get("/{workflow_id}/runs")
async def get_workflow_runs(workflow_id:str, user: User = Depends(current_scalekit_active_user), session: AsyncSession = Depends(get_async_session)):

    workflow = (
        await session.execute(
        select(Workflows).where(Workflows.n8n_workflow_id == workflow_id, Workflows.user_id == user.id)

    )).scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            url=f"{settings.n8n_base_url}/api/v1/executions",
            params={"workflowId": workflow_id},
            headers={"X-N8N-API-KEY": settings.n8n_token}
        )

    if not response.is_success:
        raise HTTPException(status_code=502, detail="Failed to fetch runs from n8n")
    return response.json()



