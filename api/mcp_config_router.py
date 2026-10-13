"""
MCP Config Router
==================
CRUD endpoints for managing per-user MCP server configurations.
Users submit a mcpServers JSON dict; we validate, discover tools, and cache.
"""

import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from api.scalekit_auth import current_scalekit_active_user
from db.models import User, UserMCPConfig, get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from services.cache import (
    invalidate_mcp_config,
    set_cached_mcp_config,
    set_cached_mcp_tool_schemas,
    set_mcp_config_version,
)
from services.mcp_config_service import discover_tools, validate_config
from utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp-config", tags=["MCP Config"])


# ── Request/Response Models ───────────────────────────────────────────────

class MCPConfigRequest(BaseModel):
    """
    The mcpServers dict from the user.
    Example:
        {
            "exa": {"url": "https://mcp.exa.ai/mcp", "headers": {"x-api-key": "..."}},
            "my-server": {"url": "https://my-mcp.example.com/mcp", "headers": {}}
        }
    """
    servers: dict


class MCPTestRequest(BaseModel):
    """Single server entry for test-connect (no save)."""
    name: str
    url: str
    headers: dict = {}


# ── GET /mcp-config ───────────────────────────────────────────────────────

@router.get("/")
async def get_mcp_config(
    user: User = Depends(current_scalekit_active_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Returns the user's current MCP config and metadata."""
    stmt = select(UserMCPConfig).where(UserMCPConfig.user_id == user.id)
    result = await db.execute(stmt)
    config_row = result.scalar_one_or_none()

    if not config_row:
        return {"config": None, "updated_at": None}

    return {
        "config": config_row.config_json,
        "config_json": config_row.config_json,
        "updated_at": config_row.updated_at.isoformat() if config_row.updated_at else None,
    }


# ── PUT /mcp-config ──────────────────────────────────────────────────────

@router.put("/")
async def save_mcp_config(
    body: MCPConfigRequest,
    user: User = Depends(current_scalekit_active_user),
    db: AsyncSession = Depends(get_async_session),
):
    """
    Validate, discover tools, and save the user's MCP config.
    Upserts — creates if new, updates if existing.
    """
    user_id = str(user.id)

    # 1. Validate config (SSRF, stdio, limits)
    raw_dict = body.servers if isinstance(body.servers, dict) and body.servers else {}
    try:
        validated = validate_config(raw_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # If saving empty servers config, delete DB row & clear caches
    if len(validated) == 0:
        stmt = select(UserMCPConfig).where(UserMCPConfig.user_id == user.id)
        result = await db.execute(stmt)
        config_row = result.scalar_one_or_none()
        if config_row:
            await db.delete(config_row)
            await db.commit()
        await invalidate_mcp_config(user_id)
        logger.info(f"Cleared all MCP servers for user {user_id}")
        return {
            "status": "saved",
            "servers": [],
            "tools_discovered": 0,
            "config": {},
            "config_json": {},
        }

    # 2. Discover tools from remote servers (validates reachability)
    try:
        tool_schemas = await discover_tools(validated)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 3. Upsert DB row
    stmt = select(UserMCPConfig).where(UserMCPConfig.user_id == user.id)
    result = await db.execute(stmt)
    config_row = result.scalar_one_or_none()

    if config_row:
        config_row.config_json = validated
    else:
        config_row = UserMCPConfig(user_id=user.id, config_json=validated)
        db.add(config_row)

    await db.commit()

    # 4. Update Redis cache (config + schemas + version hash)
    version_hash = hashlib.md5(json.dumps(validated, sort_keys=True).encode()).hexdigest()
    await invalidate_mcp_config(user_id)
    await set_cached_mcp_config(user_id, validated)
    await set_cached_mcp_tool_schemas(user_id, tool_schemas)
    await set_mcp_config_version(user_id, version_hash)

    logger.info(
        f"Saved MCP config for user {user_id}: "
        f"{len(validated)} servers, {len(tool_schemas)} tools discovered"
    )

    return {
        "status": "saved",
        "servers": list(validated.keys()),
        "tools_discovered": len(tool_schemas),
        "tools": [{"name": t["name"], "description": t["description"]} for t in tool_schemas],
    }


# ── DELETE /mcp-config ────────────────────────────────────────────────────

@router.delete("/")
async def delete_mcp_config(
    user: User = Depends(current_scalekit_active_user),
    db: AsyncSession = Depends(get_async_session),
):
    """Delete the user's MCP config and clear all caches."""
    user_id = str(user.id)

    stmt = select(UserMCPConfig).where(UserMCPConfig.user_id == user.id)
    result = await db.execute(stmt)
    config_row = result.scalar_one_or_none()

    if not config_row:
        raise HTTPException(status_code=404, detail="No MCP config found.")

    await db.delete(config_row)
    await db.commit()

    await invalidate_mcp_config(user_id)

    logger.info(f"Deleted MCP config for user {user_id}")
    return {"status": "deleted"}


# ── POST /mcp-config/test ────────────────────────────────────────────────

@router.post("/test")
async def test_mcp_server(
    body: MCPTestRequest,
    user: User = Depends(current_scalekit_active_user),
):
    """
    Test-connect to a single MCP server without saving.
    Validates the URL and attempts tool discovery.
    """
    # Validate as a single-server config
    try:
        validated = validate_config({body.name: {"url": body.url, "headers": body.headers}})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Attempt discovery
    try:
        tool_schemas = await discover_tools(validated)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "reachable": True,
        "server": body.name,
        "tools": [{"name": t["name"], "description": t["description"]} for t in tool_schemas],
    }
