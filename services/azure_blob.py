"""
Azure Blob storage helper.

This module intentionally keeps the upload API tiny so the rest of the app
can treat blob storage as an optional integration.
"""

from __future__ import annotations

from typing import Optional

from azure.storage.blob.aio import BlobServiceClient

from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _get_container_name() -> str:
    return "orphic-uploads"


async def upload_file_to_azure_blob(
    filename: str,
    content: bytes,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload a file to Azure Blob Storage and return its URL.

    If Azure connection settings are missing, raise a clear error so callers
    can treat blob upload as optional and continue without failing the request.
    """
    connection_string = settings.azure_storage_connection_string
    if not connection_string:
        raise RuntimeError("Azure storage connection string is not configured")

    container_name = _get_container_name()

    blob_name = filename.replace("\\", "/").split("/")[-1]

    async with BlobServiceClient.from_connection_string(connection_string) as service:
        container = service.get_container_client(container_name)
        try:
            await container.create_container(public_access="blob")
        except Exception:
            try:
                await container.set_container_access_policy(public_access="blob")
            except Exception:
                pass

        blob = container.get_blob_client(blob_name)
        await blob.upload_blob(
            content,
            overwrite=True,
            content_type=content_type,
        )

        return blob.url
