"""Supabase Storage access for uploaded knowledge files."""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone

from app.core.config import Settings
from app.core.logging import get_logger
from app.repositories.supabase_client import SupabaseClient

logger = get_logger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

CONTENT_TYPES = {
    "pdf": "application/pdf",
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
}


def sanitize_filename(filename: str) -> str:
    """Make a filename safe for a storage object key, preserving readability."""
    normalized = unicodedata.normalize("NFKD", filename).encode("ascii", "ignore").decode()
    cleaned = _UNSAFE.sub("-", normalized).strip("-._")
    return (cleaned or "file")[:120]


class StorageRepository:
    def __init__(self, client: SupabaseClient, settings: Settings) -> None:
        self._db = client
        self._bucket = settings.supabase_storage_bucket

    def build_path(self, user_id: str, assistant_id: str, filename: str) -> str:
        """Match the frontend's layout: `<user_id>/<assistant_id>/<ts>-<name>`.

        The leading user_id segment is what the storage RLS policy checks.
        """
        stamp = int(datetime.now(timezone.utc).timestamp() * 1000)
        return f"{user_id}/{assistant_id}/{stamp}-{sanitize_filename(filename)}"

    async def upload(self, path: str, content: bytes, file_type: str) -> None:
        await self._db.storage_upload(
            self._bucket,
            path,
            content,
            content_type=CONTENT_TYPES.get(file_type, "application/octet-stream"),
            upsert=True,
        )

    async def download(self, path: str) -> bytes:
        return await self._db.storage_download(self._bucket, path)

    async def remove(self, path: str) -> None:
        await self._db.storage_remove(self._bucket, [path])

    async def signed_url(self, path: str, expires_in: int = 3600) -> str:
        return await self._db.storage_signed_url(self._bucket, path, expires_in)
