"""Knowledge-file metadata persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger
from app.domain.models import FileStatus, KnowledgeFile
from app.repositories.supabase_client import SupabaseClient

logger = get_logger(__name__)

TABLE = "knowledge_files"
COLUMNS = (
    "id,user_id,assistant_id,filename,file_type,storage_path,status,file_size,"
    "chunk_count,error_message,indexed_at,created_at,updated_at"
)


class KnowledgeRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._db = client

    async def list_for_assistant(self, assistant_id: str, user_id: str) -> list[KnowledgeFile]:
        rows = await self._db.select(
            TABLE,
            columns=COLUMNS,
            eq={"assistant_id": assistant_id, "user_id": user_id},
            order=("created_at", "desc"),
        )
        return [KnowledgeFile.model_validate(row) for row in rows]

    async def get(self, file_id: str, user_id: str) -> KnowledgeFile | None:
        row = await self._db.select_one(
            TABLE, columns=COLUMNS, eq={"id": file_id, "user_id": user_id}
        )
        return KnowledgeFile.model_validate(row) if row else None

    async def create(
        self,
        *,
        user_id: str,
        assistant_id: str,
        filename: str,
        file_type: str,
        storage_path: str,
        file_size: int | None,
    ) -> KnowledgeFile:
        row = await self._db.insert_one(
            TABLE,
            {
                "user_id": user_id,
                "assistant_id": assistant_id,
                "filename": filename,
                "file_type": file_type,
                "storage_path": storage_path,
                "file_size": file_size,
                "status": FileStatus.PROCESSING.value,
                "chunk_count": 0,
            },
        )
        return KnowledgeFile.model_validate(row)

    async def mark_processing(self, file_id: str, user_id: str) -> None:
        await self._db.update(
            TABLE,
            {"status": FileStatus.PROCESSING.value, "error_message": None},
            eq={"id": file_id, "user_id": user_id},
        )

    async def mark_indexed(self, file_id: str, user_id: str, chunk_count: int) -> None:
        await self._db.update(
            TABLE,
            {
                "status": FileStatus.INDEXED.value,
                "chunk_count": chunk_count,
                "error_message": None,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            },
            eq={"id": file_id, "user_id": user_id},
        )

    async def mark_failed(self, file_id: str, user_id: str, error: str) -> None:
        await self._db.update(
            TABLE,
            {"status": FileStatus.FAILED.value, "error_message": error[:1000]},
            eq={"id": file_id, "user_id": user_id},
        )

    async def delete(self, file_id: str, user_id: str) -> bool:
        rows = await self._db.delete(TABLE, eq={"id": file_id, "user_id": user_id})
        return bool(rows)

    async def stats_for_assistant(self, assistant_id: str, user_id: str) -> dict[str, Any]:
        files = await self.list_for_assistant(assistant_id, user_id)
        return {
            "total_files": len(files),
            "indexed": sum(1 for f in files if f.status == FileStatus.INDEXED),
            "processing": sum(1 for f in files if f.status == FileStatus.PROCESSING),
            "failed": sum(1 for f in files if f.status == FileStatus.FAILED),
            "total_chunks": sum(f.chunk_count for f in files),
            "total_bytes": sum(f.file_size or 0 for f in files),
        }
