"""Knowledge-file DTOs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models import KnowledgeFile


class KnowledgeFileResponse(BaseModel):
    """Matches the frontend `KnowledgeFile` type, plus ingestion bookkeeping."""

    id: str
    user_id: str
    assistant_id: str
    filename: str
    file_type: str
    storage_path: str
    status: str
    file_size: int | None = None
    chunk_count: int = 0
    error_message: str | None = None
    indexed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def of(cls, file: KnowledgeFile) -> "KnowledgeFileResponse":
        return cls.model_validate(file.model_dump())


class KnowledgeStatsResponse(BaseModel):
    total_files: int = 0
    indexed: int = 0
    processing: int = 0
    failed: int = 0
    total_chunks: int = 0
    total_bytes: int = 0
    #: Points actually present in Qdrant — the authoritative retrieval count.
    vector_count: int = 0


class SignedUrlResponse(BaseModel):
    url: str
    expires_in: int


class IndexPendingResponse(BaseModel):
    """Result of indexing files that were uploaded straight to Supabase Storage."""

    scheduled: int = Field(description="Number of files queued for indexing.")
    assistant_id: str
