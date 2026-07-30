"""KnowledgeService — the document ingestion pipeline.

    store file metadata
      -> load document      (LangChain loaders)
      -> split into chunks  (LangChain splitters)
      -> generate embeddings (EmbeddingService)
      -> upsert vectors      (QdrantService)
      -> tag with assistant_id (payload, enforced on every read)

Ingestion runs in the background so an upload returns immediately with
status='processing'; the frontend already polls that column.

Two entry points exist on purpose:

    ingest_upload()  the backend receives the file (POST /knowledge/upload)
    reindex_file()   the row already exists because the frontend uploaded
                     straight to Supabase Storage — index what is already there

The second keeps your current frontend working unchanged.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.ai.loaders import detect_file_type, load_documents
from app.ai.splitters import split_documents
from app.core.config import Settings
from app.core.errors import (
    AppError,
    NotFoundError,
    PayloadTooLargeError,
    ValidationError,
)
from app.core.logging import get_logger
from app.domain.models import DocumentChunk, FileStatus, KnowledgeFile
from app.repositories.assistant_repository import AssistantRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.repositories.storage_repository import StorageRepository
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class KnowledgeService:
    def __init__(
        self,
        *,
        knowledge_repository: KnowledgeRepository,
        storage_repository: StorageRepository,
        assistant_repository: AssistantRepository,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        retrieval_service: RetrievalService,
        settings: Settings,
    ) -> None:
        self._files = knowledge_repository
        self._storage = storage_repository
        self._assistants = assistant_repository
        self._embeddings = embedding_service
        self._qdrant = qdrant_service
        self._retrieval = retrieval_service
        self._settings = settings

    # ======================================================================
    # Reads
    # ======================================================================

    async def list_files(self, assistant_id: str, user_id: str) -> list[KnowledgeFile]:
        await self._require_assistant(assistant_id, user_id)
        return await self._files.list_for_assistant(assistant_id, user_id)

    async def get_file(self, file_id: str, user_id: str) -> KnowledgeFile:
        file = await self._files.get(file_id, user_id)
        if file is None:
            raise NotFoundError("Knowledge file not found.", details={"file_id": file_id})
        return file

    async def stats(self, assistant_id: str, user_id: str) -> dict[str, Any]:
        await self._require_assistant(assistant_id, user_id)
        stats = await self._files.stats_for_assistant(assistant_id, user_id)
        stats["vector_count"] = await self._qdrant.count(
            {"user_id": user_id, "assistant_id": assistant_id}
        )
        return stats

    async def signed_url(self, file_id: str, user_id: str, expires_in: int = 3600) -> str:
        file = await self.get_file(file_id, user_id)
        return await self._storage.signed_url(file.storage_path, expires_in)

    # ======================================================================
    # Upload
    # ======================================================================

    async def ingest_upload(
        self,
        *,
        user_id: str,
        assistant_id: str,
        filename: str,
        content: bytes,
    ) -> KnowledgeFile:
        """Validate, store, and register a file. Does NOT index it.

        The caller schedules `index_file` afterwards (a FastAPI BackgroundTask),
        so the HTTP response is not held open for embedding.
        """
        await self._require_assistant(assistant_id, user_id)

        if not content:
            raise ValidationError("The uploaded file is empty.")
        if len(content) > self._settings.max_upload_bytes:
            raise PayloadTooLargeError(
                f"File exceeds the {self._settings.max_upload_mb} MB limit.",
                details={"size_bytes": len(content), "limit_bytes": self._settings.max_upload_bytes},
            )

        file_type = detect_file_type(filename)

        storage_path = self._storage.build_path(user_id, assistant_id, filename)
        await self._storage.upload(storage_path, content, file_type)

        record = await self._files.create(
            user_id=user_id,
            assistant_id=assistant_id,
            filename=filename,
            file_type=file_type,
            storage_path=storage_path,
            file_size=len(content),
        )
        logger.info(
            "Registered knowledge file %s (%s, %d bytes) for assistant %s",
            record.id,
            filename,
            len(content),
            assistant_id,
        )
        return record

    # ======================================================================
    # Indexing
    # ======================================================================

    async def index_file(self, file_id: str, user_id: str) -> int:
        """Run the ingestion pipeline for one already-registered file.

        Returns the number of chunks indexed. Never raises: it records the
        failure on the row (status='failed', error_message) because it runs
        detached from any request that could report an error.
        """
        try:
            file = await self.get_file(file_id, user_id)
        except AppError as exc:
            logger.error("Cannot index %s: %s", file_id, exc.message)
            return 0

        try:
            await self._files.mark_processing(file_id, user_id)

            content = await self._storage.download(file.storage_path)

            # Loading and splitting are synchronous and CPU-bound — a large PDF
            # would otherwise block the event loop for seconds.
            documents = await asyncio.to_thread(
                load_documents,
                content,
                file_type=str(file.file_type),
                metadata={
                    "file_id": file.id,
                    "filename": file.filename,
                    "assistant_id": file.assistant_id,
                },
            )
            splits = await asyncio.to_thread(
                split_documents,
                documents,
                file_type=str(file.file_type),
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
            )
            if not splits:
                raise ValidationError("The document produced no indexable chunks.")

            chunks = [
                DocumentChunk(
                    id=f"{file.id}:{index}",
                    text=document.page_content,
                    user_id=user_id,
                    assistant_id=file.assistant_id,
                    file_id=file.id,
                    filename=file.filename,
                    file_type=str(file.file_type),
                    chunk_index=index,
                    metadata={
                        key: value
                        for key, value in document.metadata.items()
                        if key in {"page", "total_pages"} and value is not None
                    },
                )
                for index, document in enumerate(splits)
            ]

            vectors = await self._embeddings.embed_documents([c.text for c in chunks])

            # Re-indexing must not leave stale points behind: point ids are
            # derived from (file_id, chunk_index), so a file that shrinks would
            # otherwise keep its old tail chunks.
            await self._qdrant.delete_by_match(
                {"user_id": user_id, "file_id": file.id}
            )
            records = self._qdrant.to_records(chunks, vectors)
            written = await self._qdrant.upsert(records)

            await self._files.mark_indexed(file_id, user_id, written)
            await self._retrieval.invalidate_assistant(file.assistant_id)

            logger.info(
                "Indexed %s (%s): %d chunk(s) for assistant %s",
                file.id,
                file.filename,
                written,
                file.assistant_id,
            )
            return written

        except AppError as exc:
            logger.error("Indexing failed for %s: %s", file_id, exc.message)
            await self._safe_mark_failed(file_id, user_id, exc.message)
            return 0
        except Exception as exc:
            logger.exception("Indexing crashed for %s", file_id)
            await self._safe_mark_failed(file_id, user_id, str(exc))
            return 0

    async def reindex_file(self, file_id: str, user_id: str) -> KnowledgeFile:
        """Re-run indexing for an existing row and return its updated state.

        This is the bridge for files the frontend uploaded directly to Supabase
        Storage: the row exists with status='processing' but nothing has been
        embedded yet.
        """
        await self.get_file(file_id, user_id)
        await self.index_file(file_id, user_id)
        return await self.get_file(file_id, user_id)

    async def index_pending_for_assistant(self, assistant_id: str, user_id: str) -> int:
        """Index every file still sitting in 'processing'.

        Lets the frontend's direct-to-storage upload flow be picked up in one
        call, and doubles as a recovery path after a crash mid-ingestion.
        """
        files = await self._files.list_for_assistant(assistant_id, user_id)
        pending = [f for f in files if f.status == FileStatus.PROCESSING]

        indexed = 0
        for file in pending:
            if await self.index_file(file.id, user_id) > 0:
                indexed += 1
        return indexed

    # ======================================================================
    # Deletion
    # ======================================================================

    async def delete_file(self, file_id: str, user_id: str) -> None:
        """Delete a file everywhere: vectors, blob, then the metadata row.

        Ordered so a partial failure never leaves a row pointing at data that
        is gone; the row is removed last.
        """
        file = await self.get_file(file_id, user_id)

        try:
            await self._qdrant.delete_by_match({"user_id": user_id, "file_id": file.id})
        except AppError as exc:
            # Do not block the user's delete on the vector store.
            logger.warning("Could not delete vectors for %s: %s", file.id, exc.message)

        await self._storage.remove(file.storage_path)
        await self._files.delete(file_id, user_id)
        await self._retrieval.invalidate_assistant(file.assistant_id)

        logger.info("Deleted knowledge file %s (%s)", file.id, file.filename)

    async def delete_assistant_knowledge(self, assistant_id: str, user_id: str) -> None:
        """Purge every vector for an assistant. Called when an assistant is deleted.

        The `knowledge_files` rows and storage objects are handled by the DB
        cascade and by explicit blob removal here.
        """
        files = await self._files.list_for_assistant(assistant_id, user_id)

        try:
            await self._qdrant.delete_by_match(
                {"user_id": user_id, "assistant_id": assistant_id}
            )
        except AppError as exc:
            logger.warning(
                "Could not purge vectors for assistant %s: %s", assistant_id, exc.message
            )

        for file in files:
            await self._storage.remove(file.storage_path)

        await self._retrieval.invalidate_assistant(assistant_id)
        logger.info("Purged knowledge for assistant %s (%d file(s))", assistant_id, len(files))

    # ======================================================================
    # Internals
    # ======================================================================

    async def _require_assistant(self, assistant_id: str, user_id: str) -> None:
        if not await self._assistants.exists(assistant_id, user_id):
            raise NotFoundError(
                "Assistant not found.", details={"assistant_id": assistant_id}
            )

    async def _safe_mark_failed(self, file_id: str, user_id: str, error: str) -> None:
        try:
            await self._files.mark_failed(file_id, user_id, error)
        except Exception as exc:  # pragma: no cover - best-effort bookkeeping
            logger.error("Could not record failure for %s: %s", file_id, exc)
