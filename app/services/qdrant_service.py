"""QdrantService — vector persistence.

Multi-tenancy follows Qdrant's own recommendation: ONE collection, with every
point carrying `user_id` / `assistant_id` in its payload and every query
filtered on them. Keyword payload indexes make those filters cheap. Adding a
second assistant, user or file is a payload value, not a new collection.

Implements the `VectorStore` port, so swapping in pgvector, Weaviate or Pinecone
later means writing one class — RetrievalService and KnowledgeService are
unaffected.
"""

from __future__ import annotations

import uuid
import warnings
from typing import Any, Sequence

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.ai.vectorstore import (
    PAYLOAD_ASSISTANT_ID,
    PAYLOAD_CHUNK_INDEX,
    PAYLOAD_FILE_ID,
    PAYLOAD_FILENAME,
    PAYLOAD_TEXT,
    PAYLOAD_USER_ID,
    tenant_filter,
)
from app.core.config import Settings
from app.core.errors import UpstreamError
from app.core.logging import get_logger
from app.domain.models import DocumentChunk, RetrievedChunk, VectorRecord

logger = get_logger(__name__)

#: Namespace for deterministic point IDs, so re-indexing a file overwrites its
#: points instead of duplicating them.
_POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

#: Payload fields that get a keyword index. Filters only stay fast if indexed.
_INDEXED_FIELDS = (PAYLOAD_USER_ID, PAYLOAD_ASSISTANT_ID, PAYLOAD_FILE_ID)


def point_id(file_id: str, chunk_index: int) -> str:
    """Stable UUID for a (file, chunk) pair."""
    return str(uuid.uuid5(_POINT_NAMESPACE, f"{file_id}:{chunk_index}"))


class QdrantService:
    def __init__(
        self,
        settings: Settings,
        *,
        vector_size: int,
        client: AsyncQdrantClient | None = None,
    ) -> None:
        """`client` is injectable so tests can use Qdrant's embedded mode, and so
        an alternative transport (gRPC, Qdrant Cloud with custom TLS) can be
        supplied without subclassing."""
        self._settings = settings
        self._collection = settings.qdrant_collection
        self._vector_size = vector_size
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_timeout_seconds),
        )

    @property
    def client(self) -> AsyncQdrantClient:
        """Exposed for the LangChain vector-store adapter in app/ai/vectorstore.py."""
        return self._client

    @property
    def collection(self) -> str:
        return self._collection

    async def aclose(self) -> None:
        try:
            await self._client.close()
        except Exception as exc:  # pragma: no cover - shutdown path
            logger.debug("Qdrant close failed: %s", exc)

    # ======================================================================
    # Lifecycle
    # ======================================================================

    async def ensure_ready(self, *, vector_size: int | None = None) -> None:
        """Create the collection and payload indexes if they are missing."""
        if vector_size:
            self._vector_size = vector_size

        try:
            exists = await self._client.collection_exists(self._collection)
        except Exception as exc:
            raise UpstreamError(
                f"Cannot reach Qdrant at {self._settings.qdrant_url}: {exc}"
            ) from exc

        if not exists:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, cosine)",
                self._collection,
                self._vector_size,
            )
        else:
            await self._check_dimension()

        await self._ensure_payload_indexes()

    async def _check_dimension(self) -> None:
        try:
            info = await self._client.get_collection(self._collection)
            params = info.config.params.vectors
            existing = params.size if isinstance(params, qmodels.VectorParams) else None
        except Exception as exc:
            logger.warning("Could not read Qdrant collection config: %s", exc)
            return

        if existing and existing != self._vector_size:
            raise UpstreamError(
                f"Qdrant collection '{self._collection}' has dimension {existing}, "
                f"but the embedding model produces {self._vector_size}. Either set "
                f"QDRANT_COLLECTION to a new name or delete the existing collection."
            )

    async def _ensure_payload_indexes(self) -> None:
        for field in _INDEXED_FIELDS:
            try:
                with warnings.catch_warnings():
                    # Embedded/local Qdrant warns that indexes are a no-op there.
                    # Filters still work; only the speed-up is absent.
                    warnings.simplefilter("ignore", UserWarning)
                    await self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=qmodels.PayloadSchemaType.KEYWORD,
                        wait=True,
                    )
            except UnexpectedResponse:
                # Already exists — Qdrant returns 4xx rather than a no-op.
                continue
            except Exception as exc:
                logger.debug("Payload index for '%s' not created: %s", field, exc)

    async def health(self) -> bool:
        try:
            await self._client.get_collections()
            return True
        except Exception:
            return False

    # ======================================================================
    # Writes
    # ======================================================================

    @staticmethod
    def to_records(chunks: Sequence[DocumentChunk], vectors: Sequence[Sequence[float]]) -> list[VectorRecord]:
        """Pair chunks with their vectors, building the searchable payload."""
        if len(chunks) != len(vectors):
            raise UpstreamError(
                f"Chunk/vector count mismatch: {len(chunks)} chunks, {len(vectors)} vectors."
            )

        records: list[VectorRecord] = []
        for chunk, vector in zip(chunks, vectors):
            records.append(
                VectorRecord(
                    id=point_id(chunk.file_id, chunk.chunk_index),
                    vector=[float(v) for v in vector],
                    payload={
                        PAYLOAD_TEXT: chunk.text,
                        PAYLOAD_USER_ID: chunk.user_id,
                        PAYLOAD_ASSISTANT_ID: chunk.assistant_id,
                        PAYLOAD_FILE_ID: chunk.file_id,
                        PAYLOAD_FILENAME: chunk.filename,
                        PAYLOAD_CHUNK_INDEX: chunk.chunk_index,
                        "file_type": chunk.file_type,
                        **chunk.metadata,
                    },
                )
            )
        return records

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0

        points = [
            qmodels.PointStruct(id=record.id, vector=record.vector, payload=record.payload)
            for record in records
        ]

        try:
            # Batched so a large file does not produce one oversized request.
            batch_size = 128
            for start in range(0, len(points), batch_size):
                await self._client.upsert(
                    collection_name=self._collection,
                    points=points[start : start + batch_size],
                    wait=True,
                )
        except Exception as exc:
            raise UpstreamError(f"Qdrant upsert failed: {exc}") from exc

        logger.debug("Upserted %d point(s) into '%s'", len(points), self._collection)
        return len(points)

    async def delete_by_match(self, must_match: dict[str, Any]) -> int:
        if not must_match:
            raise ValueError("delete_by_match() requires filters — refusing to wipe collection")

        try:
            existing = await self.count(must_match)
            await self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(filter=tenant_filter(**must_match)),
                wait=True,
            )
        except Exception as exc:
            raise UpstreamError(f"Qdrant delete failed: {exc}") from exc

        logger.debug("Deleted %d point(s) matching %s", existing, must_match)
        return existing

    # ======================================================================
    # Reads
    # ======================================================================

    async def search(
        self,
        vector: Sequence[float],
        *,
        must_match: dict[str, Any],
        limit: int,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=list(vector),
                query_filter=tenant_filter(**must_match),
                limit=max(1, limit),
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            raise UpstreamError(f"Qdrant search failed: {exc}") from exc

        chunks: list[RetrievedChunk] = []
        for point in response.points:
            payload = point.payload or {}
            text = payload.get(PAYLOAD_TEXT, "")
            if not text:
                continue
            chunks.append(
                RetrievedChunk(
                    id=str(point.id),
                    text=text,
                    score=float(point.score or 0.0),
                    source="vector",
                    filename=payload.get(PAYLOAD_FILENAME),
                    file_id=payload.get(PAYLOAD_FILE_ID),
                    chunk_index=payload.get(PAYLOAD_CHUNK_INDEX),
                    metadata={
                        k: v
                        for k, v in payload.items()
                        if k
                        not in {
                            PAYLOAD_TEXT,
                            PAYLOAD_USER_ID,
                            PAYLOAD_ASSISTANT_ID,
                            PAYLOAD_FILE_ID,
                            PAYLOAD_FILENAME,
                            PAYLOAD_CHUNK_INDEX,
                        }
                    },
                )
            )
        return chunks

    async def count(self, must_match: dict[str, Any]) -> int:
        try:
            result = await self._client.count(
                collection_name=self._collection,
                count_filter=tenant_filter(**must_match),
                exact=True,
            )
            return int(result.count)
        except Exception as exc:
            logger.warning("Qdrant count failed for %s: %s", must_match, exc)
            return 0
