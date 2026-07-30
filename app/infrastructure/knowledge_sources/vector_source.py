"""The Qdrant-backed knowledge source — the only one enabled today.

It implements the same `KnowledgeSource` port that a SQL source, a web-search
source or a second vector collection would, which is why RetrievalService's
fan-out logic does not need to change when more are added.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models import RetrievalQuery, RetrievedChunk
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService

logger = get_logger(__name__)


class VectorKnowledgeSource:
    """Dense-vector similarity search over the user's indexed documents."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        qdrant_service: QdrantService,
        name: str = "vector",
    ) -> None:
        self._embeddings = embedding_service
        self._qdrant = qdrant_service
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        vector = await self._embeddings.embed_query(query.question)

        must_match = {
            "user_id": query.user_id,
            "assistant_id": query.assistant_id,
            **query.filters,
        }

        chunks = await self._qdrant.search(
            vector,
            must_match=must_match,
            limit=query.candidate_k,
            score_threshold=query.score_threshold,
        )

        for chunk in chunks:
            chunk.source = self._name
        return chunks
