"""EmbeddingService — text to vectors.

Wraps the LangChain Embeddings object from app/ai/embeddings.py so that no other
service imports LangChain. Embedding calls are pushed to a worker thread because
the default provider (fastembed) is synchronous and CPU-bound; blocking the
event loop with it would stall every concurrent request.

Query embeddings are cached when a cache is configured — the same question from
the same user is common, and it is the cheapest win available.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.errors import UpstreamError
from app.core.logging import get_logger
from app.domain.ports import Cache
from app.infrastructure.cache.factory import embedding_key

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(
        self,
        embeddings: Embeddings,
        settings: Settings,
        cache: Cache,
    ) -> None:
        self._embeddings = embeddings
        self._settings = settings
        self._cache = cache
        self._dimension = settings.embedding_dimension
        self._model_name = settings.embedding_model

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of chunks. Batched to bound peak memory on big files."""
        if not texts:
            return []

        batch_size = max(1, self._settings.embedding_batch_size)
        vectors: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            try:
                batch_vectors = await asyncio.to_thread(
                    self._embeddings.embed_documents, batch
                )
            except Exception as exc:
                raise UpstreamError(f"Embedding generation failed: {exc}") from exc
            vectors.extend(batch_vectors)

        self._verify_dimension(vectors[0] if vectors else None)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query, using the cache when one is configured."""
        cleaned = text.strip()
        if not cleaned:
            raise UpstreamError("Cannot embed an empty query.")

        key = embedding_key(self._model_name, cleaned)
        if self._cache.enabled:
            cached = await self._cache.get_json(key)
            if isinstance(cached, list) and cached:
                return [float(v) for v in cached]

        try:
            vector = await asyncio.to_thread(self._embeddings.embed_query, cleaned)
        except Exception as exc:
            raise UpstreamError(f"Embedding generation failed: {exc}") from exc

        self._verify_dimension(vector)

        if self._cache.enabled:
            await self._cache.set_json(key, vector, ttl=self._settings.cache_ttl_seconds)

        return vector

    async def warm_up(self) -> None:
        """Force model download / ONNX session creation at startup.

        Without this, the very first chat request pays a multi-second penalty
        while fastembed fetches and initialises the model.
        """
        try:
            vector = await asyncio.to_thread(self._embeddings.embed_query, "warm up")
        except Exception as exc:
            logger.warning("Embedding warm-up failed: %s", exc)
            return

        actual = len(vector)
        if actual != self._dimension:
            # Trust the model over the config, and say so loudly: a mismatch
            # would otherwise fail later as an opaque Qdrant dimension error.
            logger.warning(
                "EMBEDDING_DIMENSION is %d but model '%s' returns %d. "
                "Using %d — update your .env and recreate the Qdrant collection.",
                self._dimension,
                self._model_name,
                actual,
                actual,
            )
            self._dimension = actual
        else:
            logger.info(
                "Embeddings warm: model=%s dimension=%d", self._model_name, self._dimension
            )

    def _verify_dimension(self, vector: Sequence[float] | None) -> None:
        if vector is None:
            return
        if len(vector) != self._dimension:
            raise UpstreamError(
                f"Embedding dimension mismatch: model returned {len(vector)}, "
                f"expected {self._dimension}. Update EMBEDDING_DIMENSION and "
                f"recreate the Qdrant collection."
            )
