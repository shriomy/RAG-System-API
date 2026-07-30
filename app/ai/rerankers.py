"""Reranker implementations.

`RERANKER_PROVIDER=none` installs a pass-through that only trims to `top_n`, so
RetrievalService always runs the same three-step pipeline —
retrieve candidates -> rerank -> trim — whether or not a real reranker exists.
Turning reranking on is an env change plus an API key.
"""

from __future__ import annotations

from typing import Sequence

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.domain.models import RetrievedChunk
from app.domain.ports import Reranker

logger = get_logger(__name__)


class PassthroughReranker:
    """No-op reranker: preserves retrieval order, trims to `top_n`."""

    @property
    def name(self) -> str:
        return "passthrough"

    async def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        return list(chunks[:top_n])


class CohereReranker:
    """Cross-encoder reranking via Cohere Rerank."""

    def __init__(self, *, api_key: str, model: str) -> None:
        try:
            import cohere
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise AppError(
                "cohere is required for RERANKER_PROVIDER=cohere. "
                "Install it with: pip install cohere"
            ) from exc

        self._client = cohere.AsyncClientV2(api_key=api_key)
        self._model = model

    @property
    def name(self) -> str:
        return f"cohere:{self._model}"

    async def rerank(
        self, question: str, chunks: Sequence[RetrievedChunk], *, top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        try:
            response = await self._client.rerank(
                model=self._model,
                query=question,
                documents=[chunk.text for chunk in chunks],
                top_n=min(top_n, len(chunks)),
            )
        except Exception as exc:
            # Reranking is an enhancement; never fail a chat because of it.
            logger.warning("Cohere rerank failed, falling back to retrieval order: %s", exc)
            return list(chunks[:top_n])

        reranked: list[RetrievedChunk] = []
        for result in response.results:
            chunk = chunks[result.index].model_copy(
                update={"score": float(result.relevance_score)}
            )
            reranked.append(chunk)
        return reranked


def build_reranker(settings: Settings, *, provider: str | None = None) -> Reranker:
    """Resolve the configured reranker. Falls back to pass-through on misconfig."""
    chosen = (provider or settings.reranker_provider or "none").lower()

    if chosen in ("none", "", "passthrough"):
        return PassthroughReranker()

    if chosen == "cohere":
        if not settings.reranker_api_key:
            logger.warning("RERANKER_PROVIDER=cohere but RERANKER_API_KEY is unset; disabling")
            return PassthroughReranker()
        logger.info("Reranker: cohere (%s)", settings.reranker_model)
        return CohereReranker(
            api_key=settings.reranker_api_key, model=settings.reranker_model
        )

    logger.warning("Unknown RERANKER_PROVIDER '%s'; using pass-through", chosen)
    return PassthroughReranker()
