"""RetrievalService — the retrieval pipeline.

Three fixed stages, whatever is plugged in:

    1. FAN OUT   ask every enabled KnowledgeSource, concurrently
    2. MERGE     deduplicate and sort the combined candidate pool
    3. RERANK    reorder and trim to top_k

Today stage 1 has one source (Qdrant) and stage 3 uses a pass-through
reranker. Adding a source or switching on Cohere reranking changes what those
stages contain, not the shape of the pipeline — so the graph node that calls
this service never changes.
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import Assistant, RetrievalConfig, RetrievalQuery, RetrievedChunk
from app.domain.ports import Cache, KnowledgeSource, Reranker
from app.infrastructure.cache.factory import retrieval_key

logger = get_logger(__name__)


class RetrievalService:
    def __init__(
        self,
        *,
        sources: Sequence[KnowledgeSource],
        reranker: Reranker,
        settings: Settings,
        cache: Cache,
    ) -> None:
        self._sources = list(sources)
        self._reranker = reranker
        self._settings = settings
        self._cache = cache

    @property
    def source_names(self) -> list[str]:
        return [source.name for source in self._sources]

    @property
    def reranker_name(self) -> str:
        return self._reranker.name

    async def retrieve(
        self,
        *,
        question: str,
        user_id: str,
        assistant_id: str,
        config: RetrievalConfig,
    ) -> list[RetrievedChunk]:
        """Run the pipeline and return the chunks to ground the answer on."""
        cleaned = question.strip()
        if not cleaned or not self._sources:
            return []

        cache_key = retrieval_key(assistant_id, cleaned)
        if self._cache.enabled:
            cached = await self._cache.get_json(cache_key)
            if isinstance(cached, list):
                return [RetrievedChunk.model_validate(item) for item in cached]

        query = RetrievalQuery(
            question=cleaned,
            user_id=user_id,
            assistant_id=assistant_id,
            top_k=config.top_k,
            candidate_k=max(config.candidate_k, config.top_k),
            score_threshold=config.score_threshold,
        )

        candidates = await self._fan_out(query)
        if not candidates:
            logger.debug("No candidates for assistant %s", assistant_id)
            return []

        merged = self._merge(candidates)
        final = await self._reranker.rerank(cleaned, merged, top_n=config.top_k)

        if self._cache.enabled:
            await self._cache.set_json(
                cache_key,
                [chunk.model_dump(mode="json") for chunk in final],
                ttl=self._settings.cache_ttl_seconds,
            )

        logger.debug(
            "Retrieved %d/%d chunk(s) for assistant %s [sources=%s reranker=%s]",
            len(final),
            len(merged),
            assistant_id,
            self.source_names,
            self._reranker.name,
        )
        return final

    # ======================================================================
    # Stages
    # ======================================================================

    async def _fan_out(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """Query every source concurrently; one failing source must not fail the turn."""
        results = await asyncio.gather(
            *(source.retrieve(query) for source in self._sources),
            return_exceptions=True,
        )

        candidates: list[RetrievedChunk] = []
        for source, result in zip(self._sources, results):
            if isinstance(result, BaseException):
                logger.warning("Knowledge source '%s' failed: %s", source.name, result)
                continue
            candidates.extend(result)
        return candidates

    @staticmethod
    def _merge(candidates: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Deduplicate by id, keeping the best score, then sort by score."""
        best: dict[str, RetrievedChunk] = {}
        for chunk in candidates:
            existing = best.get(chunk.id)
            if existing is None or chunk.score > existing.score:
                best[chunk.id] = chunk
        return sorted(best.values(), key=lambda c: c.score, reverse=True)

    # ======================================================================
    # Cache invalidation
    # ======================================================================

    async def invalidate_assistant(self, assistant_id: str) -> None:
        """Drop cached retrievals after the knowledge base changes."""
        if self._cache.enabled:
            await self._cache.delete_prefix(f"retrieval:{assistant_id}:")

    # ======================================================================
    # Config helper
    # ======================================================================

    def config_for(self, assistant: Assistant) -> RetrievalConfig:
        defaults = RetrievalConfig(
            top_k=self._settings.retrieval_top_k,
            candidate_k=self._settings.retrieval_candidate_k,
            score_threshold=self._settings.retrieval_score_threshold,
            reranker=self._settings.reranker_provider,
            sources=self.source_names,
        )
        return assistant.to_retrieval_config(defaults=defaults)
