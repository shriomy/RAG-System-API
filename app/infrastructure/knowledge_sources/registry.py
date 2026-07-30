"""Knowledge-source registry.

To add a source later:

    1. Write a class implementing the `KnowledgeSource` port.
    2. Register a factory here with @register_knowledge_source("my_source").
    3. Enable it globally, or per assistant via `config.retrieval.sources`.

RetrievalService then fans out to it automatically. No graph, service or route
changes.
"""

from __future__ import annotations

from typing import Callable, Protocol

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.ports import KnowledgeSource
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService

logger = get_logger(__name__)


class SourceDeps(Protocol):
    """What a knowledge-source factory may draw on."""

    settings: Settings
    embedding_service: EmbeddingService
    qdrant_service: QdrantService


KnowledgeSourceFactory = Callable[[SourceDeps], KnowledgeSource]

KNOWLEDGE_SOURCE_REGISTRY: dict[str, KnowledgeSourceFactory] = {}


def register_knowledge_source(
    name: str,
) -> Callable[[KnowledgeSourceFactory], KnowledgeSourceFactory]:
    def decorator(factory: KnowledgeSourceFactory) -> KnowledgeSourceFactory:
        KNOWLEDGE_SOURCE_REGISTRY[name] = factory
        return factory

    return decorator


@register_knowledge_source("vector")
def _build_vector_source(deps: SourceDeps) -> KnowledgeSource:
    from app.infrastructure.knowledge_sources.vector_source import VectorKnowledgeSource

    return VectorKnowledgeSource(
        embedding_service=deps.embedding_service,
        qdrant_service=deps.qdrant_service,
    )


def build_knowledge_sources(
    deps: SourceDeps, *, names: list[str] | None = None
) -> list[KnowledgeSource]:
    """Instantiate the requested sources, defaulting to the vector source."""
    selected = names or ["vector"]

    sources: list[KnowledgeSource] = []
    for name in selected:
        factory = KNOWLEDGE_SOURCE_REGISTRY.get(name)
        if factory is None:
            logger.warning(
                "Unknown knowledge source '%s'; known: %s",
                name,
                sorted(KNOWLEDGE_SOURCE_REGISTRY),
            )
            continue
        sources.append(factory(deps))

    if not sources:
        logger.warning("No knowledge sources resolved; retrieval will return nothing")
    return sources
