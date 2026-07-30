"""Knowledge-source adapters."""

from app.infrastructure.knowledge_sources.registry import (
    KNOWLEDGE_SOURCE_REGISTRY,
    register_knowledge_source,
)
from app.infrastructure.knowledge_sources.vector_source import VectorKnowledgeSource

__all__ = [
    "KNOWLEDGE_SOURCE_REGISTRY",
    "register_knowledge_source",
    "VectorKnowledgeSource",
]
