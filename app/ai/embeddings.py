"""LangChain embedding models.

OpenRouter serves chat completions only — it has no embeddings endpoint — so
the default is local `fastembed` (ONNX, no API key, no network after the first
model download). Switching to a hosted provider is an env change:

    EMBEDDING_PROVIDER=openai
    EMBEDDING_MODEL=text-embedding-3-small
    EMBEDDING_DIMENSION=1536
    EMBEDDING_API_KEY=sk-...

Changing the dimension requires a fresh Qdrant collection; QdrantService
detects the mismatch at startup and says so.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


def build_embeddings(settings: Settings) -> Embeddings:
    """Instantiate the configured LangChain Embeddings implementation."""
    provider = settings.embedding_provider

    if provider == "fastembed":
        try:
            from langchain_community.embeddings import FastEmbedEmbeddings
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise AppError(
                "fastembed is required for EMBEDDING_PROVIDER=fastembed. "
                "Install it with: pip install fastembed"
            ) from exc

        logger.info("Embeddings: fastembed (%s)", settings.embedding_model)
        return FastEmbedEmbeddings(
            model_name=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
        )

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.embedding_api_key:
            raise AppError("EMBEDDING_API_KEY is required for EMBEDDING_PROVIDER=openai.")

        logger.info("Embeddings: openai-compatible (%s)", settings.embedding_model)
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.embedding_api_key,  # type: ignore[arg-type]
            base_url=settings.embedding_base_url,
            dimensions=settings.embedding_dimension,
            chunk_size=settings.embedding_batch_size,
        )

    if provider == "voyage":
        try:
            from langchain_voyageai import VoyageAIEmbeddings
        except ImportError as exc:
            raise AppError(
                "langchain-voyageai is required for EMBEDDING_PROVIDER=voyage. "
                "Install it with: pip install langchain-voyageai"
            ) from exc

        if not settings.embedding_api_key:
            raise AppError("EMBEDDING_API_KEY is required for EMBEDDING_PROVIDER=voyage.")

        logger.info("Embeddings: voyage (%s)", settings.embedding_model)
        return VoyageAIEmbeddings(
            model=settings.embedding_model,
            voyage_api_key=settings.embedding_api_key,  # type: ignore[arg-type]
            batch_size=settings.embedding_batch_size,
        )

    raise AppError(f"Unknown EMBEDDING_PROVIDER '{provider}'.")
