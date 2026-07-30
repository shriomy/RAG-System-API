"""Cache selection."""

from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.ports import Cache
from app.infrastructure.cache.null_cache import NullCache
from app.infrastructure.cache.redis_cache import RedisCache

logger = get_logger(__name__)


def build_cache(settings: Settings) -> Cache:
    if settings.cache_provider == "redis":
        try:
            cache = RedisCache(
                settings.redis_url,
                namespace=settings.cache_namespace,
                default_ttl=settings.cache_ttl_seconds,
            )
            logger.info("Cache: redis (%s)", settings.redis_url)
            return cache
        except RuntimeError as exc:
            # Missing driver should not stop the app from serving traffic.
            logger.error("Falling back to no-op cache: %s", exc)
            return NullCache()

    logger.info("Cache: disabled")
    return NullCache()


# Canonical cache-key builders. Centralised so invalidation stays consistent
# once caching is switched on.


def assistant_key(user_id: str, assistant_id: str) -> str:
    return f"assistant:{user_id}:{assistant_id}"


def assistant_list_key(user_id: str) -> str:
    return f"assistants:{user_id}"


def memory_key(user_id: str) -> str:
    return f"memory:user:{user_id}"


def conversation_memory_key(conversation_id: str) -> str:
    return f"memory:conversation:{conversation_id}"


def retrieval_key(assistant_id: str, question: str) -> str:
    import hashlib

    digest = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:32]
    return f"retrieval:{assistant_id}:{digest}"


def embedding_key(model: str, text: str) -> str:
    import hashlib

    digest = hashlib.sha256(text.encode()).hexdigest()[:32]
    return f"embedding:{model}:{digest}"
