"""Cache adapters."""

from app.infrastructure.cache.factory import build_cache
from app.infrastructure.cache.null_cache import NullCache
from app.infrastructure.cache.redis_cache import RedisCache

__all__ = ["build_cache", "NullCache", "RedisCache"]
