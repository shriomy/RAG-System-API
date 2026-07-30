"""Redis cache adapter.

Inactive until CACHE_PROVIDER=redis. `redis` is imported lazily so the package
is a genuinely optional dependency.

Cache failures are logged and swallowed: a cache outage must degrade
performance, never correctness or availability.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisCache:
    def __init__(self, url: str, *, namespace: str = "rag", default_ttl: int = 300) -> None:
        try:
            from redis.asyncio import Redis, from_url
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise RuntimeError(
                "redis is required for CACHE_PROVIDER=redis. Install it with: pip install redis"
            ) from exc

        self._client: Redis = from_url(url, encoding="utf-8", decode_responses=True)
        self._namespace = namespace
        self._default_ttl = default_ttl

    @property
    def enabled(self) -> bool:
        return True

    def _key(self, key: str) -> str:
        return f"{self._namespace}:{key}"

    async def get(self, key: str) -> str | None:
        try:
            return await self._client.get(self._key(key))
        except Exception as exc:
            logger.warning("Cache GET failed for %s: %s", key, exc)
            return None

    async def set(self, key: str, value: str, *, ttl: int | None = None) -> None:
        try:
            await self._client.set(self._key(key), value, ex=ttl or self._default_ttl)
        except Exception as exc:
            logger.warning("Cache SET failed for %s: %s", key, exc)

    async def get_json(self, key: str) -> Any | None:
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Cache value for %s is not valid JSON; discarding", key)
            await self.delete(key)
            return None

    async def set_json(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        try:
            await self.set(key, json.dumps(value, default=str), ttl=ttl)
        except (TypeError, ValueError) as exc:
            logger.warning("Cache value for %s is not serialisable: %s", key, exc)

    async def delete(self, key: str) -> None:
        try:
            await self._client.delete(self._key(key))
        except Exception as exc:
            logger.warning("Cache DELETE failed for %s: %s", key, exc)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete every key under a prefix. Uses SCAN, never KEYS."""
        deleted = 0
        pattern = f"{self._key(prefix)}*"
        try:
            async for key in self._client.scan_iter(match=pattern, count=500):
                await self._client.delete(key)
                deleted += 1
        except Exception as exc:
            logger.warning("Cache prefix delete failed for %s: %s", prefix, exc)
        return deleted

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception as exc:  # pragma: no cover - shutdown path
            logger.debug("Redis close failed: %s", exc)
