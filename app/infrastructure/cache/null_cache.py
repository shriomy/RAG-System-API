"""No-op cache — the default.

Because it satisfies the same port, call sites can cache unconditionally and
configuration alone decides whether anything is stored. That is what makes
"add Redis later" a zero-diff change outside this package.
"""

from __future__ import annotations

from typing import Any


class NullCache:
    @property
    def enabled(self) -> bool:
        return False

    async def get(self, key: str) -> str | None:
        return None

    async def set(self, key: str, value: str, *, ttl: int | None = None) -> None:
        return None

    async def get_json(self, key: str) -> Any | None:
        return None

    async def set_json(self, key: str, value: Any, *, ttl: int | None = None) -> None:
        return None

    async def delete(self, key: str) -> None:
        return None

    async def delete_prefix(self, prefix: str) -> int:
        return 0

    async def close(self) -> None:
        return None
