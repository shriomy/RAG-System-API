"""Assistant persistence."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger
from app.domain.models import Assistant
from app.repositories.supabase_client import Row, SupabaseClient

logger = get_logger(__name__)

TABLE = "assistants"
COLUMNS = (
    "id,user_id,name,system_prompt,is_active,model,temperature,max_tokens,"
    "config,created_at,updated_at"
)


class AssistantRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._db = client

    async def list_for_user(self, user_id: str) -> list[Assistant]:
        rows = await self._db.select(
            TABLE,
            columns=COLUMNS,
            eq={"user_id": user_id},
            order=("created_at", "asc"),
        )
        return [Assistant.model_validate(row) for row in rows]

    async def get(self, assistant_id: str, user_id: str) -> Assistant | None:
        row = await self._db.select_one(
            TABLE,
            columns=COLUMNS,
            eq={"id": assistant_id, "user_id": user_id},
        )
        return Assistant.model_validate(row) if row else None

    async def get_active(self, user_id: str) -> Assistant | None:
        row = await self._db.select_one(
            TABLE,
            columns=COLUMNS,
            eq={"user_id": user_id, "is_active": True},
        )
        return Assistant.model_validate(row) if row else None

    async def create(self, user_id: str, payload: dict[str, Any]) -> Assistant:
        row = await self._db.insert_one(TABLE, {**payload, "user_id": user_id})
        return Assistant.model_validate(row)

    async def update(
        self, assistant_id: str, user_id: str, payload: dict[str, Any]
    ) -> Assistant | None:
        rows = await self._db.update(
            TABLE, payload, eq={"id": assistant_id, "user_id": user_id}
        )
        return Assistant.model_validate(rows[0]) if rows else None

    async def delete(self, assistant_id: str, user_id: str) -> bool:
        rows = await self._db.delete(TABLE, eq={"id": assistant_id, "user_id": user_id})
        return bool(rows)

    async def deactivate_all(self, user_id: str, *, except_id: str | None = None) -> None:
        """Clear the active flag for a user's assistants.

        PostgREST has no `neq` helper on our client, so the excluded row is
        simply re-activated by the caller afterwards.
        """
        await self._db.update(TABLE, {"is_active": False}, eq={"user_id": user_id})
        if except_id:
            await self._db.update(
                TABLE, {"is_active": True}, eq={"id": except_id, "user_id": user_id}
            )

    async def exists(self, assistant_id: str, user_id: str) -> bool:
        row: Row | None = await self._db.select_one(
            TABLE, columns="id", eq={"id": assistant_id, "user_id": user_id}
        )
        return row is not None
