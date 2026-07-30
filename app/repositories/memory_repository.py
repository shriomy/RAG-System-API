"""Long-term memory persistence.

Phase 1 stores two rolling summaries. Semantic memory would arrive as new
repository methods plus a new table — this class's callers would not change.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.core.logging import get_logger
from app.domain.models import ConversationMemory, UserMemory
from app.repositories.supabase_client import SupabaseClient

logger = get_logger(__name__)

USER_MEMORY = "user_memory"
CONVERSATION_MEMORY = "conversation_memory"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._db = client

    # -- user-level ----------------------------------------------------------

    async def get_user_memory(self, user_id: str) -> UserMemory:
        row = await self._db.select_one(USER_MEMORY, eq={"user_id": user_id})
        if not row:
            return UserMemory(user_id=user_id)
        return UserMemory.model_validate(row)

    async def upsert_user_memory(
        self, user_id: str, *, summary: str, turn_count: int
    ) -> UserMemory:
        rows = await self._db.upsert(
            USER_MEMORY,
            {
                "user_id": user_id,
                "summary": summary,
                "turn_count": turn_count,
                "updated_at": _now(),
            },
            on_conflict="user_id",
        )
        return UserMemory.model_validate(rows[0]) if rows else UserMemory(
            user_id=user_id, summary=summary, turn_count=turn_count
        )

    async def delete_user_memory(self, user_id: str) -> bool:
        rows = await self._db.delete(USER_MEMORY, eq={"user_id": user_id})
        return bool(rows)

    # -- conversation-level --------------------------------------------------

    async def get_conversation_memory(
        self, conversation_id: str, user_id: str
    ) -> ConversationMemory:
        row = await self._db.select_one(
            CONVERSATION_MEMORY, eq={"conversation_id": conversation_id, "user_id": user_id}
        )
        if not row:
            return ConversationMemory(conversation_id=conversation_id, user_id=user_id)
        return ConversationMemory.model_validate(row)

    async def upsert_conversation_memory(
        self,
        conversation_id: str,
        user_id: str,
        *,
        summary: str,
        message_count: int,
    ) -> ConversationMemory:
        rows = await self._db.upsert(
            CONVERSATION_MEMORY,
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "summary": summary,
                "message_count": message_count,
                "updated_at": _now(),
            },
            on_conflict="conversation_id",
        )
        if rows:
            return ConversationMemory.model_validate(rows[0])
        return ConversationMemory(
            conversation_id=conversation_id,
            user_id=user_id,
            summary=summary,
            message_count=message_count,
        )

    async def delete_conversation_memory(self, conversation_id: str, user_id: str) -> bool:
        rows = await self._db.delete(
            CONVERSATION_MEMORY, eq={"conversation_id": conversation_id, "user_id": user_id}
        )
        return bool(rows)
