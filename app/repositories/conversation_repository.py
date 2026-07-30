"""Conversation and message persistence."""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models import Conversation, Message, MessageRole
from app.repositories.supabase_client import SupabaseClient

logger = get_logger(__name__)

CONVERSATIONS = "conversations"
MESSAGES = "messages"

CONVERSATION_COLUMNS = "id,user_id,assistant_id,title,created_at,updated_at"
MESSAGE_COLUMNS = "id,conversation_id,user_id,role,content,created_at"


class ConversationRepository:
    def __init__(self, client: SupabaseClient) -> None:
        self._db = client

    # -- conversations -------------------------------------------------------

    async def list_for_user(
        self,
        user_id: str,
        *,
        assistant_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        eq: dict[str, str] = {"user_id": user_id}
        if assistant_id:
            eq["assistant_id"] = assistant_id
        rows = await self._db.select(
            CONVERSATIONS,
            columns=CONVERSATION_COLUMNS,
            eq=eq,
            order=("updated_at", "desc"),
            limit=limit,
            offset=offset,
        )
        return [Conversation.model_validate(row) for row in rows]

    async def get(self, conversation_id: str, user_id: str) -> Conversation | None:
        row = await self._db.select_one(
            CONVERSATIONS,
            columns=CONVERSATION_COLUMNS,
            eq={"id": conversation_id, "user_id": user_id},
        )
        return Conversation.model_validate(row) if row else None

    async def create(
        self, user_id: str, assistant_id: str | None, title: str
    ) -> Conversation:
        row = await self._db.insert_one(
            CONVERSATIONS,
            {"user_id": user_id, "assistant_id": assistant_id, "title": title},
        )
        return Conversation.model_validate(row)

    async def rename(self, conversation_id: str, user_id: str, title: str) -> Conversation | None:
        rows = await self._db.update(
            CONVERSATIONS, {"title": title}, eq={"id": conversation_id, "user_id": user_id}
        )
        return Conversation.model_validate(rows[0]) if rows else None

    async def delete(self, conversation_id: str, user_id: str) -> bool:
        rows = await self._db.delete(
            CONVERSATIONS, eq={"id": conversation_id, "user_id": user_id}
        )
        return bool(rows)

    # -- messages ------------------------------------------------------------

    async def list_messages(
        self,
        conversation_id: str,
        user_id: str,
        *,
        limit: int | None = None,
        order: str = "asc",
    ) -> list[Message]:
        rows = await self._db.select(
            MESSAGES,
            columns=MESSAGE_COLUMNS,
            eq={"conversation_id": conversation_id, "user_id": user_id},
            order=("created_at", "asc" if order == "asc" else "desc"),
            limit=limit,
        )
        return [Message.model_validate(row) for row in rows]

    async def list_recent_messages(
        self, conversation_id: str, user_id: str, limit: int
    ) -> list[Message]:
        """Newest `limit` messages, returned oldest-first for prompt assembly."""
        rows = await self._db.select(
            MESSAGES,
            columns=MESSAGE_COLUMNS,
            eq={"conversation_id": conversation_id, "user_id": user_id},
            order=("created_at", "desc"),
            limit=limit,
        )
        messages = [Message.model_validate(row) for row in rows]
        messages.reverse()
        return messages

    async def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: MessageRole | str,
        content: str,
    ) -> Message:
        row = await self._db.insert_one(
            MESSAGES,
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "role": str(role),
                "content": content,
            },
        )
        return Message.model_validate(row)

    async def count_messages(self, conversation_id: str, user_id: str) -> int:
        return await self._db.count(
            MESSAGES, eq={"conversation_id": conversation_id, "user_id": user_id}
        )
