"""MemoryService — long-term memory.

Phase 1, as specified: a rolling user summary, a rolling conversation summary,
and the verbatim last N messages. No semantic memory, no memory retrieval.

The seam for those additions is `load_context()`: it returns a MemoryContext
that the prompt builder consumes. When semantic memory arrives it becomes an
extra field populated by an extra repository call — the graph node, the prompt
builder's signature and the API stay as they are.
"""

from __future__ import annotations

from app.ai.prompts import CONVERSATION_SUMMARY_PROMPT, USER_SUMMARY_PROMPT
from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import (
    ChatTurn,
    ConversationMemory,
    MemoryContext,
    MessageRole,
    UserMemory,
)
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.memory_repository import MemoryRepository
from app.services.openrouter_service import OpenRouterService

logger = get_logger(__name__)


class MemoryService:
    def __init__(
        self,
        *,
        memory_repository: MemoryRepository,
        conversation_repository: ConversationRepository,
        llm: OpenRouterService,
        settings: Settings,
    ) -> None:
        self._memory = memory_repository
        self._conversations = conversation_repository
        self._llm = llm
        self._settings = settings

    # ======================================================================
    # Read path (graph: Load User Memory)
    # ======================================================================

    async def load_context(self, *, user_id: str, conversation_id: str) -> MemoryContext:
        """Assemble everything the prompt needs from memory."""
        user_memory = await self._memory.get_user_memory(user_id)
        conversation_memory = await self._memory.get_conversation_memory(
            conversation_id, user_id
        )
        recent = await self._conversations.list_recent_messages(
            conversation_id, user_id, self._settings.memory_recent_messages
        )

        return MemoryContext(
            user_summary=user_memory.summary,
            conversation_summary=conversation_memory.summary,
            recent_messages=[
                ChatTurn(role=message.role, content=message.content) for message in recent
            ],
        )

    async def get_user_memory(self, user_id: str) -> UserMemory:
        return await self._memory.get_user_memory(user_id)

    async def get_conversation_memory(
        self, conversation_id: str, user_id: str
    ) -> ConversationMemory:
        return await self._memory.get_conversation_memory(conversation_id, user_id)

    # ======================================================================
    # Write path (graph: Update Memory)
    # ======================================================================

    async def update_after_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        question: str,
        answer: str,
    ) -> MemoryContext:
        """Refresh both summaries after a completed exchange.

        Runs at the end of a turn, after the answer has already been streamed to
        the client, so its latency is invisible to the user. Failures are
        swallowed: losing a summary update must never lose a saved answer.
        """
        if not self._settings.memory_summary_enabled:
            return await self.load_context(user_id=user_id, conversation_id=conversation_id)

        exchange = f"User: {question.strip()}\nAssistant: {answer.strip()}"

        conversation_summary = await self._update_conversation_summary(
            conversation_id=conversation_id, user_id=user_id, exchange=exchange
        )
        user_summary = await self._update_user_summary(user_id=user_id, exchange=exchange)

        return MemoryContext(
            user_summary=user_summary,
            conversation_summary=conversation_summary,
        )

    async def _update_conversation_summary(
        self, *, conversation_id: str, user_id: str, exchange: str
    ) -> str:
        existing = await self._memory.get_conversation_memory(conversation_id, user_id)

        try:
            prompt = CONVERSATION_SUMMARY_PROMPT.format(
                existing_summary=existing.summary or "(none yet)",
                new_messages=exchange,
                max_chars=self._settings.memory_summary_max_chars,
            )
            summary = await self._llm.summarize(
                prompt, max_chars=self._settings.memory_summary_max_chars
            )
        except Exception as exc:
            logger.warning("Conversation summary update failed for %s: %s", conversation_id, exc)
            return existing.summary

        if not summary:
            return existing.summary

        await self._memory.upsert_conversation_memory(
            conversation_id,
            user_id,
            summary=summary,
            message_count=existing.message_count + 2,
        )
        return summary

    async def _update_user_summary(self, *, user_id: str, exchange: str) -> str:
        existing = await self._memory.get_user_memory(user_id)
        turn_count = existing.turn_count + 1

        # Rebuilding the whole user profile every turn is wasteful and makes it
        # drift; refresh it every N turns instead.
        every_n = max(1, self._settings.memory_user_summary_every_n_turns)
        is_first = not existing.summary
        due = turn_count % every_n == 0

        if not (is_first or due):
            await self._memory.upsert_user_memory(
                user_id, summary=existing.summary, turn_count=turn_count
            )
            return existing.summary

        try:
            prompt = USER_SUMMARY_PROMPT.format(
                existing_summary=existing.summary or "(none yet)",
                new_messages=exchange,
                max_chars=self._settings.memory_summary_max_chars,
            )
            summary = await self._llm.summarize(
                prompt, max_chars=self._settings.memory_summary_max_chars
            )
        except Exception as exc:
            logger.warning("User summary update failed for %s: %s", user_id, exc)
            await self._memory.upsert_user_memory(
                user_id, summary=existing.summary, turn_count=turn_count
            )
            return existing.summary

        final = summary or existing.summary
        await self._memory.upsert_user_memory(
            user_id, summary=final, turn_count=turn_count
        )
        return final

    # ======================================================================
    # Manual management (/memory endpoints)
    # ======================================================================

    async def set_user_summary(self, user_id: str, summary: str) -> UserMemory:
        existing = await self._memory.get_user_memory(user_id)
        return await self._memory.upsert_user_memory(
            user_id,
            summary=summary.strip()[: self._settings.memory_summary_max_chars],
            turn_count=existing.turn_count,
        )

    async def set_conversation_summary(
        self, conversation_id: str, user_id: str, summary: str
    ) -> ConversationMemory:
        existing = await self._memory.get_conversation_memory(conversation_id, user_id)
        return await self._memory.upsert_conversation_memory(
            conversation_id,
            user_id,
            summary=summary.strip()[: self._settings.memory_summary_max_chars],
            message_count=existing.message_count,
        )

    async def clear_user_memory(self, user_id: str) -> None:
        await self._memory.delete_user_memory(user_id)
        logger.info("Cleared long-term memory for user %s", user_id)

    async def clear_conversation_memory(self, conversation_id: str, user_id: str) -> None:
        await self._memory.delete_conversation_memory(conversation_id, user_id)

    async def recent_messages(
        self, conversation_id: str, user_id: str, limit: int | None = None
    ) -> list[ChatTurn]:
        messages = await self._conversations.list_recent_messages(
            conversation_id, user_id, limit or self._settings.memory_recent_messages
        )
        return [
            ChatTurn(role=MessageRole(message.role), content=message.content)
            for message in messages
        ]
