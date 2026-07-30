"""ConversationService — conversations and message persistence."""

from __future__ import annotations

from app.ai.prompts import TITLE_PROMPT
from app.core.config import Settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.domain.models import Conversation, Message, MessageRole
from app.repositories.conversation_repository import ConversationRepository
from app.services.openrouter_service import OpenRouterService

logger = get_logger(__name__)

FALLBACK_TITLE_LENGTH = 48


class ConversationService:
    def __init__(
        self,
        *,
        repository: ConversationRepository,
        llm: OpenRouterService,
        settings: Settings,
    ) -> None:
        self._repo = repository
        self._llm = llm
        self._settings = settings

    # ======================================================================
    # Conversations
    # ======================================================================

    async def list_conversations(
        self,
        user_id: str,
        *,
        assistant_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Conversation]:
        return await self._repo.list_for_user(
            user_id, assistant_id=assistant_id, limit=limit, offset=offset
        )

    async def get_conversation(self, conversation_id: str, user_id: str) -> Conversation:
        conversation = await self._repo.get(conversation_id, user_id)
        if conversation is None:
            raise NotFoundError(
                "Conversation not found.", details={"conversation_id": conversation_id}
            )
        return conversation

    async def create_conversation(
        self,
        user_id: str,
        *,
        assistant_id: str | None,
        title: str | None = None,
        first_message: str | None = None,
    ) -> Conversation:
        resolved = title or self._fallback_title(first_message)
        return await self._repo.create(user_id, assistant_id, resolved)

    async def ensure_conversation(
        self,
        *,
        user_id: str,
        assistant_id: str,
        conversation_id: str | None,
        first_message: str,
    ) -> tuple[Conversation, bool]:
        """Resolve the conversation for a chat turn, creating one if needed.

        Returns (conversation, was_created) so the caller can tell the client
        about a new conversation id.
        """
        if conversation_id:
            conversation = await self.get_conversation(conversation_id, user_id)
            return conversation, False

        conversation = await self._repo.create(
            user_id, assistant_id, self._fallback_title(first_message)
        )
        logger.info("Created conversation %s for user %s", conversation.id, user_id)
        return conversation, True

    async def rename_conversation(
        self, conversation_id: str, user_id: str, title: str
    ) -> Conversation:
        cleaned = title.strip()
        if not cleaned:
            raise ValidationError("Title cannot be empty.")
        conversation = await self._repo.rename(conversation_id, user_id, cleaned)
        if conversation is None:
            raise NotFoundError(
                "Conversation not found.", details={"conversation_id": conversation_id}
            )
        return conversation

    async def delete_conversation(self, conversation_id: str, user_id: str) -> None:
        """Delete a conversation. Messages and memory cascade in the database."""
        deleted = await self._repo.delete(conversation_id, user_id)
        if not deleted:
            raise NotFoundError(
                "Conversation not found.", details={"conversation_id": conversation_id}
            )
        logger.info("Deleted conversation %s for user %s", conversation_id, user_id)

    async def generate_title(self, conversation_id: str, user_id: str, question: str) -> str:
        """Replace the placeholder title with an LLM-generated one.

        Best effort: a failure here leaves the truncated fallback in place.
        """
        try:
            title = await self._llm.generate_title(TITLE_PROMPT.format(question=question))
        except Exception as exc:
            logger.debug("Title generation failed for %s: %s", conversation_id, exc)
            return self._fallback_title(question)

        if not title:
            return self._fallback_title(question)

        await self._repo.rename(conversation_id, user_id, title)
        return title

    # ======================================================================
    # Messages
    # ======================================================================

    async def list_messages(
        self, conversation_id: str, user_id: str, *, limit: int | None = None
    ) -> list[Message]:
        # Confirms ownership before returning any message content.
        await self.get_conversation(conversation_id, user_id)
        return await self._repo.list_messages(conversation_id, user_id, limit=limit)

    async def add_user_message(
        self, conversation_id: str, user_id: str, content: str
    ) -> Message:
        return await self._repo.add_message(
            conversation_id, user_id, MessageRole.USER, content
        )

    async def add_assistant_message(
        self, conversation_id: str, user_id: str, content: str
    ) -> Message:
        return await self._repo.add_message(
            conversation_id, user_id, MessageRole.ASSISTANT, content
        )

    async def save_exchange(
        self,
        *,
        conversation_id: str,
        user_id: str,
        question: str,
        answer: str,
    ) -> tuple[Message, Message]:
        """Persist a completed turn (graph: Save Conversation)."""
        user_message = await self.add_user_message(conversation_id, user_id, question)
        assistant_message = await self.add_assistant_message(
            conversation_id, user_id, answer
        )
        return user_message, assistant_message

    async def count_messages(self, conversation_id: str, user_id: str) -> int:
        return await self._repo.count_messages(conversation_id, user_id)

    async def is_first_turn(self, conversation_id: str, user_id: str) -> bool:
        """True when nothing has been said yet — used to decide about titling."""
        return await self._repo.count_messages(conversation_id, user_id) == 0

    # ======================================================================
    # Internals
    # ======================================================================

    @staticmethod
    def _fallback_title(text: str | None) -> str:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return "New conversation"
        if len(cleaned) <= FALLBACK_TITLE_LENGTH:
            return cleaned
        return cleaned[:FALLBACK_TITLE_LENGTH].rstrip() + "…"
