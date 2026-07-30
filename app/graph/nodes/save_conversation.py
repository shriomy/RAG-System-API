"""Node: Save Conversation.

Persists the completed exchange. Both messages are written here, after
generation, rather than the question being written up-front — that way a failed
turn does not leave an orphaned user message in the thread, and the memory node
that follows sees a consistent history.
"""

from __future__ import annotations

from app.core.errors import AppError
from app.core.logging import get_logger
from app.graph.state import AgentState
from app.services.conversation_service import ConversationService

logger = get_logger(__name__)


def make_save_conversation_node(conversation_service: ConversationService):
    async def save_conversation(state: AgentState) -> dict:
        try:
            user_message, assistant_message = await conversation_service.save_exchange(
                conversation_id=state["conversation_id"],
                user_id=state["user_id"],
                question=state["question"],
                answer=state["answer"],
            )
        except AppError as exc:
            # The answer was already streamed to the client; report the failure
            # in state rather than discarding a successful generation.
            logger.error("Could not save exchange: %s", exc.message)
            return {"error": f"The reply could not be saved: {exc.message}"}

        logger.debug(
            "Saved exchange in conversation %s (%s, %s)",
            state["conversation_id"],
            user_message.id,
            assistant_message.id,
        )
        return {
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
        }

    return save_conversation
