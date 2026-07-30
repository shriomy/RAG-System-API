"""Node: Update Memory — the final node.

Refreshes the rolling conversation summary and (periodically) the long-term user
summary. It runs after Save Conversation and after the answer has been streamed,
so the extra LLM call costs the user no perceived latency.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.graph.state import AgentState
from app.services.memory_service import MemoryService

logger = get_logger(__name__)


def make_update_memory_node(memory_service: MemoryService):
    async def update_memory(state: AgentState) -> dict:
        answer = state.get("answer") or ""
        if not answer:
            return {}

        try:
            context = await memory_service.update_after_turn(
                user_id=state["user_id"],
                conversation_id=state["conversation_id"],
                question=state["question"],
                answer=answer,
            )
        except Exception as exc:
            # Memory is an enhancement; a failure here must not fail the turn.
            logger.warning("Memory update failed: %s", exc)
            return {}

        logger.debug(
            "Updated memory: user_summary=%d chars, conversation_summary=%d chars",
            len(context.user_summary),
            len(context.conversation_summary),
        )
        return {
            "user_summary": context.user_summary,
            "conversation_summary": context.conversation_summary,
        }

    return update_memory
