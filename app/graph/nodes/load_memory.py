"""Node: Load User Memory.

Loads the long-term user summary, the conversation summary, and the verbatim
recent messages. When semantic memory is added later this node's body does not
change — MemoryService.load_context returns a richer MemoryContext and the
prompt builder consumes the extra field.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.graph.state import AgentState, dump_memory
from app.services.memory_service import MemoryService

logger = get_logger(__name__)


def make_load_memory_node(memory_service: MemoryService):
    async def load_user_memory(state: AgentState) -> dict:
        context = await memory_service.load_context(
            user_id=state["user_id"],
            conversation_id=state["conversation_id"],
        )

        logger.debug(
            "Loaded memory: user_summary=%d chars, conversation_summary=%d chars, "
            "%d recent message(s)",
            len(context.user_summary),
            len(context.conversation_summary),
            len(context.recent_messages),
        )
        return dump_memory(context)

    return load_user_memory
