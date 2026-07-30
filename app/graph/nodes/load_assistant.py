"""Node: Load Assistant.

Fetches the assistant configuration — including which model to use — and puts it
in state. Everything downstream reads the model from here, which is what makes
model switching a database change.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.graph.state import AgentState
from app.services.assistant_service import AssistantService

logger = get_logger(__name__)


def make_load_assistant_node(assistant_service: AssistantService):
    async def load_assistant(state: AgentState) -> dict:
        assistant = await assistant_service.get_assistant(
            state["assistant_id"], state["user_id"]
        )

        logger.debug(
            "Loaded assistant %s (model=%s)", assistant.id, assistant.model
        )
        return {
            "assistant": assistant.model_dump(mode="json"),
            "system_prompt": assistant.system_prompt,
            "model": assistant.model,
        }

    return load_assistant
