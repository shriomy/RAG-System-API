"""Node: Build Prompt.

Combines, in order: the assistant's system prompt, the long-term user summary,
the conversation summary, the recent messages, the retrieved chunks, and the
current question.

The assembly itself lives in app/ai/prompts.py (LangChain PromptTemplates), so
this node only marshals state in and out.
"""

from __future__ import annotations

from app.ai.prompts import build_chat_prompt
from app.core.logging import get_logger
from app.domain.models import ChatTurn
from app.graph.state import AgentState, load_chunks

logger = get_logger(__name__)


def make_build_prompt_node():
    async def build_prompt(state: AgentState) -> dict:
        recent = [
            ChatTurn.model_validate(turn) for turn in state.get("recent_messages") or []
        ]

        messages = build_chat_prompt(
            system_prompt=state.get("system_prompt", ""),
            question=state["question"],
            retrieved_chunks=load_chunks(state),
            user_summary=state.get("user_summary", ""),
            conversation_summary=state.get("conversation_summary", ""),
            recent_messages=recent,
            knowledge_scope=state.get("knowledge_scope") or {},
        )

        logger.debug(
            "Built prompt: %d message(s), %d chars total",
            len(messages),
            sum(len(m["content"]) for m in messages),
        )
        return {"prompt_messages": messages}

    return build_prompt
