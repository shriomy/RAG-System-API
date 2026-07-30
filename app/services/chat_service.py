"""ChatService — the entry point for a chat turn.

Responsibilities, in order:

    1. validate the question and run input guardrails
    2. resolve (or create) the conversation
    3. build the initial AgentState and run the graph
    4. translate graph events into transport-neutral chat events

It deliberately does NOT know about SSE — the route formats events for the wire.
That keeps a future WebSocket transport a routing change only.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from app.core.config import Settings
from app.core.errors import AppError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.domain.models import ChatResult, TokenUsage
from app.graph.state import AgentState
from app.infrastructure.guardrails.pipeline import GuardrailPipeline
from app.services.assistant_service import AssistantService
from app.services.conversation_service import ConversationService
from app.services.graph_service import GraphService

logger = get_logger(__name__)

MAX_QUESTION_CHARS = 32_000


class ChatService:
    def __init__(
        self,
        *,
        graph_service: GraphService,
        conversation_service: ConversationService,
        assistant_service: AssistantService,
        guardrails: GuardrailPipeline,
        settings: Settings,
    ) -> None:
        self._graph = graph_service
        self._conversations = conversation_service
        self._assistants = assistant_service
        self._guardrails = guardrails
        self._settings = settings

    # ======================================================================
    # Streaming turn
    # ======================================================================

    async def stream_chat(
        self,
        *,
        user_id: str,
        assistant_id: str | None,
        conversation_id: str | None,
        question: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one turn, yielding chat events.

        Event shapes:
            {"type": "start",    "conversation_id", "assistant_id", "model", "is_new_conversation"}
            {"type": "token",    "content"}
            {"type": "revision", "content"}      guardrail rewrote the answer
            {"type": "done",     ...ChatResult, "title"?}
            {"type": "error",    "message", "code"}
        """
        try:
            prepared = await self._prepare(
                user_id=user_id,
                assistant_id=assistant_id,
                conversation_id=conversation_id,
                question=question,
            )
        except AppError as exc:
            yield {"type": "error", "code": exc.code, "message": exc.message}
            return

        state: AgentState = prepared["state"]
        yield {
            "type": "start",
            "conversation_id": state["conversation_id"],
            "assistant_id": state["assistant_id"],
            "model": prepared["model"],
            "is_new_conversation": prepared["is_new_conversation"],
        }

        final_state: AgentState | None = None
        try:
            async for kind, payload in self._graph.stream(state):
                if kind == "token":
                    yield {"type": "token", "content": payload}
                elif kind == "revision":
                    yield {"type": "revision", "content": payload}
                elif kind == "result":
                    final_state = payload
        except AppError as exc:
            logger.info("Chat turn failed: %s", exc.message)
            yield {"type": "error", "code": exc.code, "message": exc.message}
            return
        except Exception as exc:
            logger.exception("Chat turn crashed")
            yield {
                "type": "error",
                "code": "internal_error",
                "message": f"The request failed: {exc}",
            }
            return

        if final_state is None:
            yield {
                "type": "error",
                "code": "internal_error",
                "message": "The agent produced no result.",
            }
            return

        result = self._to_result(final_state)
        done: dict[str, Any] = {"type": "done", **result.model_dump(mode="json")}

        # Title the conversation from its first question, once the answer is out.
        if prepared["needs_title"]:
            done["title"] = await self._conversations.generate_title(
                state["conversation_id"], user_id, state["question"]
            )

        if final_state.get("error"):
            done["warning"] = final_state["error"]

        yield done

    # ======================================================================
    # Non-streaming turn
    # ======================================================================

    async def chat(
        self,
        *,
        user_id: str,
        assistant_id: str | None,
        conversation_id: str | None,
        question: str,
    ) -> ChatResult:
        """Run one turn and return the complete result. Useful for testing and
        for clients that do not want a stream."""
        prepared = await self._prepare(
            user_id=user_id,
            assistant_id=assistant_id,
            conversation_id=conversation_id,
            question=question,
        )

        final_state = await self._graph.invoke(prepared["state"])
        result = self._to_result(final_state)

        if prepared["needs_title"]:
            await self._conversations.generate_title(
                result.conversation_id, user_id, prepared["state"]["question"]
            )
        return result

    # ======================================================================
    # Internals
    # ======================================================================

    async def _prepare(
        self,
        *,
        user_id: str,
        assistant_id: str | None,
        conversation_id: str | None,
        question: str,
    ) -> dict[str, Any]:
        """Validate, guard, resolve the assistant and conversation, build state."""
        cleaned = question.strip()
        if not cleaned:
            raise ValidationError("The question cannot be empty.")
        if len(cleaned) > MAX_QUESTION_CHARS:
            raise ValidationError(
                f"The question exceeds {MAX_QUESTION_CHARS} characters."
            )

        # Input guardrails run before anything is persisted or billed.
        if self._guardrails.enabled:
            cleaned, _ = await self._guardrails.check_input(
                cleaned, context={"user_id": user_id, "assistant_id": assistant_id}
            )

        assistant = await self._resolve_assistant(user_id, assistant_id)

        conversation, created = await self._conversations.ensure_conversation(
            user_id=user_id,
            assistant_id=assistant.id,
            conversation_id=conversation_id,
            first_message=cleaned,
        )

        # A conversation with no messages yet gets an LLM-generated title after
        # the answer is delivered.
        needs_title = created or await self._conversations.is_first_turn(
            conversation.id, user_id
        )

        state = self._graph.new_state(
            user_id=user_id,
            assistant_id=assistant.id,
            conversation_id=conversation.id,
            question=cleaned,
        )

        return {
            "state": state,
            "model": assistant.model,
            "is_new_conversation": created,
            "needs_title": needs_title,
        }

    async def _resolve_assistant(self, user_id: str, assistant_id: str | None):
        """Use the requested assistant, else the active one, else fail clearly."""
        if assistant_id:
            return await self._assistants.get_assistant(assistant_id, user_id)

        active = await self._assistants.get_active_assistant(user_id)
        if active:
            return active

        existing = await self._assistants.list_assistants(user_id)
        if existing:
            return existing[0]

        raise NotFoundError(
            "No assistant available. Create one before starting a chat."
        )

    @staticmethod
    def _to_result(state: AgentState) -> ChatResult:
        usage = state.get("usage") or {}
        return ChatResult(
            answer=state.get("answer", ""),
            conversation_id=state["conversation_id"],
            user_message_id=state.get("user_message_id"),
            assistant_message_id=state.get("assistant_message_id"),
            citations=state.get("citations") or [],
            usage=TokenUsage(**usage) if usage else None,
            model=state.get("model"),
        )
