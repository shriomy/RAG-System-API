"""AgentState — the shared, strongly-typed state passed between nodes.

Each node reads the keys it needs and returns only the keys it changed;
LangGraph merges those partial updates into the running state.

Everything held here is JSON-serialisable (retrieved docs and messages are
dicts, not domain objects), which is what will let a LangGraph checkpointer be
switched on later for durable execution and human-in-the-loop.

Fields are grouped as: identity, loaded context, generation, extension slots.
The extension slots exist so that adding tools, guardrails or citations does not
require touching this TypedDict again.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from app.domain.models import MemoryContext, RetrievedChunk


def _replace(_: Any, new: Any) -> Any:
    """Last write wins. Explicit so the merge behaviour is documented, not implied."""
    return new


def _append(existing: list[Any] | None, new: list[Any] | None) -> list[Any]:
    """Accumulate across nodes — used where several nodes contribute entries."""
    return [*(existing or []), *(new or [])]


class AgentState(TypedDict, total=False):
    """State for one agent run (one user turn)."""

    # -- identity (set by the caller, never mutated by nodes) ---------------
    user_id: str
    assistant_id: str
    conversation_id: str
    question: str

    # -- loaded context ----------------------------------------------------
    #: Serialised Assistant, written by Load Assistant.
    assistant: dict[str, Any]
    #: The assistant's configured system prompt.
    system_prompt: str
    #: Rolling long-term summary of the user.
    user_summary: str
    #: Rolling summary of this conversation.
    conversation_summary: str
    #: Verbatim recent turns, oldest first: [{"role", "content"}, ...].
    recent_messages: list[dict[str, str]]
    #: Serialised RetrievedChunks from the retrieval pipeline.
    retrieved_docs: list[dict[str, Any]]
    #: Scope classifier result; controls whether retrieval runs.
    knowledge_scope: Annotated[dict[str, Any], _replace]
    #: The assembled message list handed to the LLM.
    prompt_messages: list[dict[str, str]]

    # -- generation --------------------------------------------------------
    answer: str
    model: str
    usage: dict[str, int]

    # -- persistence results ----------------------------------------------
    user_message_id: str
    assistant_message_id: str

    # -- extension slots ---------------------------------------------------
    #: Source attributions derived from retrieved_docs.
    citations: list[dict[str, Any]]
    #: Reserved for the tool node.
    tool_calls: Annotated[list[dict[str, Any]], _append]
    #: Reserved for the guardrail pipeline.
    guardrail_events: Annotated[list[dict[str, Any]], _append]
    #: Free-form per-run annotations (timings, flags, trace ids).
    metadata: Annotated[dict[str, Any], _replace]
    #: Set when a node fails non-fatally; surfaced to the client.
    error: str | None


# ---------------------------------------------------------------------------
# Conversions between AgentState's plain dicts and domain objects.
# Keeping them here means nodes stay one-liners and no serialisation logic is
# duplicated across nodes.
# ---------------------------------------------------------------------------


def initial_state(
    *,
    user_id: str,
    assistant_id: str,
    conversation_id: str,
    question: str,
) -> AgentState:
    """Build the state the graph starts from."""
    return AgentState(
        user_id=user_id,
        assistant_id=assistant_id,
        conversation_id=conversation_id,
        question=question,
        system_prompt="",
        user_summary="",
        conversation_summary="",
        recent_messages=[],
        retrieved_docs=[],
        knowledge_scope={},
        prompt_messages=[],
        answer="",
        usage={},
        citations=[],
        tool_calls=[],
        guardrail_events=[],
        metadata={},
        error=None,
    )


def dump_chunks(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [chunk.model_dump(mode="json") for chunk in chunks]


def load_chunks(state: AgentState) -> list[RetrievedChunk]:
    return [
        RetrievedChunk.model_validate(item) for item in state.get("retrieved_docs") or []
    ]


def dump_memory(context: MemoryContext) -> dict[str, Any]:
    """Flatten a MemoryContext into the state keys the prompt node reads."""
    return {
        "user_summary": context.user_summary,
        "conversation_summary": context.conversation_summary,
        "recent_messages": [
            {"role": str(turn.role), "content": turn.content}
            for turn in context.recent_messages
        ],
    }
