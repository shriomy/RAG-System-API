"""Chat DTOs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    """A chat turn.

    `assistant_id` may be omitted — the backend then falls back to the user's
    active assistant. `conversation_id` may be omitted to start a new
    conversation; the id of the created one comes back in the `start` event.
    """

    question: str = Field(min_length=1, max_length=32_000)
    assistant_id: str | None = None
    conversation_id: str | None = None


class TokenUsageResponse(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Citation(BaseModel):
    id: str
    source: str
    filename: str | None = None
    file_id: str | None = None
    chunk_index: int | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    """Non-streaming chat result (POST /chat/sync)."""

    model_config = ConfigDict(protected_namespaces=())

    answer: str
    conversation_id: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    usage: TokenUsageResponse | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Streaming events. Documented as models even though they go out as SSE `data:`
# payloads, so the contract is visible in the OpenAPI schema.
# ---------------------------------------------------------------------------


class StartEvent(BaseModel):
    type: Literal["start"] = "start"
    conversation_id: str
    assistant_id: str
    model: str
    is_new_conversation: bool = False


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    content: str


class RevisionEvent(BaseModel):
    """A guardrail rewrote the answer; replace what was rendered from tokens."""

    type: Literal["revision"] = "revision"
    content: str


class DoneEvent(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    type: Literal["done"] = "done"
    answer: str
    conversation_id: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    usage: TokenUsageResponse | None = None
    model: str | None = None
    #: Present when the conversation was auto-titled on its first turn.
    title: str | None = None
    #: Non-fatal problem, e.g. the answer streamed but could not be saved.
    warning: str | None = None


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


ChatStreamEvent = StartEvent | TokenEvent | RevisionEvent | DoneEvent | ErrorEvent


def describe_stream_protocol() -> dict[str, Any]:
    """Used in the OpenAPI description for the streaming endpoint."""
    return {
        "content_type": "text/event-stream",
        "events": ["start", "token", "revision", "done", "error"],
        "terminator": "[DONE]",
    }
