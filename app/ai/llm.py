"""LangChain chat models, pointed at OpenRouter.

OpenRouter exposes an OpenAI-compatible API, so `ChatOpenAI` with a custom
`base_url` is all that is required. Nothing here names a specific model: the
slug always comes from a `ModelSpec` built out of the assistant row, which is
what makes model switching a data change.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import ModelSpec

logger = get_logger(__name__)

_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
}


def openrouter_headers(settings: Settings) -> dict[str, str]:
    """Attribution headers OpenRouter surfaces in its dashboard."""
    headers: dict[str, str] = {}
    if settings.openrouter_app_url:
        headers["HTTP-Referer"] = settings.openrouter_app_url
    if settings.openrouter_app_title:
        headers["X-Title"] = settings.openrouter_app_title
    return headers


def build_chat_model(
    settings: Settings,
    spec: ModelSpec,
    *,
    streaming: bool = False,
    tools: Sequence[Any] | None = None,
) -> BaseChatModel:
    """Build a chat model for one call.

    Cheap to construct (it is a thin config wrapper around an HTTP client), so
    it is built per request rather than cached — that keeps per-assistant model
    settings correct with no invalidation logic.
    """
    model: BaseChatModel = ChatOpenAI(
        model=spec.model,
        api_key=settings.openrouter_api_key,  # type: ignore[arg-type]
        base_url=settings.openrouter_base_url,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        streaming=streaming,
        stop=spec.stop or None,
        default_headers=openrouter_headers(settings),
        model_kwargs={"extra_body": spec.extra_body} if spec.extra_body else {},
    )

    # Tool binding is here so that enabling tools/MCP later needs no change in
    # the graph node — it just starts passing a non-empty `tools`.
    if tools:
        model = model.bind_tools(list(tools))  # type: ignore[assignment]
    return model


def to_lc_messages(messages: Sequence[dict[str, str]]) -> list[BaseMessage]:
    """Convert our plain role/content dicts into LangChain messages."""
    converted: list[BaseMessage] = []
    for message in messages:
        role = (message.get("role") or "user").lower()
        content = message.get("content") or ""
        message_cls = _ROLE_TO_MESSAGE.get(role, HumanMessage)
        converted.append(message_cls(content=content))
    return converted


def extract_usage(message: Any) -> dict[str, int]:
    """Pull token usage out of a LangChain response, tolerant of shape drift."""
    usage = getattr(message, "usage_metadata", None) or {}
    if not usage:
        metadata = getattr(message, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or metadata.get("usage") or {}

    return {
        "prompt_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
        "completion_tokens": int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        ),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }
