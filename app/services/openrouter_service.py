"""OpenRouterService — the LLM gateway.

Implements the `LLMProvider` port. Every model parameter comes from a
`ModelSpec` derived from the assistant row, so:

    * changing an assistant's model is an UPDATE on `assistants.model`
    * changing provider routing / reasoning effort / fallbacks is a JSON edit in
      `assistants.config.llm.extra_body`

Neither requires a deploy. No model name is hard-coded anywhere except the
`OPENROUTER_DEFAULT_MODEL` fallback in settings.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from app.ai.llm import build_chat_model, extract_usage, to_lc_messages
from app.ai.parsers import summary_parser, title_parser
from app.core.config import Settings
from app.core.errors import UpstreamError
from app.core.logging import get_logger
from app.core.observability import run_metadata, tracing_callbacks
from app.domain.models import Assistant, ModelSpec

logger = get_logger(__name__)


class OpenRouterService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    # ======================================================================
    # Model resolution
    # ======================================================================

    def resolve_spec(self, assistant: Assistant | None) -> ModelSpec:
        """Build a ModelSpec from an assistant, falling back to global defaults."""
        if assistant is None:
            return ModelSpec(
                model=self._settings.openrouter_default_model,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        return assistant.to_model_spec(default_model=self._settings.openrouter_default_model)

    def summary_spec(self) -> ModelSpec:
        """A cheap, deterministic spec for memory maintenance."""
        return ModelSpec(
            model=self._settings.summary_model,
            temperature=0.2,
            max_tokens=800,
        )

    # ======================================================================
    # Generation
    # ======================================================================

    async def astream(
        self,
        messages: Sequence[dict[str, str]],
        spec: ModelSpec,
        *,
        tools: Sequence[Any] | None = None,
        run_name: str = "chat",
        metadata: dict[str, Any] | None = None,
        usage_sink: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream response text chunk by chunk.

        Yields only content deltas. Token usage cannot be returned by a
        generator, and this service is a singleton shared by concurrent
        requests, so usage is written into the caller-owned `usage_sink` dict
        rather than onto `self` — no cross-request interference.
        """
        model = build_chat_model(self._settings, spec, streaming=True, tools=tools)

        try:
            stream = model.astream(
                to_lc_messages(messages),
                config={
                    "run_name": run_name,
                    "callbacks": tracing_callbacks(),
                    "metadata": run_metadata(model=spec.model, **(metadata or {})),
                },
            )
            async for chunk in stream:
                if usage_sink is not None:
                    usage = extract_usage(chunk)
                    if usage["total_tokens"] or usage["completion_tokens"]:
                        usage_sink.update(usage)

                content = chunk.content
                if isinstance(content, str) and content:
                    yield content
                elif isinstance(content, list):
                    # Some providers emit content blocks rather than plain text.
                    for block in content:
                        if isinstance(block, str) and block:
                            yield block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text") or ""
                            if text:
                                yield text
        except Exception as exc:
            raise UpstreamError(self._describe(exc, spec)) from exc

    async def acomplete(
        self,
        messages: Sequence[dict[str, str]],
        spec: ModelSpec,
        *,
        run_name: str = "completion",
        metadata: dict[str, Any] | None = None,
        usage_sink: dict[str, Any] | None = None,
    ) -> str:
        """Single non-streaming completion."""
        model = build_chat_model(self._settings, spec, streaming=False)

        try:
            response = await model.ainvoke(
                to_lc_messages(messages),
                config={
                    "run_name": run_name,
                    "callbacks": tracing_callbacks(),
                    "metadata": run_metadata(model=spec.model, **(metadata or {})),
                },
            )
        except Exception as exc:
            raise UpstreamError(self._describe(exc, spec)) from exc

        if usage_sink is not None:
            usage_sink.update(extract_usage(response))

        content = response.content
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "".join(parts).strip()
        return str(content).strip()

    # ======================================================================
    # Parsed helpers (used by MemoryService / ConversationService)
    # ======================================================================

    async def summarize(self, prompt: str, *, max_chars: int) -> str:
        """Run a summarisation prompt and clean the result."""
        raw = await self.acomplete(
            [{"role": "user", "content": prompt}],
            self.summary_spec(),
            run_name="summarize",
        )
        return summary_parser(max_chars).parse(raw)

    async def generate_title(self, prompt: str) -> str:
        """Run a titling prompt and clean the result."""
        raw = await self.acomplete(
            [{"role": "user", "content": prompt}],
            self.summary_spec(),
            run_name="title",
        )
        return title_parser().parse(raw)

    # ======================================================================
    # Internals
    # ======================================================================

    def _describe(self, exc: Exception, spec: ModelSpec) -> str:
        """Turn provider errors into something a user can act on."""
        text = str(exc)
        lowered = text.lower()

        if "401" in text or "invalid api key" in lowered or "no auth" in lowered:
            return "OpenRouter rejected the API key. Check OPENROUTER_API_KEY."
        if "402" in text or "credit" in lowered or "insufficient" in lowered:
            return "OpenRouter reports insufficient credits for this request."
        if "404" in text or "not a valid model" in lowered or "no endpoints found" in lowered:
            return (
                f"OpenRouter does not recognise the model '{spec.model}'. "
                f"Update the assistant's model to a valid slug."
            )
        if "429" in text or "rate limit" in lowered:
            return "OpenRouter rate limit reached. Please retry shortly."
        if "timeout" in lowered or "timed out" in lowered:
            return f"The model '{spec.model}' did not respond in time."

        logger.exception("OpenRouter call failed (model=%s)", spec.model)
        return f"The language model request failed: {text[:200]}"
