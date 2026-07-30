"""Node: LLM.

Calls OpenRouter with the assembled prompt and the assistant's own ModelSpec.

Streaming: tokens are forwarded through an `on_token` async callback passed in
via the run config, rather than by relying on a specific LangGraph stream mode.
That keeps the transport decision inside ChatService and means the node behaves
identically for streaming and non-streaming callers.

Extension points already wired here:
  * `tools` — ToolResolver returns [] today; when tools or MCP are switched on,
    they are bound to the model with no change to this file.
  * output guardrails — the pipeline is empty today; when enabled it can rewrite
    the answer before it is persisted.
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig

from app.core.logging import get_logger
from app.domain.models import Assistant
from app.graph.state import AgentState
from app.infrastructure.guardrails.pipeline import GuardrailPipeline
from app.infrastructure.tools.registry import ToolResolver
from app.services.openrouter_service import OpenRouterService

logger = get_logger(__name__)


def make_llm_node(
    llm: OpenRouterService,
    *,
    tool_resolver: ToolResolver,
    guardrails: GuardrailPipeline,
):
    async def call_llm(state: AgentState, config: RunnableConfig) -> dict:
        assistant = Assistant.model_validate(state["assistant"])
        spec = llm.resolve_spec(assistant)

        configurable = config.get("configurable") or {}
        on_token = configurable.get("on_token")

        tools = await tool_resolver.get_tools(assistant)

        usage_sink: dict[str, Any] = {}
        parts: list[str] = []

        stream = llm.astream(
            state["prompt_messages"],
            spec,
            tools=tools or None,
            run_name="agent_llm",
            metadata={
                "assistant_id": state["assistant_id"],
                "conversation_id": state["conversation_id"],
                "retrieved_docs": len(state.get("retrieved_docs") or []),
            },
            usage_sink=usage_sink,
        )

        async for token in stream:
            parts.append(token)
            if on_token is not None:
                await on_token(token)

        answer = "".join(parts).strip()

        guardrail_events: list[dict[str, Any]] = []
        if guardrails.enabled and answer:
            checked, events = await guardrails.check_output(
                answer,
                context={
                    "user_id": state["user_id"],
                    "assistant_id": state["assistant_id"],
                },
            )
            guardrail_events = [event.model_dump(mode="json") for event in events]
            if checked != answer:
                # The raw text has already been streamed; ChatService emits a
                # `revision` event so the client can replace what it rendered.
                answer = checked
                if on_token is not None:
                    revise = configurable.get("on_revision")
                    if revise is not None:
                        await revise(answer)

        if not answer:
            answer = "I wasn't able to generate a response. Please try again."
            logger.warning(
                "Empty completion from model %s (assistant %s)", spec.model, assistant.id
            )

        logger.debug(
            "LLM produced %d chars via %s (tools=%d)", len(answer), spec.model, len(tools)
        )
        return {
            "answer": answer,
            "model": spec.model,
            "usage": usage_sink,
            "guardrail_events": guardrail_events,
        }

    return call_llm
