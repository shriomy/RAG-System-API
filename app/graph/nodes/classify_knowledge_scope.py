"""Node: Classify Knowledge Scope.

Runs a cheap LLM classifier before retrieval. It decides whether the question
looks like it should consult the assistant's knowledge base or can skip the
RAG lookup path.

The node is intentionally fail-open: if the classifier errors or returns
something unparsable, the graph falls back to retrieval rather than blocking
the turn.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.ai.prompts import KNOWLEDGE_SCOPE_PROMPT
from app.core.errors import AppError
from app.core.logging import get_logger
from app.domain.models import Assistant
from app.graph.state import AgentState
from app.services.openrouter_service import OpenRouterService

logger = get_logger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            cleaned.strip(),
            flags=re.IGNORECASE | re.DOTALL,
        )

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("Classifier did not return JSON.")
        parsed = json.loads(match.group(0))

    if not isinstance(parsed, dict):
        raise ValueError("Classifier output must be a JSON object.")
    return parsed


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "in_scope", "retrieve"}
    return bool(value)


def _as_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def make_classify_knowledge_scope_node(llm: OpenRouterService):
    async def classify_knowledge_scope(state: AgentState) -> dict:
        assistant = Assistant.model_validate(state["assistant"])
        assistant_context = assistant.knowledge_scope_context() or assistant.system_prompt

        prompt = KNOWLEDGE_SCOPE_PROMPT.format(
            assistant_context=assistant_context or "(no assistant knowledge summary provided)",
            system_prompt=state.get("system_prompt", "") or assistant.system_prompt,
            question=state["question"],
        )

        fallback = {
            "in_scope": True,
            "confidence": 0.5,
            "reason": "Classifier unavailable; falling back to retrieval.",
            "source": "fallback",
        }

        try:
            raw = await llm.acomplete(
                [{"role": "user", "content": prompt}],
                llm.summary_spec(),
                run_name="knowledge_scope_classifier",
            )
            parsed = _extract_json(raw)
            decision = {
                "in_scope": _as_bool(parsed.get("in_scope", True)),
                "confidence": _as_confidence(parsed.get("confidence", 0.5)),
                "reason": str(parsed.get("reason", "")).strip() or "Classifier returned no reason.",
                "source": "llm",
            }
        except (AppError, Exception) as exc:
            logger.warning(
                "Knowledge-scope classification failed; falling back to retrieval: %s",
                exc,
            )
            decision = fallback | {"reason": str(exc)}

        logger.debug(
            "Knowledge scope classified %s (confidence=%.2f): %s",
            "in-scope" if decision["in_scope"] else "out-of-scope",
            decision["confidence"],
            decision["reason"],
        )

        metadata = {
            **(state.get("metadata") or {}),
            "knowledge_scope": decision,
        }
        return {"knowledge_scope": decision, "metadata": metadata}

    return classify_knowledge_scope