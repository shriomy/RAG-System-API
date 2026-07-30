"""Guardrail registry.

A guardrail is any object implementing the `Guardrail` port. Register a factory
under a name, list that name in GUARDRAILS, and the pipeline picks it up — no
change to the graph, the chat service or the routes.

Two examples ship as reference implementations; neither is enabled by default.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import GuardrailResult
from app.domain.ports import Guardrail

logger = get_logger(__name__)

GuardrailFactory = Callable[[Settings], Guardrail]

GUARDRAIL_REGISTRY: dict[str, GuardrailFactory] = {}


def register_guardrail(name: str) -> Callable[[GuardrailFactory], GuardrailFactory]:
    def decorator(factory: GuardrailFactory) -> GuardrailFactory:
        GUARDRAIL_REGISTRY[name] = factory
        return factory

    return decorator


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


class PIIRedactionGuardrail:
    """Masks obvious PII in model output rather than blocking it."""

    @property
    def name(self) -> str:
        return "pii"

    async def check_input(self, text: str, *, context: dict[str, Any]) -> GuardrailResult:
        return GuardrailResult(allowed=True, guardrail=self.name)

    async def check_output(self, text: str, *, context: dict[str, Any]) -> GuardrailResult:
        redacted = text
        hits: list[str] = []
        for label, pattern in _PII_PATTERNS.items():
            redacted, count = pattern.subn(f"[redacted:{label}]", redacted)
            if count:
                hits.append(label)

        if not hits:
            return GuardrailResult(allowed=True, guardrail=self.name)
        return GuardrailResult(
            allowed=True,
            guardrail=self.name,
            reason=f"Redacted {', '.join(hits)}",
            replacement=redacted,
        )


class InputLengthGuardrail:
    """Rejects absurdly long questions before they reach the model."""

    def __init__(self, max_chars: int = 8000) -> None:
        self._max_chars = max_chars

    @property
    def name(self) -> str:
        return "input_length"

    async def check_input(self, text: str, *, context: dict[str, Any]) -> GuardrailResult:
        if len(text) <= self._max_chars:
            return GuardrailResult(allowed=True, guardrail=self.name)
        return GuardrailResult(
            allowed=False,
            guardrail=self.name,
            reason=f"Question exceeds {self._max_chars} characters.",
        )

    async def check_output(self, text: str, *, context: dict[str, Any]) -> GuardrailResult:
        return GuardrailResult(allowed=True, guardrail=self.name)


@register_guardrail("pii")
def _build_pii(_: Settings) -> Guardrail:
    return PIIRedactionGuardrail()


@register_guardrail("input_length")
def _build_input_length(_: Settings) -> Guardrail:
    return InputLengthGuardrail()
