"""Guardrail pipeline.

Always present in the request path, empty by default. Because ChatService and
the graph already call `check_input` / `check_output` on every turn, switching
guardrails on is purely configuration:

    GUARDRAILS_ENABLED=true
    GUARDRAILS=input_length,pii
"""

from __future__ import annotations

from typing import Any, Sequence

from app.core.config import Settings
from app.core.errors import GuardrailViolation
from app.core.logging import get_logger
from app.domain.models import GuardrailResult
from app.domain.ports import Guardrail
from app.infrastructure.guardrails.registry import GUARDRAIL_REGISTRY

logger = get_logger(__name__)


class GuardrailPipeline:
    """Runs guardrails in order; first block wins."""

    def __init__(self, guardrails: Sequence[Guardrail] = ()) -> None:
        self._guardrails = list(guardrails)

    @property
    def enabled(self) -> bool:
        return bool(self._guardrails)

    @property
    def names(self) -> list[str]:
        return [g.name for g in self._guardrails]

    async def check_input(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> tuple[str, list[GuardrailResult]]:
        """Validate (and possibly rewrite) user input.

        Raises GuardrailViolation when a guardrail blocks. Returns the
        possibly-rewritten text plus every non-clean result, for the trace.
        """
        return await self._run(text, context or {}, phase="input")

    async def check_output(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> tuple[str, list[GuardrailResult]]:
        """Validate (and possibly rewrite) model output."""
        return await self._run(text, context or {}, phase="output")

    async def _run(
        self, text: str, context: dict[str, Any], *, phase: str
    ) -> tuple[str, list[GuardrailResult]]:
        current = text
        events: list[GuardrailResult] = []

        for guardrail in self._guardrails:
            checker = guardrail.check_input if phase == "input" else guardrail.check_output
            try:
                result = await checker(current, context=context)
            except Exception as exc:
                # A broken guardrail must not take down the request; log loudly.
                logger.exception("Guardrail '%s' raised during %s: %s", guardrail.name, phase, exc)
                continue

            if not result.allowed:
                logger.info("Guardrail '%s' blocked %s: %s", guardrail.name, phase, result.reason)
                raise GuardrailViolation(
                    result.reason or f"Blocked by guardrail '{guardrail.name}'.",
                    details={"guardrail": guardrail.name, "phase": phase},
                )

            if result.replacement is not None and result.replacement != current:
                current = result.replacement
            if result.reason:
                events.append(result)

        return current, events


def build_guardrail_pipeline(settings: Settings) -> GuardrailPipeline:
    if not settings.guardrails_enabled or not settings.guardrails:
        logger.info("Guardrails: disabled")
        return GuardrailPipeline()

    guardrails: list[Guardrail] = []
    for name in settings.guardrails:
        factory = GUARDRAIL_REGISTRY.get(name)
        if factory is None:
            logger.warning(
                "Unknown guardrail '%s'; known: %s", name, sorted(GUARDRAIL_REGISTRY)
            )
            continue
        guardrails.append(factory(settings))

    logger.info("Guardrails: %s", [g.name for g in guardrails] or "none resolved")
    return GuardrailPipeline(guardrails)
