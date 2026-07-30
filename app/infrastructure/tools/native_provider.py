"""Native (in-process) tool provider.

Tools registered in TOOL_REGISTRY are filtered twice: by the global
ENABLED_TOOLS allow-list, then by the assistant's own `config.tools` list. That
means one deployment can expose different tools per assistant with no code
change.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import Assistant
from app.infrastructure.tools.registry import TOOL_REGISTRY

logger = get_logger(__name__)


class NativeToolProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def name(self) -> str:
        return "native"

    async def get_tools(self, assistant: Assistant) -> list[Any]:
        if not self._settings.tools_enabled:
            return []

        allowed = set(self._settings.enabled_tools)
        requested = set(assistant.enabled_tools) or allowed
        selected = sorted(allowed & requested) if allowed else sorted(requested)

        tools: list[Any] = []
        for name in selected:
            factory = TOOL_REGISTRY.get(name)
            if factory is None:
                logger.warning("Unknown tool '%s'; known: %s", name, sorted(TOOL_REGISTRY))
                continue
            try:
                tools.append(factory(self._settings))
            except Exception as exc:
                logger.exception("Tool '%s' failed to build: %s", name, exc)

        return tools
