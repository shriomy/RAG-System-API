"""Tool registry and resolver.

`ToolResolver` aggregates every ToolProvider (native, MCP, and anything added
later) behind one call. The graph asks the resolver, so the number and kind of
providers is invisible to it.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import Assistant
from app.domain.ports import ToolProvider

logger = get_logger(__name__)

ToolFactory = Callable[[Settings], Any]

#: name -> factory returning a LangChain BaseTool.
#: Register with @register_tool("name"); no other file needs editing.
TOOL_REGISTRY: dict[str, ToolFactory] = {}


def register_tool(name: str) -> Callable[[ToolFactory], ToolFactory]:
    def decorator(factory: ToolFactory) -> ToolFactory:
        TOOL_REGISTRY[name] = factory
        return factory

    return decorator


class ToolResolver:
    """Single entry point for "what tools may this assistant use?"."""

    def __init__(self, providers: Sequence[ToolProvider] = ()) -> None:
        self._providers = list(providers)

    @property
    def enabled(self) -> bool:
        return bool(self._providers)

    async def get_tools(self, assistant: Assistant) -> list[Any]:
        if not self._providers:
            return []

        tools: list[Any] = []
        for provider in self._providers:
            try:
                tools.extend(await provider.get_tools(assistant))
            except Exception as exc:
                logger.exception("Tool provider '%s' failed: %s", provider.name, exc)

        if tools:
            logger.debug(
                "Resolved %d tool(s) for assistant %s", len(tools), assistant.id
            )
        return tools


def build_tool_resolver(settings: Settings) -> ToolResolver:
    from app.infrastructure.tools.mcp_provider import MCPToolProvider
    from app.infrastructure.tools.native_provider import NativeToolProvider

    providers: list[ToolProvider] = []
    if settings.tools_enabled:
        providers.append(NativeToolProvider(settings))
    if settings.mcp_enabled:
        providers.append(MCPToolProvider(settings))

    if providers:
        logger.info("Tool providers: %s", [p.name for p in providers])
    else:
        logger.info("Tool providers: none (tools and MCP disabled)")
    return ToolResolver(providers)
