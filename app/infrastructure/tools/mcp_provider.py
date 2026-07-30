"""MCP tool provider.

Inactive until MCP_ENABLED=true. The provider exists now so the wiring is
already in place: ToolResolver asks every provider for tools, the graph's tool
node consumes whatever comes back, and the LLM node binds them. Switching MCP
on therefore adds servers, not architecture.

Configure servers as JSON in MCP_SERVERS:

    {
      "filesystem": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]
      },
      "internal-api": {
        "transport": "streamable_http",
        "url": "https://mcp.internal/mcp"
      }
    }

Then per assistant, `config.mcp_servers: ["filesystem"]` selects which of them
that assistant may use.
"""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger
from app.domain.models import Assistant

logger = get_logger(__name__)


class MCPToolProvider:
    """Loads tools from configured MCP servers via langchain-mcp-adapters."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: Any | None = None
        self._cache: dict[str, list[Any]] = {}

    @property
    def name(self) -> str:
        return "mcp"

    async def _ensure_client(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError:
            logger.error(
                "MCP_ENABLED=true but langchain-mcp-adapters is not installed. "
                "Install it with: pip install langchain-mcp-adapters"
            )
            return None

        self._client = MultiServerMCPClient(self._settings.mcp_servers)
        logger.info("MCP client ready (servers: %s)", sorted(self._settings.mcp_servers))
        return self._client

    async def get_tools(self, assistant: Assistant) -> list[Any]:
        if not self._settings.mcp_enabled or not self._settings.mcp_servers:
            return []

        client = await self._ensure_client()
        if client is None:
            return []

        # An empty per-assistant list means "every configured server".
        wanted = assistant.mcp_server_names or list(self._settings.mcp_servers)

        tools: list[Any] = []
        for server in wanted:
            if server not in self._settings.mcp_servers:
                logger.warning("Assistant references unknown MCP server '%s'", server)
                continue
            if server in self._cache:
                tools.extend(self._cache[server])
                continue
            try:
                server_tools = await client.get_tools(server_name=server)
            except Exception as exc:
                logger.exception("Could not load tools from MCP server '%s': %s", server, exc)
                continue
            self._cache[server] = server_tools
            tools.extend(server_tools)

        return tools

    async def close(self) -> None:
        self._cache.clear()
        self._client = None
