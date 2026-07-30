"""Ports — the extension seams of the system.

Each protocol here is a place where a feature can be added later without
touching the graph, the services that orchestrate, or the API routes:

    Embedder          swap embedding provider
    VectorStore       swap / add a vector database
    KnowledgeSource   add knowledge sources (SQL, web, API, another collection)
    Reranker          add reranking to the retrieval pipeline
    Cache             add Redis (or any cache) behind a uniform interface
    Guardrail         add input/output guardrails as a pipeline
    ToolProvider      add native tools and MCP server tools
    LLMProvider       swap the LLM gateway

The rule that makes this work: services depend on these protocols, never on
concrete adapters. Concrete adapters are chosen once, at wiring time, from
settings (see app/container.py).
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

from app.domain.models import (
    Assistant,
    GuardrailResult,
    ModelSpec,
    RetrievalQuery,
    RetrievedChunk,
    VectorRecord,
)


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors."""

    @property
    def dimension(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Vector persistence and similarity search."""

    async def ensure_ready(self) -> None: ...

    async def upsert(self, records: Sequence[VectorRecord]) -> int: ...

    async def search(
        self,
        vector: Sequence[float],
        *,
        must_match: dict[str, Any],
        limit: int,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]: ...

    async def delete_by_match(self, must_match: dict[str, Any]) -> int: ...

    async def count(self, must_match: dict[str, Any]) -> int: ...

    async def health(self) -> bool: ...


@runtime_checkable
class KnowledgeSource(Protocol):
    """One place answers can be retrieved from.

    RetrievalService fans out over every enabled source and merges results, so
    adding a second source (a SQL table, an HTTP API, a web search) means
    implementing this protocol and registering it — nothing else changes.
    """

    @property
    def name(self) -> str: ...

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorders candidate chunks by relevance to the question."""

    @property
    def name(self) -> str: ...

    async def rerank(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        *,
        top_n: int,
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class Cache(Protocol):
    """Key/value cache. The null implementation is a no-op, so call sites can
    cache unconditionally and let configuration decide whether it does anything.
    """

    @property
    def enabled(self) -> bool: ...

    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, *, ttl: int | None = None) -> None: ...

    async def get_json(self, key: str) -> Any | None: ...

    async def set_json(self, key: str, value: Any, *, ttl: int | None = None) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def delete_prefix(self, prefix: str) -> int: ...

    async def close(self) -> None: ...


@runtime_checkable
class Guardrail(Protocol):
    """A single policy check. Guardrails compose into an ordered pipeline."""

    @property
    def name(self) -> str: ...

    async def check_input(self, text: str, *, context: dict[str, Any]) -> GuardrailResult: ...

    async def check_output(self, text: str, *, context: dict[str, Any]) -> GuardrailResult: ...


@runtime_checkable
class ToolProvider(Protocol):
    """Supplies tools for an assistant.

    Returns LangChain `BaseTool` instances (typed loosely here to keep the
    domain layer import-free). Native tools and MCP-server tools are both just
    providers, which is why adding MCP later does not alter the graph's shape —
    only whether the tool node has anything to do.
    """

    @property
    def name(self) -> str: ...

    async def get_tools(self, assistant: Assistant) -> list[Any]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Gateway to chat models. OpenRouter today, anything tomorrow."""

    def resolve_spec(self, assistant: Assistant | None) -> ModelSpec: ...

    async def astream(
        self,
        messages: Sequence[dict[str, str]],
        spec: ModelSpec,
        *,
        tools: Sequence[Any] | None = None,
    ) -> Any: ...

    async def acomplete(
        self,
        messages: Sequence[dict[str, str]],
        spec: ModelSpec,
    ) -> str: ...
