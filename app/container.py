"""Composition root.

This is the ONE place that knows which concrete implementation backs each port.
Services receive their dependencies here and nowhere else, which is what makes
the extension seams real: adding Redis, a reranker, another knowledge source,
tools or MCP servers changes this file and an adapter — never a service, a graph
node or a route.

Built once per process during the FastAPI lifespan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.ai.embeddings import build_embeddings
from app.ai.rerankers import build_reranker
from app.core.config import Settings
from app.core.logging import get_logger
from app.core.security import SupabaseJWTVerifier
from app.domain.ports import Cache, KnowledgeSource, Reranker
from app.graph.builder import GraphDependencies
from app.infrastructure.cache.factory import build_cache
from app.infrastructure.guardrails.pipeline import GuardrailPipeline, build_guardrail_pipeline
from app.infrastructure.knowledge_sources.registry import build_knowledge_sources
from app.infrastructure.tools.registry import ToolResolver, build_tool_resolver
from app.repositories.assistant_repository import AssistantRepository
from app.repositories.conversation_repository import ConversationRepository
from app.repositories.knowledge_repository import KnowledgeRepository
from app.repositories.memory_repository import MemoryRepository
from app.repositories.storage_repository import StorageRepository
from app.repositories.supabase_client import SupabaseClient
from app.services.assistant_service import AssistantService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.embedding_service import EmbeddingService
from app.services.graph_service import GraphService
from app.services.knowledge_service import KnowledgeService
from app.services.memory_service import MemoryService
from app.services.openrouter_service import OpenRouterService
from app.services.qdrant_service import QdrantService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


@dataclass
class Container:
    """Holds every long-lived object in the process."""

    settings: Settings

    # infrastructure
    supabase: SupabaseClient = field(init=False)
    cache: Cache = field(init=False)
    jwt_verifier: SupabaseJWTVerifier = field(init=False)
    guardrails: GuardrailPipeline = field(init=False)
    tool_resolver: ToolResolver = field(init=False)
    reranker: Reranker = field(init=False)

    # repositories
    assistant_repository: AssistantRepository = field(init=False)
    conversation_repository: ConversationRepository = field(init=False)
    knowledge_repository: KnowledgeRepository = field(init=False)
    memory_repository: MemoryRepository = field(init=False)
    storage_repository: StorageRepository = field(init=False)

    # services
    embedding_service: EmbeddingService = field(init=False)
    qdrant_service: QdrantService = field(init=False)
    openrouter_service: OpenRouterService = field(init=False)
    assistant_service: AssistantService = field(init=False)
    retrieval_service: RetrievalService = field(init=False)
    memory_service: MemoryService = field(init=False)
    conversation_service: ConversationService = field(init=False)
    knowledge_service: KnowledgeService = field(init=False)
    graph_service: GraphService = field(init=False)
    chat_service: ChatService = field(init=False)

    knowledge_sources: list[KnowledgeSource] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        settings = self.settings

        # -- infrastructure --------------------------------------------------
        self.supabase = SupabaseClient(settings)
        self.cache = build_cache(settings)
        self.jwt_verifier = SupabaseJWTVerifier(settings)
        self.guardrails = build_guardrail_pipeline(settings)
        self.tool_resolver = build_tool_resolver(settings)
        self.reranker = build_reranker(settings)

        # -- repositories ---------------------------------------------------
        self.assistant_repository = AssistantRepository(self.supabase)
        self.conversation_repository = ConversationRepository(self.supabase)
        self.knowledge_repository = KnowledgeRepository(self.supabase)
        self.memory_repository = MemoryRepository(self.supabase)
        self.storage_repository = StorageRepository(self.supabase, settings)

        # -- AI services ----------------------------------------------------
        self.embedding_service = EmbeddingService(
            build_embeddings(settings), settings, self.cache
        )
        self.qdrant_service = QdrantService(
            settings, vector_size=settings.embedding_dimension
        )
        self.openrouter_service = OpenRouterService(settings)

        # -- retrieval ------------------------------------------------------
        # Sources come from the registry, so a new source is a registration.
        self.knowledge_sources = build_knowledge_sources(self)  # type: ignore[arg-type]
        self.retrieval_service = RetrievalService(
            sources=self.knowledge_sources,
            reranker=self.reranker,
            settings=settings,
            cache=self.cache,
        )

        # -- domain services ------------------------------------------------
        self.assistant_service = AssistantService(
            self.assistant_repository, settings, self.cache
        )
        self.memory_service = MemoryService(
            memory_repository=self.memory_repository,
            conversation_repository=self.conversation_repository,
            llm=self.openrouter_service,
            settings=settings,
        )
        self.conversation_service = ConversationService(
            repository=self.conversation_repository,
            llm=self.openrouter_service,
            settings=settings,
        )
        self.knowledge_service = KnowledgeService(
            knowledge_repository=self.knowledge_repository,
            storage_repository=self.storage_repository,
            assistant_repository=self.assistant_repository,
            embedding_service=self.embedding_service,
            qdrant_service=self.qdrant_service,
            retrieval_service=self.retrieval_service,
            settings=settings,
        )

        # -- orchestration --------------------------------------------------
        self.graph_service = GraphService(
            GraphDependencies(
                assistant_service=self.assistant_service,
                memory_service=self.memory_service,
                retrieval_service=self.retrieval_service,
                conversation_service=self.conversation_service,
                llm_service=self.openrouter_service,
                tool_resolver=self.tool_resolver,
                guardrails=self.guardrails,
            ),
            # Extension point (4): pass a LangGraph checkpointer here for
            # durable, resumable runs.
            checkpointer=None,
        )
        self.chat_service = ChatService(
            graph_service=self.graph_service,
            conversation_service=self.conversation_service,
            assistant_service=self.assistant_service,
            guardrails=self.guardrails,
            settings=settings,
        )

    # ======================================================================
    # Lifecycle
    # ======================================================================

    async def startup(self) -> None:
        """Prepare external resources. Failures are logged, not fatal.

        The API should still start when Qdrant is down: assistant management,
        conversations and memory all keep working, and chat degrades to
        answering without retrieval.
        """
        try:
            await self.embedding_service.warm_up()
        except Exception as exc:
            logger.error("Embedding warm-up failed: %s", exc)

        try:
            await self.qdrant_service.ensure_ready(
                vector_size=self.embedding_service.dimension
            )
        except Exception as exc:
            logger.error(
                "Qdrant is not ready (%s). Retrieval will return nothing until it is.", exc
            )

    async def shutdown(self) -> None:
        await self.graph_service.drain()
        await self.qdrant_service.aclose()
        await self.cache.close()
        await self.supabase.aclose()
        logger.info("Container shut down")

    # ======================================================================
    # Diagnostics
    # ======================================================================

    async def health(self) -> dict[str, Any]:
        supabase_ok = await self.supabase.health()
        qdrant_ok = await self.qdrant_service.health()

        return {
            "status": "ok" if (supabase_ok and qdrant_ok) else "degraded",
            "dependencies": {
                "supabase": "ok" if supabase_ok else "unreachable",
                "qdrant": "ok" if qdrant_ok else "unreachable",
            },
            "config": {
                "default_model": self.settings.openrouter_default_model,
                "embedding_provider": self.settings.embedding_provider,
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": self.embedding_service.dimension,
                "qdrant_collection": self.settings.qdrant_collection,
                "knowledge_sources": self.retrieval_service.source_names,
                "reranker": self.retrieval_service.reranker_name,
                "cache": self.settings.cache_provider,
                "guardrails": self.guardrails.names,
                "tools_enabled": self.settings.tools_enabled,
                "mcp_enabled": self.settings.mcp_enabled,
                "langsmith_tracing": self.settings.langsmith_tracing,
            },
        }
