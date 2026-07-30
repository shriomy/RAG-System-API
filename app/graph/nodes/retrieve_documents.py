"""Node: Retrieve Documents.

Delegates the whole retrieve -> merge -> rerank pipeline to RetrievalService.
Adding knowledge sources or turning on reranking happens inside that service,
so this node is final.

Retrieval failures degrade rather than abort: an assistant with an unreachable
vector store should still answer, just without grounding.
"""

from __future__ import annotations

from app.core.errors import AppError
from app.core.logging import get_logger
from app.domain.models import Assistant
from app.graph.state import AgentState, dump_chunks
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


def make_retrieve_documents_node(retrieval_service: RetrievalService):
    async def retrieve_documents(state: AgentState) -> dict:
        assistant = Assistant.model_validate(state["assistant"])
        config = retrieval_service.config_for(assistant)

        try:
            chunks = await retrieval_service.retrieve(
                question=state["question"],
                user_id=state["user_id"],
                assistant_id=state["assistant_id"],
                config=config,
            )
        except AppError as exc:
            logger.warning("Retrieval failed, continuing without context: %s", exc.message)
            return {
                "retrieved_docs": [],
                "citations": [],
                "metadata": {**(state.get("metadata") or {}), "retrieval_error": exc.message},
            }

        logger.debug("Retrieved %d chunk(s) for assistant %s", len(chunks), assistant.id)
        return {
            "retrieved_docs": dump_chunks(chunks),
            "citations": [chunk.as_citation() for chunk in chunks],
        }

    return retrieve_documents
