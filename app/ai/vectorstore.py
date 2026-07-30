"""LangChain <-> Qdrant adapters.

QdrantService (app/services/qdrant_service.py) drives Qdrant with the native
async client, because multi-tenant payload indexes, filtered deletes and
score thresholds are all clearer there.

This module is the LangChain interop seam: it exposes the same collection as a
`QdrantVectorStore` and as a LangChain `BaseRetriever`. That is what lets you
later drop in anything from the LangChain retriever ecosystem —
MultiQueryRetriever, ContextualCompressionRetriever, ParentDocumentRetriever,
EnsembleRetriever — and register it as a KnowledgeSource without touching the
graph, the services or the routes.
"""

from __future__ import annotations

from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as qmodels

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Payload keys. Kept as constants because both this module and QdrantService
# build filters over them.
PAYLOAD_TEXT = "text"
PAYLOAD_USER_ID = "user_id"
PAYLOAD_ASSISTANT_ID = "assistant_id"
PAYLOAD_FILE_ID = "file_id"
PAYLOAD_FILENAME = "filename"
PAYLOAD_CHUNK_INDEX = "chunk_index"


def tenant_filter(**equals: Any) -> qmodels.Filter:
    """Build a Qdrant `must`-match filter from simple equality pairs.

    This is the multi-tenancy primitive: every read and every delete is scoped
    by `user_id` and `assistant_id`, so one collection serves all tenants.
    """
    conditions: list[qmodels.Condition] = [
        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        for key, value in equals.items()
        if value is not None
    ]
    return qmodels.Filter(must=conditions)


def build_langchain_vectorstore(
    settings: Settings,
    embeddings: Embeddings,
    *,
    async_client: AsyncQdrantClient | None = None,
) -> QdrantVectorStore:
    """Wrap the existing collection in a LangChain vector store."""
    return QdrantVectorStore(
        client=QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=int(settings.qdrant_timeout_seconds),
        ),
        async_client=async_client,
        collection_name=settings.qdrant_collection,
        embedding=embeddings,
        content_payload_key=PAYLOAD_TEXT,
        retrieval_mode=RetrievalMode.DENSE,
    )


def build_langchain_retriever(
    settings: Settings,
    embeddings: Embeddings,
    *,
    user_id: str,
    assistant_id: str,
    top_k: int,
    async_client: AsyncQdrantClient | None = None,
) -> BaseRetriever:
    """A tenant-scoped LangChain retriever over the shared collection."""
    store = build_langchain_vectorstore(settings, embeddings, async_client=async_client)
    retriever: VectorStoreRetriever = store.as_retriever(
        search_kwargs={
            "k": top_k,
            "filter": tenant_filter(
                **{PAYLOAD_USER_ID: user_id, PAYLOAD_ASSISTANT_ID: assistant_id}
            ),
        }
    )
    return retriever
