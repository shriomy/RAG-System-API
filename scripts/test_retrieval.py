"""Retrieval integration test — real embeddings, real Qdrant client, no server.

Exercises the actual ingestion and retrieval path end to end:

    load -> split -> embed (fastembed) -> upsert (Qdrant) -> search -> rerank

Uses Qdrant's embedded mode, so it needs no running container and no API keys.
The only thing faked is Supabase.

Also verifies the property that matters most for a multi-tenant RAG system:
one user/assistant can never retrieve another's chunks.

Run:  .venv\\Scripts\\python.exe scripts\\test_retrieval.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-jwt-secret-value-at-least-32-chars")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


HANDBOOK = """
# ACME Employee Handbook

## Password policy
To reset your password, open Settings, choose Security, then click
"Reset password". A reset link is emailed to your work address and expires
after 30 minutes. Passwords must be at least 14 characters long.

## Expense reimbursement
Submit expenses through the Finance portal within 30 days of purchase.
Receipts are required for anything above 25 dollars. Reimbursement is paid
with the next payroll run.

## Holiday allowance
Full-time employees receive 25 days of paid holiday per year, plus public
holidays. Unused days may be carried over, up to a maximum of 5 days.
"""

SECRET_DOC = """
# Project Nightingale — restricted

The launch date is 3 March. The budget ceiling is 2.4 million dollars.
Only the steering committee may discuss these figures.
"""


async def main() -> int:
    print("=" * 70)
    print("Retrieval integration test (embedded Qdrant + real embeddings)")
    print("=" * 70)

    from qdrant_client import AsyncQdrantClient

    from app.ai.embeddings import build_embeddings
    from app.ai.loaders import load_documents
    from app.ai.rerankers import build_reranker
    from app.ai.splitters import split_documents
    from app.core.config import get_settings
    from app.domain.models import DocumentChunk, RetrievalConfig
    from app.infrastructure.cache.null_cache import NullCache
    from app.infrastructure.knowledge_sources.vector_source import VectorKnowledgeSource
    from app.services.embedding_service import EmbeddingService
    from app.services.qdrant_service import QdrantService
    from app.services.retrieval_service import RetrievalService

    settings = get_settings()
    cache = NullCache()

    print("\n[1] Embedding model")
    embedding_service = EmbeddingService(build_embeddings(settings), settings, cache)
    await embedding_service.warm_up()
    check(
        "model loads and reports 384 dimensions",
        embedding_service.dimension == 384,
        str(embedding_service.dimension),
    )

    print("\n[2] Qdrant collection")
    qdrant = QdrantService(
        settings,
        vector_size=embedding_service.dimension,
        client=AsyncQdrantClient(location=":memory:"),
    )
    await qdrant.ensure_ready()
    check("collection created", await qdrant.health())

    print("\n[3] Ingest: load -> split -> embed -> upsert")

    async def ingest(
        *, text: str, user_id: str, assistant_id: str, file_id: str, filename: str
    ) -> int:
        documents = load_documents(
            text.encode(), file_type="md", metadata={"file_id": file_id}
        )
        splits = split_documents(
            documents, file_type="md", chunk_size=350, chunk_overlap=60
        )
        chunks = [
            DocumentChunk(
                id=f"{file_id}:{i}",
                text=document.page_content,
                user_id=user_id,
                assistant_id=assistant_id,
                file_id=file_id,
                filename=filename,
                file_type="md",
                chunk_index=i,
            )
            for i, document in enumerate(splits)
        ]
        vectors = await embedding_service.embed_documents([c.text for c in chunks])
        return await qdrant.upsert(qdrant.to_records(chunks, vectors))

    written = await ingest(
        text=HANDBOOK,
        user_id="user-A",
        assistant_id="asst-1",
        file_id="file-handbook",
        filename="handbook.md",
    )
    check("handbook chunks upserted", written > 2, f"{written} chunk(s)")

    # A different user's document, to prove isolation later.
    await ingest(
        text=SECRET_DOC,
        user_id="user-B",
        assistant_id="asst-2",
        file_id="file-secret",
        filename="nightingale.md",
    )
    check(
        "user-A sees only their own chunks",
        await qdrant.count({"user_id": "user-A"}) == written,
    )

    print("\n[4] Retrieval relevance")
    retrieval = RetrievalService(
        sources=[
            VectorKnowledgeSource(
                embedding_service=embedding_service, qdrant_service=qdrant
            )
        ],
        reranker=build_reranker(settings),
        settings=settings,
        cache=cache,
    )
    config = RetrievalConfig(top_k=2, candidate_k=10)

    hits = await retrieval.retrieve(
        question="How do I reset my password?",
        user_id="user-A",
        assistant_id="asst-1",
        config=config,
    )
    check("returns results", len(hits) > 0, f"{len(hits)} hit(s)")
    check("respects top_k=2", len(hits) <= 2, str(len(hits)))
    if hits:
        top = hits[0].text.lower()
        check(
            "top hit is the password section",
            "reset" in top and "password" in top,
            hits[0].text[:90].replace("\n", " "),
        )
        check("hit carries its filename", hits[0].filename == "handbook.md")
        check("hits are score-ordered", all(
            hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1)
        ))

    expenses = await retrieval.retrieve(
        question="What is the deadline for submitting expenses?",
        user_id="user-A",
        assistant_id="asst-1",
        config=config,
    )
    check(
        "a different question retrieves a different section",
        bool(expenses) and "expense" in expenses[0].text.lower(),
        expenses[0].text[:90].replace("\n", " ") if expenses else "no hits",
    )

    print("\n[5] Tenant isolation")
    leak = await retrieval.retrieve(
        question="What is the Project Nightingale launch date and budget?",
        user_id="user-A",
        assistant_id="asst-1",
        config=config,
    )
    check(
        "user-A cannot retrieve user-B's document",
        all("Nightingale" not in hit.text for hit in leak),
        str([h.filename for h in leak]),
    )

    wrong_assistant = await retrieval.retrieve(
        question="How do I reset my password?",
        user_id="user-A",
        assistant_id="asst-999",
        config=config,
    )
    check("an unknown assistant retrieves nothing", wrong_assistant == [])

    owner = await retrieval.retrieve(
        question="What is the Project Nightingale launch date?",
        user_id="user-B",
        assistant_id="asst-2",
        config=config,
    )
    check(
        "the owning user does retrieve it",
        bool(owner) and "Nightingale" in owner[0].text,
        owner[0].text[:70].replace("\n", " ") if owner else "no hits",
    )

    print("\n[6] Re-index does not duplicate")
    before = await qdrant.count({"user_id": "user-A", "file_id": "file-handbook"})
    await ingest(
        text=HANDBOOK,
        user_id="user-A",
        assistant_id="asst-1",
        file_id="file-handbook",
        filename="handbook.md",
    )
    after = await qdrant.count({"user_id": "user-A", "file_id": "file-handbook"})
    check(
        "point ids are deterministic, so re-ingest overwrites",
        before == after,
        f"{before} -> {after}",
    )

    print("\n[7] Delete")
    removed = await qdrant.delete_by_match({"user_id": "user-A", "file_id": "file-handbook"})
    check("delete reports the number removed", removed == after, str(removed))
    check(
        "chunks are gone after delete",
        await qdrant.count({"user_id": "user-A", "file_id": "file-handbook"}) == 0,
    )
    check(
        "the other user's data is untouched",
        await qdrant.count({"user_id": "user-B"}) > 0,
    )

    await qdrant.aclose()

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED — {len(failures)} check(s) did not pass:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
