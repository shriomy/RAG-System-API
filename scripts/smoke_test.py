"""Offline smoke test — no Supabase, Qdrant or OpenRouter required.

Verifies the parts that must be right before any network call matters:

  1. every module imports (no circular imports, no missing symbols)
  2. the FastAPI app builds and exposes the expected routes
  3. the container wires every service
  4. the LangGraph compiles with the specified node sequence
  5. prompt assembly layers system prompt / memory / context / question correctly
  6. the graph runs end to end against fake services, streaming tokens

Run:  .venv\\Scripts\\python.exe scripts\\smoke_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Placeholders so Settings validates without real credentials. Set before any
# app import, because get_settings() is cached on first call.
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-jwt-secret-value-at-least-32-chars")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")

PASS = "  [PASS]"
FAIL = "  [FAIL]"

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


# ===========================================================================
# 1. Imports
# ===========================================================================


def test_imports() -> None:
    print("\n[1] Module imports")
    import importlib

    modules = [
        "app.core.config",
        "app.core.errors",
        "app.core.security",
        "app.core.observability",
        "app.domain.models",
        "app.domain.ports",
        "app.repositories.supabase_client",
        "app.repositories.assistant_repository",
        "app.repositories.conversation_repository",
        "app.repositories.knowledge_repository",
        "app.repositories.memory_repository",
        "app.repositories.storage_repository",
        "app.ai.loaders",
        "app.ai.splitters",
        "app.ai.embeddings",
        "app.ai.llm",
        "app.ai.prompts",
        "app.ai.parsers",
        "app.ai.vectorstore",
        "app.ai.rerankers",
        "app.infrastructure.cache",
        "app.infrastructure.guardrails",
        "app.infrastructure.tools",
        "app.infrastructure.knowledge_sources",
        "app.services.embedding_service",
        "app.services.qdrant_service",
        "app.services.openrouter_service",
        "app.services.assistant_service",
        "app.services.retrieval_service",
        "app.services.memory_service",
        "app.services.conversation_service",
        "app.services.knowledge_service",
        "app.services.graph_service",
        "app.services.chat_service",
        "app.graph.state",
        "app.graph.builder",
        "app.container",
        "app.main",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            check(name, False, f"{type(exc).__name__}: {exc}")
            return
    check(f"all {len(modules)} modules import", True)


# ===========================================================================
# 2. Routes
# ===========================================================================


def test_routes() -> None:
    print("\n[2] FastAPI routes")
    from app.main import app

    paths = {
        (route.path, tuple(sorted(getattr(route, "methods", []) or [])))
        for route in app.routes
    }
    flat = {path for path, _ in paths}

    expected = [
        ("/health", "GET"),
        ("/ready", "GET"),
        ("/api/v1/auth/me", "GET"),
        ("/api/v1/auth/verify", "POST"),
        ("/api/v1/assistants", "GET"),
        ("/api/v1/assistants", "POST"),
        ("/api/v1/assistants/{assistant_id}", "PATCH"),
        ("/api/v1/assistants/{assistant_id}", "DELETE"),
        ("/api/v1/assistants/{assistant_id}/activate", "POST"),
        ("/api/v1/knowledge", "GET"),
        ("/api/v1/knowledge/upload", "POST"),
        ("/api/v1/knowledge/index-pending", "POST"),
        ("/api/v1/knowledge/stats", "GET"),
        ("/api/v1/knowledge/{file_id}", "DELETE"),
        ("/api/v1/knowledge/{file_id}/reindex", "POST"),
        ("/api/v1/chat", "POST"),
        ("/api/v1/chat/sync", "POST"),
        ("/api/v1/conversations", "GET"),
        ("/api/v1/conversations/{conversation_id}", "GET"),
        ("/api/v1/conversations/{conversation_id}", "DELETE"),
        ("/api/v1/conversations/{conversation_id}/messages", "GET"),
        ("/api/v1/memory/user", "GET"),
        ("/api/v1/memory/user", "PUT"),
        ("/api/v1/memory/snapshot", "GET"),
        ("/api/v1/memory/conversations/{conversation_id}", "GET"),
    ]

    missing = []
    for path, method in expected:
        if path not in flat:
            missing.append(f"{method} {path} (path absent)")
            continue
        methods = {m for p, ms in paths if p == path for m in ms}
        if method not in methods:
            missing.append(f"{method} {path} (methods={sorted(methods)})")

    check(f"{len(expected)} expected routes registered", not missing, "; ".join(missing))


# ===========================================================================
# 3. Container wiring
# ===========================================================================


def test_container() -> None:
    print("\n[3] Container wiring")
    from app.container import Container
    from app.core.config import get_settings

    try:
        container = Container(get_settings())
    except Exception as exc:
        check("container builds", False, f"{type(exc).__name__}: {exc}")
        return

    check("container builds", True)

    services = [
        "assistant_service",
        "knowledge_service",
        "retrieval_service",
        "memory_service",
        "conversation_service",
        "chat_service",
        "graph_service",
        "embedding_service",
        "qdrant_service",
        "openrouter_service",
    ]
    absent = [name for name in services if getattr(container, name, None) is None]
    check(f"all {len(services)} required services present", not absent, str(absent))

    check(
        "knowledge sources resolved",
        container.retrieval_service.source_names == ["vector"],
        str(container.retrieval_service.source_names),
    )
    check(
        "reranker defaults to pass-through",
        container.retrieval_service.reranker_name == "passthrough",
        container.retrieval_service.reranker_name,
    )
    check("cache defaults to disabled", container.cache.enabled is False)
    check("guardrails default to empty", container.guardrails.enabled is False)
    check("tool resolver default to empty", container.tool_resolver.enabled is False)


# ===========================================================================
# 4. Graph shape
# ===========================================================================


def test_graph_shape() -> None:
    print("\n[4] Graph topology")
    from app.container import Container
    from app.core.config import get_settings
    from app.graph.builder import NODE_SEQUENCE

    container = Container(get_settings())
    graph = container.graph_service.graph

    nodes = set(graph.get_graph().nodes.keys())
    missing = [name for name in NODE_SEQUENCE if name not in nodes]
    check("all 7 nodes registered", not missing, str(missing))

    expected_edges = list(zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]))
    actual = {(edge.source, edge.target) for edge in graph.get_graph().edges}
    bad = [f"{a}->{b}" for a, b in expected_edges if (a, b) not in actual]
    check("nodes wired in the specified order", not bad, str(bad))
    check(
        "START -> load_assistant",
        ("__start__", "load_assistant") in actual,
        str(sorted(actual)),
    )
    check("update_memory -> END", ("update_memory", "__end__") in actual)


# ===========================================================================
# 5. Prompt assembly
# ===========================================================================


def test_prompt_assembly() -> None:
    print("\n[5] Prompt assembly")
    from app.ai.prompts import build_chat_prompt
    from app.domain.models import ChatTurn, MessageRole, RetrievedChunk

    messages = build_chat_prompt(
        system_prompt="You are ACME support.",
        question="How do I reset my password?",
        retrieved_chunks=[
            RetrievedChunk(
                id="c1",
                text="To reset a password, open Settings > Security.",
                score=0.91,
                filename="handbook.pdf",
            )
        ],
        user_summary="Works at ACME as an admin.",
        conversation_summary="Earlier they asked about billing.",
        recent_messages=[
            ChatTurn(role=MessageRole.USER, content="Hi"),
            ChatTurn(role=MessageRole.ASSISTANT, content="Hello! How can I help?"),
        ],
    )

    check("returns system + history + question", len(messages) == 4, str(len(messages)))
    check("first message is the system message", messages[0]["role"] == "system")

    system = messages[0]["content"]
    check("system prompt included", "ACME support" in system)
    check("user summary included", "Works at ACME as an admin." in system)
    check("conversation summary included", "Earlier they asked about billing." in system)

    check("history preserved in order", messages[1]["content"] == "Hi")
    check("history roles preserved", messages[2]["role"] == "assistant")

    final = messages[-1]["content"]
    check("final message is the user question", messages[-1]["role"] == "user")
    check("retrieved chunk text included", "Settings > Security" in final)
    check("source attributed", "handbook.pdf" in final)
    check("question included", "reset my password" in final)

    # Empty-context path must not crash or fabricate a citation.
    bare = build_chat_prompt(system_prompt="Be brief.", question="Hello?")
    check("works with no context or memory", len(bare) == 2)
    check("states when no excerpts found", "No relevant excerpts" in bare[-1]["content"])


# ===========================================================================
# 6. End-to-end graph run against fakes
# ===========================================================================


async def test_graph_execution() -> None:
    print("\n[6] End-to-end graph run (fake services)")

    from app.domain.models import (
        Assistant,
        ChatTurn,
        Conversation,
        MemoryContext,
        Message,
        MessageRole,
        RetrievalConfig,
        RetrievedChunk,
    )
    from app.graph.builder import GraphDependencies
    from app.infrastructure.guardrails.pipeline import GuardrailPipeline
    from app.infrastructure.tools.registry import ToolResolver
    from app.services.graph_service import GraphService

    calls: list[str] = []

    fake_assistant = Assistant(
        id="a1",
        user_id="u1",
        name="Support bot",
        system_prompt="You are ACME support.",
        model="anthropic/claude-sonnet-4.5",
        temperature=0.3,
        max_tokens=512,
    )

    class FakeAssistantService:
        async def get_assistant(self, assistant_id, user_id):
            calls.append("load_assistant")
            return fake_assistant

    class FakeMemoryService:
        async def load_context(self, *, user_id, conversation_id):
            calls.append("load_memory")
            return MemoryContext(
                user_summary="Admin at ACME.",
                conversation_summary="Discussed billing.",
                recent_messages=[ChatTurn(role=MessageRole.USER, content="Hi")],
            )

        async def update_after_turn(self, *, user_id, conversation_id, question, answer):
            calls.append("update_memory")
            return MemoryContext(
                user_summary="Admin at ACME. Asked about passwords.",
                conversation_summary="Discussed billing, then password reset.",
            )

    class FakeRetrievalService:
        def config_for(self, assistant):
            return RetrievalConfig(top_k=3, candidate_k=10)

        async def retrieve(self, *, question, user_id, assistant_id, config):
            calls.append("retrieve")
            return [
                RetrievedChunk(
                    id="c1",
                    text="Open Settings > Security to reset a password.",
                    score=0.9,
                    filename="handbook.pdf",
                    file_id="f1",
                    chunk_index=0,
                )
            ]

    class FakeConversationService:
        async def save_exchange(self, *, conversation_id, user_id, question, answer):
            calls.append("save_conversation")
            return (
                Message(
                    id="m1",
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role=MessageRole.USER,
                    content=question,
                ),
                Message(
                    id="m2",
                    conversation_id=conversation_id,
                    user_id=user_id,
                    role=MessageRole.ASSISTANT,
                    content=answer,
                ),
            )

    class FakeLLM:
        def resolve_spec(self, assistant):
            from app.domain.models import ModelSpec

            return ModelSpec(model=assistant.model, temperature=assistant.temperature)

        async def astream(self, messages, spec, *, tools=None, run_name="", metadata=None, usage_sink=None):
            calls.append("llm")
            # Prove the prompt reached the model with its layers intact.
            system = messages[0]["content"]
            assert "ACME support" in system, "system prompt missing"
            assert "Admin at ACME." in system, "user summary missing"
            assert "Discussed billing." in system, "conversation summary missing"
            assert "Settings > Security" in messages[-1]["content"], "context missing"
            if usage_sink is not None:
                usage_sink.update(
                    {"prompt_tokens": 120, "completion_tokens": 8, "total_tokens": 128}
                )
            for token in ["Open ", "Settings ", "> ", "Security."]:
                yield token

    service = GraphService(
        GraphDependencies(
            assistant_service=FakeAssistantService(),      # type: ignore[arg-type]
            memory_service=FakeMemoryService(),            # type: ignore[arg-type]
            retrieval_service=FakeRetrievalService(),      # type: ignore[arg-type]
            conversation_service=FakeConversationService(),  # type: ignore[arg-type]
            llm_service=FakeLLM(),                         # type: ignore[arg-type]
            tool_resolver=ToolResolver(),
            guardrails=GuardrailPipeline(),
        )
    )

    state = service.new_state(
        user_id="u1",
        assistant_id="a1",
        conversation_id="c1",
        question="How do I reset my password?",
    )

    tokens: list[str] = []
    final = None
    async for kind, payload in service.stream(state):
        if kind == "token":
            tokens.append(payload)
        elif kind == "result":
            final = payload

    check(
        "nodes executed in order",
        calls
        == [
            "load_assistant",
            "load_memory",
            "retrieve",
            "llm",
            "save_conversation",
            "update_memory",
        ],
        str(calls),
    )
    check("tokens streamed individually", len(tokens) == 4, str(tokens))
    check("final state returned", final is not None)

    if final:
        check(
            "answer assembled from tokens",
            final["answer"] == "Open Settings > Security.",
            repr(final.get("answer")),
        )
        check("model recorded from assistant", final["model"] == "anthropic/claude-sonnet-4.5")
        check("usage captured", final["usage"]["total_tokens"] == 128, str(final.get("usage")))
        check("message ids persisted", final["assistant_message_id"] == "m2")
        check("citations built", len(final["citations"]) == 1, str(final.get("citations")))
        check(
            "citation carries filename",
            final["citations"][0]["filename"] == "handbook.pdf",
        )
        check(
            "memory refreshed after the turn",
            "password" in final["user_summary"],
            repr(final.get("user_summary")),
        )
        check("no error recorded", not final.get("error"))


# ===========================================================================
# 7. Splitters / loaders / parsers
# ===========================================================================


def test_ingestion_pieces() -> None:
    print("\n[7] Ingestion components")
    from app.ai.loaders import detect_file_type, load_documents
    from app.ai.parsers import summary_parser, title_parser
    from app.ai.splitters import split_documents
    from app.core.errors import UnsupportedMediaTypeError

    check("detects .pdf", detect_file_type("report.PDF") == "pdf")
    check("detects .md", detect_file_type("notes.md") == "md")
    check("detects .markdown as md", detect_file_type("notes.markdown") == "md")
    check("detects .txt", detect_file_type("log.txt") == "txt")

    try:
        detect_file_type("virus.exe")
        check("rejects unsupported types", False, "no exception raised")
    except UnsupportedMediaTypeError:
        check("rejects unsupported types", True)

    text = ("Section one. " * 60 + "\n\n" + "Section two. " * 60).encode()
    documents = load_documents(text, file_type="txt", metadata={"file_id": "f1"})
    check("loads a text document", len(documents) == 1, str(len(documents)))
    check("metadata attached", documents[0].metadata.get("file_id") == "f1")

    chunks = split_documents(documents, file_type="txt", chunk_size=200, chunk_overlap=40)
    check("splits into multiple chunks", len(chunks) > 3, str(len(chunks)))
    check(
        "chunk_index is sequential from 0",
        [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks))),
    )
    check("chunks respect the size budget", all(len(c.page_content) <= 200 for c in chunks))

    check(
        "summary parser strips labels",
        summary_parser(100).parse("Updated summary: The user likes tea.")
        == "The user likes tea.",
    )
    check(
        "summary parser enforces the budget",
        len(summary_parser(20).parse("x" * 100)) <= 20,
    )
    check(
        "title parser strips quotes",
        title_parser().parse('"Password reset help"') == "Password reset help",
    )


# ===========================================================================
# 8. Auth rejects bad tokens
# ===========================================================================


def test_auth() -> None:
    print("\n[8] JWT verification")
    import time

    import jwt

    from app.core.config import get_settings
    from app.core.errors import UnauthorizedError
    from app.core.security import SupabaseJWTVerifier

    settings = get_settings()
    verifier = SupabaseJWTVerifier(settings)
    secret = settings.supabase_jwt_secret

    valid = jwt.encode(
        {
            "sub": "user-123",
            "email": "a@b.com",
            "role": "authenticated",
            "aud": "authenticated",
            "exp": int(time.time()) + 3600,
        },
        secret,
        algorithm="HS256",
    )
    user = verifier.verify(valid)
    check("accepts a valid token", user.id == "user-123", user.id)
    check("extracts the email claim", user.email == "a@b.com")

    expired = jwt.encode(
        {"sub": "u", "aud": "authenticated", "exp": int(time.time()) - 10},
        secret,
        algorithm="HS256",
    )
    try:
        verifier.verify(expired)
        check("rejects an expired token", False, "accepted")
    except UnauthorizedError as exc:
        check("rejects an expired token", "expired" in exc.message.lower(), exc.message)

    wrong = jwt.encode(
        {"sub": "u", "aud": "authenticated", "exp": int(time.time()) + 3600},
        "a-different-secret",
        algorithm="HS256",
    )
    try:
        verifier.verify(wrong)
        check("rejects a wrongly-signed token", False, "accepted")
    except UnauthorizedError:
        check("rejects a wrongly-signed token", True)

    bad_audience = jwt.encode(
        {"sub": "u", "aud": "anon", "exp": int(time.time()) + 3600},
        secret,
        algorithm="HS256",
    )
    try:
        verifier.verify(bad_audience)
        check("rejects a wrong audience", False, "accepted")
    except UnauthorizedError:
        check("rejects a wrong audience", True)


# ===========================================================================
# 9. Model switching is data, not code
# ===========================================================================


def test_model_switching() -> None:
    print("\n[9] Model configuration")
    from app.core.config import get_settings
    from app.domain.models import Assistant
    from app.services.openrouter_service import OpenRouterService

    service = OpenRouterService(get_settings())

    default = service.resolve_spec(None)
    check(
        "falls back to the configured default",
        default.model == "openai/gpt-4o-mini",
        default.model,
    )

    assistant = Assistant(
        id="a",
        user_id="u",
        name="n",
        model="google/gemini-2.0-flash-001",
        temperature=0.1,
        max_tokens=4096,
    )
    spec = service.resolve_spec(assistant)
    check("model taken from the assistant row", spec.model == "google/gemini-2.0-flash-001")
    check("temperature taken from the assistant row", spec.temperature == 0.1)
    check("max_tokens taken from the assistant row", spec.max_tokens == 4096)

    # config.llm overrides the columns, without a schema change.
    tuned = Assistant(
        id="a",
        user_id="u",
        name="n",
        model="anthropic/claude-sonnet-4.5",
        config={"llm": {"temperature": 0.9, "extra_body": {"provider": {"sort": "price"}}}},
    )
    tuned_spec = service.resolve_spec(tuned)
    check("config.llm overrides the column", tuned_spec.temperature == 0.9)
    check(
        "extra_body passed through for provider routing",
        tuned_spec.extra_body == {"provider": {"sort": "price"}},
        str(tuned_spec.extra_body),
    )

    retrieval = Assistant(
        id="a", user_id="u", name="n", config={"retrieval": {"top_k": 12}}
    )
    from app.domain.models import RetrievalConfig

    merged = retrieval.to_retrieval_config(defaults=RetrievalConfig(top_k=5, candidate_k=20))
    check("config.retrieval overrides top_k", merged.top_k == 12, str(merged.top_k))
    check("unspecified retrieval keys keep defaults", merged.candidate_k == 20)


# ===========================================================================


def main() -> int:
    print("=" * 70)
    print("RAG System API — smoke test")
    print("=" * 70)

    test_imports()
    test_routes()
    test_container()
    test_graph_shape()
    test_prompt_assembly()
    asyncio.run(test_graph_execution())
    test_ingestion_pieces()
    test_auth()
    test_model_switching()

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED — {len(failures)} check(s) did not pass:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
