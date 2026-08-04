"""Domain models — the vocabulary shared by services, nodes and adapters."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


class AuthenticatedUser(_Model):
    id: str
    email: str | None = None
    role: str = "authenticated"
    session_id: str | None = None
    expires_at: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class FileType(StrEnum):
    PDF = "pdf"
    TXT = "txt"
    MD = "md"


class FileStatus(StrEnum):
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class ModelSpec(_Model):
    """Everything needed to instantiate an LLM call.

    Built from the assistant row, which is why switching models is a data
    change: no code in services or nodes names a model.
    """

    model: str
    temperature: float = 0.7
    max_tokens: int | None = 1024
    stop: list[str] | None = None
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(_Model):
    """Per-assistant retrieval overrides, sourced from `assistants.config`."""

    top_k: int = 5
    candidate_k: int = 20
    score_threshold: float | None = None
    reranker: str | None = None
    sources: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Persisted entities
# ---------------------------------------------------------------------------


class Assistant(_Model):
    id: str
    user_id: str
    name: str
    system_prompt: str = ""
    is_active: bool = False
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1024
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # -- config-derived views (no code change needed to add knobs) -----------

    def to_model_spec(self, *, default_model: str) -> ModelSpec:
        overrides = self.config.get("llm", {}) if isinstance(self.config, dict) else {}
        return ModelSpec(
            model=self.model or default_model,
            temperature=float(overrides.get("temperature", self.temperature)),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            stop=overrides.get("stop"),
            extra_body=overrides.get("extra_body", {}) or {},
        )

    def to_retrieval_config(self, *, defaults: RetrievalConfig) -> RetrievalConfig:
        raw = self.config.get("retrieval", {}) if isinstance(self.config, dict) else {}
        if not isinstance(raw, dict):
            raw = {}
        return RetrievalConfig(
            top_k=int(raw.get("top_k", defaults.top_k)),
            candidate_k=int(raw.get("candidate_k", defaults.candidate_k)),
            score_threshold=raw.get("score_threshold", defaults.score_threshold),
            reranker=raw.get("reranker", defaults.reranker),
            sources=raw.get("sources", defaults.sources) or [],
        )

    def knowledge_scope_context(self) -> str:
        """Return the assistant-specific context used by the scope classifier."""
        raw = self.config.get("knowledge_scope", {}) if isinstance(self.config, dict) else {}
        if isinstance(raw, dict):
            for key in ("summary", "context", "description", "notes"):
                value = raw.get(key)
                if value and str(value).strip():
                    return str(value).strip()

        legacy = self.config.get("knowledge_summary") if isinstance(self.config, dict) else None
        if legacy and str(legacy).strip():
            return str(legacy).strip()

        return self.system_prompt.strip()

    @property
    def enabled_tools(self) -> list[str]:
        raw = self.config.get("tools", []) if isinstance(self.config, dict) else []
        return list(raw) if isinstance(raw, list) else []

    @property
    def mcp_server_names(self) -> list[str]:
        raw = self.config.get("mcp_servers", []) if isinstance(self.config, dict) else []
        return list(raw) if isinstance(raw, list) else []


class Conversation(_Model):
    id: str
    user_id: str
    assistant_id: str | None = None
    title: str = "New conversation"
    created_at: datetime | None = None
    updated_at: datetime | None = None


class Message(_Model):
    id: str
    conversation_id: str
    user_id: str
    role: MessageRole
    content: str
    created_at: datetime | None = None


class KnowledgeFile(_Model):
    id: str
    user_id: str
    assistant_id: str
    filename: str
    file_type: FileType
    storage_path: str
    status: FileStatus = FileStatus.PROCESSING
    file_size: int | None = None
    chunk_count: int = 0
    error_message: str | None = None
    indexed_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class UserMemory(_Model):
    user_id: str
    summary: str = ""
    turn_count: int = 0
    updated_at: datetime | None = None


class ConversationMemory(_Model):
    conversation_id: str
    user_id: str
    summary: str = ""
    message_count: int = 0
    updated_at: datetime | None = None


# ---------------------------------------------------------------------------
# Retrieval / ingestion value objects
# ---------------------------------------------------------------------------


class DocumentChunk(_Model):
    """A split chunk on its way into the vector store."""

    id: str
    text: str
    user_id: str
    assistant_id: str
    file_id: str
    filename: str
    file_type: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRecord(_Model):
    """A chunk paired with its embedding, ready to upsert."""

    id: str
    vector: list[float]
    payload: dict[str, Any]


class RetrievedChunk(_Model):
    """A retrieval hit, source-agnostic.

    `source` names which KnowledgeSource produced it, which is what makes
    fanning out to more sources later a no-op for downstream code.
    """

    id: str
    text: str
    score: float
    source: str = "vector"
    filename: str | None = None
    file_id: str | None = None
    chunk_index: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_citation(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "filename": self.filename,
            "file_id": self.file_id,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
        }


class RetrievalQuery(_Model):
    """What a KnowledgeSource is asked for. Uniform across sources."""

    question: str
    user_id: str
    assistant_id: str
    top_k: int = 5
    candidate_k: int = 20
    score_threshold: float | None = None
    filters: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chat / memory value objects
# ---------------------------------------------------------------------------


class ChatTurn(_Model):
    """One message in the recent-history window handed to the prompt builder."""

    role: MessageRole
    content: str


class MemoryContext(_Model):
    """The long-term memory slice loaded for a single agent run."""

    user_summary: str = ""
    conversation_summary: str = ""
    recent_messages: list[ChatTurn] = Field(default_factory=list)


class TokenUsage(_Model):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResult(_Model):
    """Outcome of a completed agent run."""

    answer: str
    conversation_id: str
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    usage: TokenUsage | None = None
    model: str | None = None


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------


class GuardrailResult(_Model):
    allowed: bool = True
    guardrail: str = ""
    reason: str | None = None
    # A guardrail may rewrite rather than block (redaction, masking).
    replacement: str | None = None
