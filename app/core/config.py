"""Application settings.

Every pluggable subsystem (embeddings, reranking, cache, guardrails, tools,
MCP, observability) is selected here by name and resolved through a registry.
Turning a feature on is an env change plus an implementation registration —
never a change to services, nodes, or routes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

EmbeddingProvider = Literal["fastembed", "openai", "voyage"]
RerankerProvider = Literal["none", "cohere"]
CacheProvider = Literal["none", "redis"]

#: pydantic-settings JSON-decodes complex types (list, dict) straight from the
#: env var before any validator runs, so `FOO=a,b` would raise a JSONDecodeError.
#: NoDecode suppresses that and hands the raw string to our `_parse_csv`
#: validator, which is what lets these be plain comma-separated env vars.
CsvList = Annotated[list[str], NoDecode]


def _csv(value: Any) -> list[str]:
    """Parse a comma-separated env var into a clean list of strings."""
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- application ---------------------------------------------------------
    app_name: str = "RAG System API"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_origins: CsvList = Field(default_factory=lambda: ["http://localhost:3000"])

    # -- supabase ------------------------------------------------------------
    supabase_url: str
    supabase_service_role_key: str
    supabase_anon_key: str | None = None
    supabase_jwt_secret: str | None = None
    supabase_jwks_url: str | None = None
    supabase_jwt_audience: str = "authenticated"
    supabase_storage_bucket: str = "knowledge_files"
    supabase_timeout_seconds: float = 30.0

    # -- openrouter ----------------------------------------------------------
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_default_model: str = "openai/gpt-4o-mini"
    openrouter_app_url: str | None = None
    openrouter_app_title: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    summary_model: str = "openai/gpt-4o-mini"

    # -- embeddings ----------------------------------------------------------
    embedding_provider: EmbeddingProvider = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None

    # -- qdrant --------------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "rag_documents"
    qdrant_timeout_seconds: float = 30.0

    # -- ingestion -----------------------------------------------------------
    chunk_size: int = 1000
    chunk_overlap: int = 150
    max_upload_mb: int = 25

    # -- retrieval -----------------------------------------------------------
    retrieval_top_k: int = 5
    retrieval_candidate_k: int = 20
    retrieval_score_threshold: float | None = None
    reranker_provider: RerankerProvider = "none"
    reranker_model: str = "rerank-v3.5"
    reranker_api_key: str | None = None

    # -- memory --------------------------------------------------------------
    memory_recent_messages: int = 10
    memory_summary_enabled: bool = True
    memory_summary_max_chars: int = 2000
    memory_user_summary_every_n_turns: int = 3

    # -- cache ---------------------------------------------------------------
    cache_provider: CacheProvider = "none"
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 300
    cache_namespace: str = "rag"

    # -- guardrails ----------------------------------------------------------
    guardrails_enabled: bool = False
    guardrails: CsvList = Field(default_factory=list)

    # -- tools / mcp ---------------------------------------------------------
    tools_enabled: bool = False
    enabled_tools: CsvList = Field(default_factory=list)
    mcp_enabled: bool = False
    mcp_servers: dict[str, Any] = Field(default_factory=dict)

    # -- observability -------------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "rag-system-api"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # ---- validators --------------------------------------------------------

    @field_validator("cors_origins", "guardrails", "enabled_tools", mode="before")
    @classmethod
    def _parse_csv(cls, value: Any) -> list[str]:
        return _csv(value)

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _parse_mcp_servers(cls, value: Any) -> dict[str, Any]:
        if not value:
            return {}
        if isinstance(value, dict):
            return value
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError as exc:  # pragma: no cover - config error path
            raise ValueError(f"MCP_SERVERS must be valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("MCP_SERVERS must be a JSON object")
        return parsed

    @field_validator("retrieval_score_threshold", mode="before")
    @classmethod
    def _empty_float_to_none(cls, value: Any) -> Any:
        return None if value in ("", None) else value

    @field_validator("supabase_url", "openrouter_base_url", "qdrant_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @model_validator(mode="after")
    def _check_auth_config(self) -> Settings:
        # Fall back to the project's published JWKS when no shared secret is set.
        if not self.supabase_jwt_secret and not self.supabase_jwks_url:
            self.supabase_jwks_url = f"{self.supabase_url}/auth/v1/.well-known/jwks.json"
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.retrieval_candidate_k < self.retrieval_top_k:
            self.retrieval_candidate_k = self.retrieval_top_k
        return self

    # ---- derived helpers ---------------------------------------------------

    @property
    def rest_url(self) -> str:
        return f"{self.supabase_url}/rest/v1"

    @property
    def storage_url(self) -> str:
        return f"{self.supabase_url}/storage/v1"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()  # type: ignore[call-arg]
