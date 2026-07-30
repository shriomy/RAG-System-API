"""Assistant DTOs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models import Assistant


class AssistantResponse(BaseModel):
    """Matches the frontend `Assistant` type, plus the model configuration."""

    model_config = ConfigDict(protected_namespaces=())

    id: str
    user_id: str
    name: str
    system_prompt: str
    is_active: bool
    model: str
    temperature: float
    max_tokens: int
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def of(cls, assistant: Assistant) -> "AssistantResponse":
        return cls.model_validate(assistant.model_dump())


class AssistantCreateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(min_length=1, max_length=120)
    system_prompt: str | None = Field(default=None, max_length=20_000)
    #: Any OpenRouter model slug. Omit to use OPENROUTER_DEFAULT_MODEL.
    model: str | None = Field(default=None, max_length=120)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    #: Forward-compatible overrides: retrieval, tools, mcp_servers, llm.extra_body.
    config: dict[str, Any] | None = None


class AssistantUpdateRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    name: str | None = Field(default=None, min_length=1, max_length=120)
    system_prompt: str | None = Field(default=None, max_length=20_000)
    model: str | None = Field(default=None, max_length=120)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=200_000)
    is_active: bool | None = None
    config: dict[str, Any] | None = None
