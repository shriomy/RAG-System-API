"""Long-term memory DTOs.

Phase 1 exposes the user summary, the conversation summary, and the recent
message window. Semantic memory would add fields here without changing these.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models import ChatTurn, ConversationMemory, UserMemory


class UserMemoryResponse(BaseModel):
    user_id: str
    summary: str = ""
    turn_count: int = 0
    updated_at: datetime | None = None

    @classmethod
    def of(cls, memory: UserMemory) -> "UserMemoryResponse":
        return cls.model_validate(memory.model_dump())


class ConversationMemoryResponse(BaseModel):
    conversation_id: str
    user_id: str
    summary: str = ""
    message_count: int = 0
    updated_at: datetime | None = None

    @classmethod
    def of(cls, memory: ConversationMemory) -> "ConversationMemoryResponse":
        return cls.model_validate(memory.model_dump())


class RecentMessage(BaseModel):
    role: str
    content: str

    @classmethod
    def of(cls, turn: ChatTurn) -> "RecentMessage":
        return cls(role=str(turn.role), content=turn.content)


class MemorySnapshotResponse(BaseModel):
    """Everything the agent would load for a turn — the memory the LLM sees."""

    user_summary: str = ""
    conversation_summary: str = ""
    recent_messages: list[RecentMessage] = Field(default_factory=list)
    recent_message_limit: int = 10


class UpdateUserMemoryRequest(BaseModel):
    summary: str = Field(max_length=20_000)


class UpdateConversationMemoryRequest(BaseModel):
    summary: str = Field(max_length=20_000)
