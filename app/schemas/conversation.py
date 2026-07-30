"""Conversation and message DTOs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.models import Conversation, Message


class ConversationResponse(BaseModel):
    """Matches the frontend `Conversation` type."""

    id: str
    user_id: str
    assistant_id: str | None = None
    title: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def of(cls, conversation: Conversation) -> "ConversationResponse":
        return cls.model_validate(conversation.model_dump())


class MessageResponse(BaseModel):
    """Matches the frontend `Message` type."""

    id: str
    conversation_id: str
    user_id: str
    role: str
    content: str
    created_at: datetime | None = None

    @classmethod
    def of(cls, message: Message) -> "MessageResponse":
        return cls.model_validate(message.model_dump())


class ConversationCreateRequest(BaseModel):
    assistant_id: str | None = None
    title: str | None = Field(default=None, max_length=200)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    messages: list[MessageResponse] = Field(default_factory=list)
    message_count: int = 0
