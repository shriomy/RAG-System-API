"""/conversations — list conversations, read messages, delete conversations."""

from __future__ import annotations

from fastapi import APIRouter, Query, status

from app.api.deps import ConversationServiceDep, CurrentUser, MemoryServiceDep
from app.schemas.common import MessageResponse as Ack
from app.schemas.conversation import (
    ConversationCreateRequest,
    ConversationDetailResponse,
    ConversationResponse,
    ConversationUpdateRequest,
    MessageResponse,
)

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get(
    "", response_model=list[ConversationResponse], summary="List the user's conversations"
)
async def list_conversations(
    user: CurrentUser,
    service: ConversationServiceDep,
    assistant_id: str | None = Query(default=None, description="Filter by assistant"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[ConversationResponse]:
    conversations = await service.list_conversations(
        user.id, assistant_id=assistant_id, limit=limit, offset=offset
    )
    return [ConversationResponse.of(c) for c in conversations]


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an empty conversation",
    description=(
        "Optional — POST /chat creates one automatically when "
        "`conversation_id` is omitted."
    ),
)
async def create_conversation(
    payload: ConversationCreateRequest,
    user: CurrentUser,
    service: ConversationServiceDep,
) -> ConversationResponse:
    conversation = await service.create_conversation(
        user.id, assistant_id=payload.assistant_id, title=payload.title
    )
    return ConversationResponse.of(conversation)


@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    summary="Get a conversation with its messages",
)
async def get_conversation(
    conversation_id: str,
    user: CurrentUser,
    service: ConversationServiceDep,
) -> ConversationDetailResponse:
    conversation = await service.get_conversation(conversation_id, user.id)
    messages = await service.list_messages(conversation_id, user.id)
    return ConversationDetailResponse(
        conversation=ConversationResponse.of(conversation),
        messages=[MessageResponse.of(m) for m in messages],
        message_count=len(messages),
    )


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
    summary="List a conversation's messages, oldest first",
)
async def list_messages(
    conversation_id: str,
    user: CurrentUser,
    service: ConversationServiceDep,
    limit: int | None = Query(default=None, ge=1, le=1000),
) -> list[MessageResponse]:
    messages = await service.list_messages(conversation_id, user.id, limit=limit)
    return [MessageResponse.of(m) for m in messages]


@router.patch(
    "/{conversation_id}", response_model=ConversationResponse, summary="Rename a conversation"
)
async def rename_conversation(
    conversation_id: str,
    payload: ConversationUpdateRequest,
    user: CurrentUser,
    service: ConversationServiceDep,
) -> ConversationResponse:
    conversation = await service.rename_conversation(
        conversation_id, user.id, payload.title
    )
    return ConversationResponse.of(conversation)


@router.delete(
    "/{conversation_id}",
    response_model=Ack,
    summary="Delete a conversation, its messages and its memory",
)
async def delete_conversation(
    conversation_id: str,
    user: CurrentUser,
    service: ConversationServiceDep,
    memory: MemoryServiceDep,
) -> Ack:
    # Messages and conversation_memory cascade in the database; the explicit
    # call keeps the behaviour true even if the cascade is ever relaxed.
    await memory.clear_conversation_memory(conversation_id, user.id)
    await service.delete_conversation(conversation_id, user.id)
    return Ack(message="Conversation deleted.")
