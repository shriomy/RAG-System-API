"""/memory — read and update long-term memory.

Phase 1: the user summary, the conversation summary, and the recent message
window. No semantic memory or memory retrieval, as specified.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.deps import ConversationServiceDep, CurrentUser, MemoryServiceDep, SettingsDep
from app.schemas.common import MessageResponse as Ack
from app.schemas.memory import (
    ConversationMemoryResponse,
    MemorySnapshotResponse,
    RecentMessage,
    UpdateConversationMemoryRequest,
    UpdateUserMemoryRequest,
    UserMemoryResponse,
)

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get(
    "/user",
    response_model=UserMemoryResponse,
    summary="Get the caller's long-term summary",
)
async def get_user_memory(
    user: CurrentUser, service: MemoryServiceDep
) -> UserMemoryResponse:
    return UserMemoryResponse.of(await service.get_user_memory(user.id))


@router.put(
    "/user",
    response_model=UserMemoryResponse,
    summary="Overwrite the caller's long-term summary",
)
async def update_user_memory(
    payload: UpdateUserMemoryRequest, user: CurrentUser, service: MemoryServiceDep
) -> UserMemoryResponse:
    return UserMemoryResponse.of(await service.set_user_summary(user.id, payload.summary))


@router.delete("/user", response_model=Ack, summary="Clear the caller's long-term summary")
async def clear_user_memory(user: CurrentUser, service: MemoryServiceDep) -> Ack:
    await service.clear_user_memory(user.id)
    return Ack(message="Long-term user memory cleared.")


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationMemoryResponse,
    summary="Get a conversation's rolling summary",
)
async def get_conversation_memory(
    conversation_id: str,
    user: CurrentUser,
    service: MemoryServiceDep,
    conversations: ConversationServiceDep,
) -> ConversationMemoryResponse:
    # Ownership check before returning summarised content.
    await conversations.get_conversation(conversation_id, user.id)
    memory = await service.get_conversation_memory(conversation_id, user.id)
    return ConversationMemoryResponse.of(memory)


@router.put(
    "/conversations/{conversation_id}",
    response_model=ConversationMemoryResponse,
    summary="Overwrite a conversation's rolling summary",
)
async def update_conversation_memory(
    conversation_id: str,
    payload: UpdateConversationMemoryRequest,
    user: CurrentUser,
    service: MemoryServiceDep,
    conversations: ConversationServiceDep,
) -> ConversationMemoryResponse:
    await conversations.get_conversation(conversation_id, user.id)
    memory = await service.set_conversation_summary(
        conversation_id, user.id, payload.summary
    )
    return ConversationMemoryResponse.of(memory)


@router.delete(
    "/conversations/{conversation_id}",
    response_model=Ack,
    summary="Clear a conversation's rolling summary",
)
async def clear_conversation_memory(
    conversation_id: str,
    user: CurrentUser,
    service: MemoryServiceDep,
    conversations: ConversationServiceDep,
) -> Ack:
    await conversations.get_conversation(conversation_id, user.id)
    await service.clear_conversation_memory(conversation_id, user.id)
    return Ack(message="Conversation memory cleared.")


@router.get(
    "/snapshot",
    response_model=MemorySnapshotResponse,
    summary="Exactly what the agent would load as memory for the next turn",
    description=(
        "Mirrors the Load User Memory node, so you can inspect the memory the "
        "LLM will actually see without running a chat turn."
    ),
)
async def snapshot(
    user: CurrentUser,
    service: MemoryServiceDep,
    conversations: ConversationServiceDep,
    settings: SettingsDep,
    conversation_id: str = Query(description="Conversation to snapshot"),
) -> MemorySnapshotResponse:
    await conversations.get_conversation(conversation_id, user.id)
    context = await service.load_context(
        user_id=user.id, conversation_id=conversation_id
    )
    return MemorySnapshotResponse(
        user_summary=context.user_summary,
        conversation_summary=context.conversation_summary,
        recent_messages=[RecentMessage.of(turn) for turn in context.recent_messages],
        recent_message_limit=settings.memory_recent_messages,
    )
