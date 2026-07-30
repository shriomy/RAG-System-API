"""/assistants — assistant CRUD and prompt/model configuration."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import AssistantServiceDep, CurrentUser, KnowledgeServiceDep
from app.schemas.assistant import (
    AssistantCreateRequest,
    AssistantResponse,
    AssistantUpdateRequest,
)
from app.schemas.common import MessageResponse

router = APIRouter(prefix="/assistants", tags=["assistants"])


@router.get("", response_model=list[AssistantResponse], summary="List the user's assistants")
async def list_assistants(
    user: CurrentUser, service: AssistantServiceDep
) -> list[AssistantResponse]:
    assistants = await service.list_assistants(user.id)
    return [AssistantResponse.of(a) for a in assistants]


@router.post(
    "",
    response_model=AssistantResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an assistant",
)
async def create_assistant(
    payload: AssistantCreateRequest, user: CurrentUser, service: AssistantServiceDep
) -> AssistantResponse:
    assistant = await service.create_assistant(
        user.id,
        name=payload.name,
        system_prompt=payload.system_prompt,
        model=payload.model,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
        config=payload.config,
    )
    return AssistantResponse.of(assistant)


@router.get("/{assistant_id}", response_model=AssistantResponse, summary="Get one assistant")
async def get_assistant(
    assistant_id: str, user: CurrentUser, service: AssistantServiceDep
) -> AssistantResponse:
    return AssistantResponse.of(await service.get_assistant(assistant_id, user.id))


@router.patch(
    "/{assistant_id}",
    response_model=AssistantResponse,
    summary="Update an assistant",
    description=(
        "Use this to change the system prompt or to switch the model — setting "
        "`model` to any OpenRouter slug takes effect on the next turn, with no "
        "deploy or restart."
    ),
)
async def update_assistant(
    assistant_id: str,
    payload: AssistantUpdateRequest,
    user: CurrentUser,
    service: AssistantServiceDep,
) -> AssistantResponse:
    assistant = await service.update_assistant(
        assistant_id, user.id, payload.model_dump(exclude_unset=True)
    )
    return AssistantResponse.of(assistant)


@router.post(
    "/{assistant_id}/activate",
    response_model=AssistantResponse,
    summary="Make this the active assistant",
)
async def activate_assistant(
    assistant_id: str, user: CurrentUser, service: AssistantServiceDep
) -> AssistantResponse:
    return AssistantResponse.of(await service.set_active(assistant_id, user.id))


@router.delete(
    "/{assistant_id}",
    response_model=MessageResponse,
    summary="Delete an assistant and its knowledge base",
)
async def delete_assistant(
    assistant_id: str,
    user: CurrentUser,
    service: AssistantServiceDep,
    knowledge: KnowledgeServiceDep,
) -> MessageResponse:
    # Vectors and storage blobs first: the DB cascade removes the rows, and
    # nothing else would know which vectors had been orphaned.
    await knowledge.delete_assistant_knowledge(assistant_id, user.id)
    await service.delete_assistant(assistant_id, user.id)
    return MessageResponse(message="Assistant deleted.")
