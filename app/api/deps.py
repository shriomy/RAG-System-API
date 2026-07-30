"""FastAPI dependencies.

Services are resolved from the container stored on `app.state`, so there is a
single instance of each per process and no global singletons.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.container import Container
from app.core.config import Settings, get_settings
from app.core.errors import UnauthorizedError
from app.domain.models import AuthenticatedUser
from app.services.assistant_service import AssistantService
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService
from app.services.knowledge_service import KnowledgeService
from app.services.memory_service import MemoryService

# auto_error=False so a missing header raises our own 401 envelope rather than
# FastAPI's default shape.
bearer_scheme = HTTPBearer(auto_error=False, description="Supabase access token")


def get_container(request: Request) -> Container:
    container: Container | None = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - only if lifespan did not run
        raise RuntimeError("Container is not initialised.")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def get_current_user(
    container: ContainerDep,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> AuthenticatedUser:
    """Verify the Supabase JWT and return the caller's identity.

    Every user-scoped endpoint depends on this; the returned `id` is the only
    source of `user_id` used in queries, so a client cannot act on another
    user's data by passing a different id in a body or path.
    """
    if credentials is None or not credentials.credentials:
        raise UnauthorizedError("Missing Authorization: Bearer <token> header.")
    return container.jwt_verifier.verify(credentials.credentials)


CurrentUser = Annotated[AuthenticatedUser, Depends(get_current_user)]


# ---------------------------------------------------------------------------
# Services
# ---------------------------------------------------------------------------


def get_assistant_service(container: ContainerDep) -> AssistantService:
    return container.assistant_service


def get_knowledge_service(container: ContainerDep) -> KnowledgeService:
    return container.knowledge_service


def get_conversation_service(container: ContainerDep) -> ConversationService:
    return container.conversation_service


def get_memory_service(container: ContainerDep) -> MemoryService:
    return container.memory_service


def get_chat_service(container: ContainerDep) -> ChatService:
    return container.chat_service


AssistantServiceDep = Annotated[AssistantService, Depends(get_assistant_service)]
KnowledgeServiceDep = Annotated[KnowledgeService, Depends(get_knowledge_service)]
ConversationServiceDep = Annotated[ConversationService, Depends(get_conversation_service)]
MemoryServiceDep = Annotated[MemoryService, Depends(get_memory_service)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
