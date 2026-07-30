"""API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import assistants, auth, chat, conversations, health, knowledge, memory


def build_api_router(prefix: str) -> APIRouter:
    """Mount every v1 router under the configured prefix."""
    router = APIRouter(prefix=prefix)
    router.include_router(auth.router)
    router.include_router(assistants.router)
    router.include_router(knowledge.router)
    router.include_router(chat.router)
    router.include_router(conversations.router)
    router.include_router(memory.router)
    return router


#: Health endpoints stay off the versioned prefix so probes have a stable path.
health_router = health.router
