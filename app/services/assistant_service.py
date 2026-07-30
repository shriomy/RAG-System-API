"""AssistantService — assistant CRUD and configuration."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.errors import NotFoundError, ValidationError
from app.core.logging import get_logger
from app.domain.models import Assistant, RetrievalConfig
from app.domain.ports import Cache
from app.infrastructure.cache.factory import assistant_key, assistant_list_key
from app.repositories.assistant_repository import AssistantRepository

logger = get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer questions accurately using the "
    "knowledge base provided to you, and say so when you do not know."
)


class AssistantService:
    def __init__(
        self,
        repository: AssistantRepository,
        settings: Settings,
        cache: Cache,
    ) -> None:
        self._repo = repository
        self._settings = settings
        self._cache = cache

    # ======================================================================
    # Reads
    # ======================================================================

    async def list_assistants(self, user_id: str) -> list[Assistant]:
        return await self._repo.list_for_user(user_id)

    async def get_assistant(self, assistant_id: str, user_id: str) -> Assistant:
        """Fetch an assistant or raise. Used by the graph's first node."""
        if self._cache.enabled:
            cached = await self._cache.get_json(assistant_key(user_id, assistant_id))
            if cached:
                return Assistant.model_validate(cached)

        assistant = await self._repo.get(assistant_id, user_id)
        if assistant is None:
            raise NotFoundError(
                "Assistant not found.", details={"assistant_id": assistant_id}
            )

        if self._cache.enabled:
            await self._cache.set_json(
                assistant_key(user_id, assistant_id),
                assistant.model_dump(mode="json"),
                ttl=self._settings.cache_ttl_seconds,
            )
        return assistant

    async def get_active_assistant(self, user_id: str) -> Assistant | None:
        return await self._repo.get_active(user_id)

    # ======================================================================
    # Writes
    # ======================================================================

    async def create_assistant(
        self,
        user_id: str,
        *,
        name: str,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> Assistant:
        payload: dict[str, Any] = {
            "name": name.strip() or "New assistant",
            "system_prompt": (
                system_prompt if system_prompt is not None else DEFAULT_SYSTEM_PROMPT
            ),
            "model": model or self._settings.openrouter_default_model,
            "temperature": (
                temperature if temperature is not None else self._settings.llm_temperature
            ),
            "max_tokens": max_tokens or self._settings.llm_max_tokens,
            "config": config or {},
        }
        assistant = await self._repo.create(user_id, payload)
        await self._invalidate(user_id, assistant.id)
        logger.info("Created assistant %s for user %s", assistant.id, user_id)
        return assistant

    async def update_assistant(
        self, assistant_id: str, user_id: str, patch: dict[str, Any]
    ) -> Assistant:
        # Reject a no-op rather than issuing an empty PATCH.
        clean = {k: v for k, v in patch.items() if v is not None}
        if not clean:
            raise ValidationError("No fields to update.")

        if "name" in clean and not str(clean["name"]).strip():
            raise ValidationError("Assistant name cannot be empty.")

        updated = await self._repo.update(assistant_id, user_id, clean)
        if updated is None:
            raise NotFoundError(
                "Assistant not found.", details={"assistant_id": assistant_id}
            )

        await self._invalidate(user_id, assistant_id)
        return updated

    async def delete_assistant(self, assistant_id: str, user_id: str) -> None:
        """Delete an assistant.

        Its knowledge files and conversations go with it via ON DELETE CASCADE;
        the matching Qdrant points are removed by KnowledgeService, which is
        called by the route so that this service stays free of vector concerns.
        """
        deleted = await self._repo.delete(assistant_id, user_id)
        if not deleted:
            raise NotFoundError(
                "Assistant not found.", details={"assistant_id": assistant_id}
            )
        await self._invalidate(user_id, assistant_id)
        logger.info("Deleted assistant %s for user %s", assistant_id, user_id)

    async def set_active(self, assistant_id: str, user_id: str) -> Assistant:
        """Make one assistant active, deactivating the rest."""
        if not await self._repo.exists(assistant_id, user_id):
            raise NotFoundError(
                "Assistant not found.", details={"assistant_id": assistant_id}
            )
        await self._repo.deactivate_all(user_id, except_id=assistant_id)
        await self._invalidate(user_id, assistant_id)
        return await self.get_assistant(assistant_id, user_id)

    # ======================================================================
    # Internals
    # ======================================================================

    async def _invalidate(self, user_id: str, assistant_id: str) -> None:
        if not self._cache.enabled:
            return
        await self._cache.delete(assistant_key(user_id, assistant_id))
        await self._cache.delete(assistant_list_key(user_id))
