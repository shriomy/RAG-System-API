"""Application entry point.

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.router import build_api_router, health_router
from app.container import Container
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.observability import configure_observability

logger = get_logger(__name__)

DESCRIPTION = """
Agentic RAG backend — FastAPI, LangChain and LangGraph over Supabase, Qdrant and
OpenRouter.

**Auth.** Every endpoint except `/health` and `/ready` requires
`Authorization: Bearer <supabase access token>`. The token is verified against
the Supabase project; its `sub` claim is the only source of the user id used in
queries.

**Agent workflow.** `POST /api/v1/chat` runs a LangGraph workflow:
`load assistant -> load user memory -> retrieve documents -> build prompt -> LLM
-> save conversation -> update memory`, streaming tokens over SSE.

**Model selection.** The model is read from `assistants.model`. Changing it is a
`PATCH /api/v1/assistants/{id}` — never a deploy.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the container on startup, tear it down on shutdown."""
    settings: Settings = get_settings()

    configure_observability(settings)

    container = Container(settings)
    app.state.container = container

    await container.startup()
    logger.info(
        "%s v%s ready [env=%s] — docs at /docs",
        settings.app_name,
        __version__,
        settings.environment,
    )

    try:
        yield
    finally:
        await container.shutdown()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        # Hide interactive docs in production.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Lets the browser read these on the streaming response.
        expose_headers=["X-Accel-Buffering"],
    )

    register_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(build_api_router(settings.api_v1_prefix))

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "api": settings.api_v1_prefix,
        }

    return app


app = create_app()
