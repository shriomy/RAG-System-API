"""/health — liveness, readiness and effective configuration."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.deps import ContainerDep
from app.schemas.common import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def health() -> HealthResponse:
    """Always 200 while the process is up. Use /ready for dependency status."""
    return HealthResponse(status="ok", version=__version__)


@router.get(
    "/ready",
    response_model=HealthResponse,
    summary="Readiness check with dependency and configuration detail",
    description=(
        "Reports Supabase and Qdrant reachability plus which pluggable "
        "implementations are active — the quickest way to confirm a feature flag "
        "actually took effect."
    ),
)
async def ready(container: ContainerDep) -> HealthResponse:
    snapshot = await container.health()
    return HealthResponse(
        status=snapshot["status"],
        version=__version__,
        dependencies=snapshot["dependencies"],
        config=snapshot["config"],
    )
