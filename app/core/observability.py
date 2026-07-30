"""Observability seam.

LangSmith is configured through environment variables that LangChain reads on
its own, so enabling tracing is genuinely zero-code: set LANGSMITH_TRACING=true
and an API key. `tracing_callbacks()` exists so that *additional* callback
handlers (a custom cost tracker, an OTel bridge, Langfuse) can be introduced
later by editing this one function — every LLM/graph invocation already asks
for its callbacks here.
"""

from __future__ import annotations

import os
from typing import Any

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_observability(settings: Settings) -> None:
    """Export the env vars LangChain/LangGraph look for. Call once at startup."""
    if not settings.langsmith_tracing:
        # Make sure a stray env var doesn't silently enable tracing.
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        logger.info("LangSmith tracing disabled")
        return

    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_TRACING is true but LANGSMITH_API_KEY is unset; skipping")
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    # Legacy aliases still honoured by some langchain versions.
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)


def tracing_callbacks() -> list[Any]:
    """Extra callback handlers attached to every LLM / graph invocation.

    Empty today — LangSmith hooks itself in globally via env vars. Append
    handlers here to add observability backends without touching call sites.
    """
    return []


def run_metadata(**fields: Any) -> dict[str, Any]:
    """Metadata attached to traced runs, so LangSmith runs are filterable."""
    return {k: v for k, v in fields.items() if v is not None}
