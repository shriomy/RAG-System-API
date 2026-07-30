"""Shared response shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MessageResponse(BaseModel):
    """Simple acknowledgement for deletes and side-effecting actions."""

    message: str
    details: dict[str, Any] | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Documents the error envelope produced by app/core/errors.py."""

    error: ErrorDetail


class HealthResponse(BaseModel):
    status: str
    version: str
    dependencies: dict[str, str] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
