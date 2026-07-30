"""Shared response shapes."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class MessageResponse(BaseModel):
    """Simple acknowledgement for deletes and side-effecting actions."""

    message: str
    details: dict[str, Any] | None = None


class ListResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int = 0

    @classmethod
    def of(cls, items: list[T]) -> "ListResponse[T]":
        return cls(items=items, total=len(items))


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
