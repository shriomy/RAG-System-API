"""Auth DTOs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AuthUserResponse(BaseModel):
    """The verified identity behind the presented Supabase JWT."""

    id: str
    email: str | None = None
    role: str = "authenticated"
    expires_at: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifyTokenRequest(BaseModel):
    """For verifying a token supplied in the body rather than the header."""

    token: str = Field(min_length=10)


class VerifyTokenResponse(BaseModel):
    valid: bool
    user: AuthUserResponse | None = None
    reason: str | None = None
