"""/auth — Supabase JWT verification.

The frontend already authenticates against Supabase directly; the backend's job
is only to verify the token it presents. There is no login/signup here on
purpose — Supabase owns credentials, and duplicating that would mean handling
passwords in two places.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.deps import ContainerDep, CurrentUser
from app.core.errors import UnauthorizedError
from app.schemas.auth import AuthUserResponse, VerifyTokenRequest, VerifyTokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get(
    "/me",
    response_model=AuthUserResponse,
    summary="Return the identity behind the presented token",
)
async def me(user: CurrentUser) -> AuthUserResponse:
    return AuthUserResponse(
        id=user.id,
        email=user.email,
        role=user.role,
        expires_at=user.expires_at,
        metadata=user.metadata,
    )


@router.post(
    "/verify",
    response_model=VerifyTokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Verify a token supplied in the body",
    description=(
        "Returns `valid: false` with a reason rather than a 401, so callers can "
        "probe a token without exception handling."
    ),
)
async def verify_token(
    payload: VerifyTokenRequest, container: ContainerDep
) -> VerifyTokenResponse:
    try:
        user = container.jwt_verifier.verify(payload.token)
    except UnauthorizedError as exc:
        return VerifyTokenResponse(valid=False, reason=exc.message)

    return VerifyTokenResponse(
        valid=True,
        user=AuthUserResponse(
            id=user.id,
            email=user.email,
            role=user.role,
            expires_at=user.expires_at,
            metadata=user.metadata,
        ),
    )
