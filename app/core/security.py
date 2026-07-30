"""Supabase JWT verification.

Supports both signing schemes Supabase projects use:

* HS256 with the project's shared JWT secret (legacy / default projects)
* RS256 / ES256 against the project's published JWKS (asymmetric signing keys)

Which one runs is decided by config: set SUPABASE_JWT_SECRET for the first,
leave it blank for the second.
"""

from __future__ import annotations

from typing import Any

import jwt
from jwt import PyJWKClient

from app.core.config import Settings
from app.core.errors import UnauthorizedError
from app.core.logging import get_logger
from app.domain.models import AuthenticatedUser

logger = get_logger(__name__)

_SYMMETRIC_ALGORITHMS = ["HS256"]
_ASYMMETRIC_ALGORITHMS = ["RS256", "ES256", "RS512", "ES512", "EdDSA"]


class SupabaseJWTVerifier:
    """Verifies `Authorization: Bearer <supabase access token>`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jwks_client: PyJWKClient | None = None

        if settings.supabase_jwt_secret:
            self._mode = "secret"
        else:
            self._mode = "jwks"
            # PyJWKClient caches keys in-process and refetches on unknown kid.
            self._jwks_client = PyJWKClient(
                str(settings.supabase_jwks_url),
                cache_keys=True,
                lifespan=600,
            )
        logger.info("Supabase JWT verification mode: %s", self._mode)

    # -- public API ----------------------------------------------------------

    def verify(self, token: str) -> AuthenticatedUser:
        """Decode and validate a token, or raise UnauthorizedError."""
        if not token:
            raise UnauthorizedError("Missing bearer token.")

        try:
            claims = self._decode(token)
        except jwt.ExpiredSignatureError as exc:
            raise UnauthorizedError("Token has expired.") from exc
        except jwt.InvalidAudienceError as exc:
            raise UnauthorizedError("Token audience is invalid.") from exc
        except jwt.PyJWTError as exc:
            logger.info("JWT rejected: %s", exc)
            raise UnauthorizedError("Token is invalid.") from exc
        except Exception as exc:  # JWKS fetch failures surface here
            logger.warning("JWT verification failed unexpectedly: %s", exc)
            raise UnauthorizedError("Unable to verify token.") from exc

        return self._to_user(claims)

    # -- internals -----------------------------------------------------------

    def _decode(self, token: str) -> dict[str, Any]:
        options = {"require": ["exp", "sub"]}

        if self._mode == "secret":
            return jwt.decode(
                token,
                str(self._settings.supabase_jwt_secret),
                algorithms=_SYMMETRIC_ALGORITHMS,
                audience=self._settings.supabase_jwt_audience,
                options=options,
            )

        assert self._jwks_client is not None
        signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=_ASYMMETRIC_ALGORITHMS,
            audience=self._settings.supabase_jwt_audience,
            options=options,
        )

    @staticmethod
    def _to_user(claims: dict[str, Any]) -> AuthenticatedUser:
        user_id = claims.get("sub")
        if not user_id:
            raise UnauthorizedError("Token has no subject claim.")

        metadata = claims.get("user_metadata") or {}
        return AuthenticatedUser(
            id=str(user_id),
            email=claims.get("email"),
            role=claims.get("role", "authenticated"),
            session_id=claims.get("session_id"),
            expires_at=claims.get("exp"),
            metadata=metadata if isinstance(metadata, dict) else {},
        )
