"""HTTP-layer test — real ASGI request cycle, no external services.

Verifies the things that are easy to get subtly wrong at the edges:

  * the app starts even when Supabase and Qdrant are unreachable
  * unauthenticated requests are rejected with our error envelope, not FastAPI's
  * a forged or expired token is rejected
  * a valid token reaches the route and yields the right user id
  * request validation produces the documented envelope
  * CORS preflight succeeds for the frontend's origin
  * the SSE stream is well-formed and terminates, even on failure

Needs httpx (already a dependency).

Run:  .venv\\Scripts\\python.exe scripts\\test_api.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-jwt-secret-value-at-least-32-chars")
os.environ.setdefault("OPENROUTER_API_KEY", "test-openrouter-key")
# Point Qdrant somewhere dead on purpose: the API must still come up.
os.environ["QDRANT_URL"] = "http://127.0.0.1:59999"

import jwt  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  [PASS] {name}")
    else:
        print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
        failures.append(name)


def make_token(*, sub: str = "user-123", ttl: int = 3600, secret: str | None = None) -> str:
    settings = get_settings()
    return jwt.encode(
        {
            "sub": sub,
            "email": "dev@example.com",
            "role": "authenticated",
            "aud": "authenticated",
            "exp": int(time.time()) + ttl,
        },
        secret or str(settings.supabase_jwt_secret),
        algorithm="HS256",
    )


def main() -> int:
    print("=" * 70)
    print("HTTP layer test")
    print("=" * 70)

    # Entering TestClient runs the lifespan, which builds the container and
    # tries to reach Qdrant. It must not raise.
    print("\n[1] Startup with dependencies unreachable")
    try:
        client_cm = TestClient(app)
        client = client_cm.__enter__()
        check("app starts although Qdrant is unreachable", True)
    except Exception as exc:
        check("app starts although Qdrant is unreachable", False, f"{type(exc).__name__}: {exc}")
        return 1

    try:
        print("\n[2] Health endpoints")
        response = client.get("/health")
        check("GET /health is 200", response.status_code == 200, str(response.status_code))
        check("reports ok", response.json().get("status") == "ok", response.text[:120])

        response = client.get("/ready")
        body = response.json()
        check("GET /ready is 200", response.status_code == 200, str(response.status_code))
        check(
            "readiness reports Qdrant as unreachable",
            body["dependencies"]["qdrant"] == "unreachable",
            str(body.get("dependencies")),
        )
        check(
            "readiness reports degraded",
            body["status"] == "degraded",
            body.get("status", ""),
        )
        check(
            "readiness exposes the active configuration",
            body["config"]["knowledge_sources"] == ["vector"]
            and body["config"]["reranker"] == "passthrough"
            and body["config"]["cache"] == "none",
            str(body.get("config")),
        )

        print("\n[3] Auth enforcement")
        response = client.get("/api/v1/assistants")
        check("no token -> 401", response.status_code == 401, str(response.status_code))
        body = response.json()
        check(
            "401 uses our error envelope",
            body.get("error", {}).get("code") == "unauthorized",
            response.text[:160],
        )

        response = client.get(
            "/api/v1/assistants", headers={"Authorization": "Bearer not-a-jwt"}
        )
        check("garbage token -> 401", response.status_code == 401, str(response.status_code))

        response = client.get(
            "/api/v1/assistants",
            headers={"Authorization": f"Bearer {make_token(ttl=-60)}"},
        )
        check("expired token -> 401", response.status_code == 401, str(response.status_code))

        response = client.get(
            "/api/v1/assistants",
            headers={"Authorization": f"Bearer {make_token(secret='wrong-secret')}"},
        )
        check("forged token -> 401", response.status_code == 401, str(response.status_code))

        print("\n[4] Valid token reaches the route")
        token = make_token(sub="user-abc")
        headers = {"Authorization": f"Bearer {token}"}

        response = client.get("/api/v1/auth/me", headers=headers)
        check("GET /auth/me is 200", response.status_code == 200, response.text[:160])
        check(
            "user id comes from the token's sub claim",
            response.json().get("id") == "user-abc",
            response.text[:160],
        )

        response = client.post("/api/v1/auth/verify", json={"token": token})
        check("POST /auth/verify accepts a good token", response.json().get("valid") is True)

        response = client.post("/api/v1/auth/verify", json={"token": make_token(ttl=-10)})
        body = response.json()
        check(
            "POST /auth/verify reports invalid without a 401",
            response.status_code == 200 and body.get("valid") is False,
            response.text[:160],
        )
        check("verify explains why", bool(body.get("reason")), response.text[:160])

        print("\n[5] Request validation")
        response = client.post("/api/v1/chat/sync", json={}, headers=headers)
        check("missing question -> 422", response.status_code == 422, str(response.status_code))
        check(
            "422 uses our error envelope",
            response.json().get("error", {}).get("code") == "validation_error",
            response.text[:200],
        )

        response = client.post(
            "/api/v1/chat/sync", json={"question": "   "}, headers=headers
        )
        check(
            "whitespace-only question is rejected",
            response.status_code in (422, 400),
            str(response.status_code),
        )

        print("\n[6] CORS preflight for the frontend origin")
        response = client.options(
            "/api/v1/assistants",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "authorization",
            },
        )
        check("preflight is 200", response.status_code == 200, str(response.status_code))
        check(
            "origin is allowed",
            response.headers.get("access-control-allow-origin") == "http://localhost:3000",
            str(response.headers.get("access-control-allow-origin")),
        )
        check(
            "credentials are allowed",
            response.headers.get("access-control-allow-credentials") == "true",
        )

        print("\n[7] SSE stream shape")
        # Supabase is unreachable, so this turn fails — which is exactly the case
        # worth testing: the stream must still be valid SSE and terminate.
        with client.stream(
            "POST",
            "/api/v1/chat",
            json={"question": "Hello there", "assistant_id": "does-not-exist"},
            headers=headers,
        ) as response:
            check(
                "content type is text/event-stream",
                response.headers.get("content-type", "").startswith("text/event-stream"),
                response.headers.get("content-type", ""),
            )
            check(
                "buffering is disabled for proxies",
                response.headers.get("x-accel-buffering") == "no",
            )
            frames = [line for line in response.iter_lines() if line.startswith("data:")]

        check("stream produced frames", len(frames) > 0, str(len(frames)))
        check(
            "stream terminates with [DONE]",
            frames and frames[-1].strip() == "data: [DONE]",
            frames[-1] if frames else "no frames",
        )

        import json as _json

        events = []
        for frame in frames:
            payload = frame[len("data:") :].strip()
            if payload == "[DONE]":
                continue
            events.append(_json.loads(payload))
        check("every non-terminal frame is valid JSON", True)
        check(
            "an upstream failure is reported as an error event",
            any(event.get("type") == "error" for event in events),
            str([e.get("type") for e in events]),
        )

        print("\n[8] OpenAPI schema")
        schema = client.get("/openapi.json").json()
        check("schema is served", "paths" in schema)
        check(
            "chat endpoint documents the SSE contract",
            "text/event-stream"
            in str(schema["paths"]["/api/v1/chat"]["post"].get("responses", {})),
        )
    finally:
        client_cm.__exit__(None, None, None)

    print("\n" + "=" * 70)
    if failures:
        print(f"FAILED — {len(failures)} check(s) did not pass:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
