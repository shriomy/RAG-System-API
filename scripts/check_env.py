"""Preflight check — verifies your .env and every external dependency.

Run this before starting the API. It tells you exactly what is missing and how to
fix it, instead of leaving you to decode a runtime traceback.

Checks, in order:
  1. .env is present and the required variables are set (and look plausible)
  2. Supabase REST reachable with the service_role key
  3. Every required table and column exists (i.e. migrations have been run)
  4. The storage bucket exists
  5. The JWT secret actually verifies a token this project would issue
  6. Qdrant reachable
  7. OpenRouter reachable and the default model responds

Run:  .venv\\Scripts\\python.exe scripts\\check_env.py
      .venv\\Scripts\\python.exe scripts\\check_env.py --skip-llm   (no token spend)
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK = "  [ OK ]"
BAD = "  [FAIL]"
WARN = "  [WARN]"

problems: list[str] = []
warnings_: list[str] = []


def ok(message: str) -> None:
    print(f"{OK} {message}")


def bad(message: str, fix: str) -> None:
    print(f"{BAD} {message}")
    print(f"         → {fix}")
    problems.append(message)


def warn(message: str, fix: str = "") -> None:
    print(f"{WARN} {message}")
    if fix:
        print(f"         → {fix}")
    warnings_.append(message)


# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------


def check_config():
    print("\n[1] Configuration (.env)")

    if not Path(".env").exists():
        bad(".env not found", "Copy .env.example to .env and fill in the secrets.")
        return None

    try:
        from app.core.config import get_settings

        settings = get_settings()
    except Exception as exc:
        bad(f"Settings failed to load: {exc}", "Fix the reported field in .env.")
        return None

    ok(".env parsed")

    if not settings.supabase_service_role_key:
        bad(
            "SUPABASE_SERVICE_ROLE_KEY is empty",
            "Supabase → Project Settings → API → service_role (secret).",
        )
    elif not settings.supabase_service_role_key.startswith("eyJ"):
        warn(
            "SUPABASE_SERVICE_ROLE_KEY does not look like a JWT",
            "It should be a long token starting with 'eyJ'. Copy the service_role key, "
            "not the project ref or password.",
        )
    else:
        ok("SUPABASE_SERVICE_ROLE_KEY set")

    secret = settings.supabase_jwt_secret
    if not secret:
        warn(
            "SUPABASE_JWT_SECRET is empty — falling back to JWKS verification",
            f"Correct only if your project uses asymmetric signing keys. "
            f"Will verify against {settings.supabase_jwks_url}",
        )
    elif secret.strip().upper() in {"HS256", "RS256", "ES256"}:
        bad(
            f"SUPABASE_JWT_SECRET is set to '{secret}' — that is the ALGORITHM name",
            "You need the actual secret VALUE: Supabase → Project Settings → API → "
            "JWT Settings → JWT Secret. It is a long random string.",
        )
    elif len(secret) < 32:
        warn(
            f"SUPABASE_JWT_SECRET is only {len(secret)} characters",
            "Supabase JWT secrets are 40+ characters. Double-check you copied all of it.",
        )
    else:
        ok("SUPABASE_JWT_SECRET set")

    if not settings.openrouter_api_key:
        bad("OPENROUTER_API_KEY is empty", "Get one at https://openrouter.ai/keys")
    elif not settings.openrouter_api_key.startswith("sk-or-"):
        warn(
            "OPENROUTER_API_KEY does not start with 'sk-or-'",
            "OpenRouter keys look like sk-or-v1-…",
        )
    else:
        ok("OPENROUTER_API_KEY set")

    print(f"         embedding: {settings.embedding_provider} / {settings.embedding_model}")
    print(f"         qdrant:    {settings.qdrant_url} / {settings.qdrant_collection}")
    print(f"         model:     {settings.openrouter_default_model} (fallback)")
    print(f"         cors:      {settings.cors_origins}")

    return settings


# ---------------------------------------------------------------------------
# 2-4. Supabase
# ---------------------------------------------------------------------------

REQUIRED_TABLES: dict[str, list[str]] = {
    "assistants": ["id", "user_id", "name", "system_prompt", "is_active",
                   "model", "temperature", "max_tokens", "config"],
    "conversations": ["id", "user_id", "assistant_id", "title"],
    "messages": ["id", "conversation_id", "user_id", "role", "content"],
    "knowledge_files": ["id", "user_id", "assistant_id", "filename", "file_type",
                        "storage_path", "status", "chunk_count", "error_message",
                        "indexed_at"],
    "user_memory": ["user_id", "summary", "turn_count"],
    "conversation_memory": ["conversation_id", "user_id", "summary", "message_count"],
}


async def check_supabase(settings) -> None:
    print("\n[2] Supabase")

    if not settings.supabase_service_role_key:
        warn("Skipped — no service_role key")
        return

    import httpx

    from app.repositories.supabase_client import SupabaseClient

    client = SupabaseClient(settings)
    try:
        # -- reachability + credentials ---------------------------------------
        try:
            await client.select("assistants", columns="id", limit=1)
            ok(f"Reachable and authenticated ({settings.supabase_url})")
        except Exception as exc:
            text = str(exc)
            if "401" in text or "JWT" in text or "apikey" in text.lower():
                bad(
                    "Supabase rejected the service_role key",
                    "Re-copy SUPABASE_SERVICE_ROLE_KEY from Project Settings → API.",
                )
            elif "42P01" in text or "does not exist" in text:
                bad(
                    "Table 'assistants' does not exist",
                    "Run migrations/001_baseline.sql then 002_agent_backend.sql "
                    "in the Supabase SQL editor.",
                )
            else:
                bad(f"Supabase request failed: {text[:200]}", "Check SUPABASE_URL and network.")
            return

        # -- schema ------------------------------------------------------------
        print("\n[3] Database schema (migrations)")
        for table, columns in REQUIRED_TABLES.items():
            try:
                await client.select(table, columns=",".join(columns), limit=1)
                ok(f"{table} — all {len(columns)} required columns present")
            except Exception as exc:
                text = str(exc)
                if "does not exist" in text or "42P01" in text:
                    missing_table = f"'{table}'" in text or table in text
                    if missing_table and "column" not in text.lower():
                        bad(
                            f"Table '{table}' is missing",
                            "Run migrations/001_baseline.sql and 002_agent_backend.sql.",
                        )
                    else:
                        bad(
                            f"Table '{table}' exists but a column is missing: {text[:160]}",
                            "Run migrations/002_agent_backend.sql — it adds the new columns.",
                        )
                else:
                    bad(f"Could not inspect '{table}': {text[:160]}", "See the error above.")

        # -- storage bucket ----------------------------------------------------
        print("\n[4] Storage bucket")
        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                response = await http.get(
                    f"{settings.storage_url}/bucket/{settings.supabase_storage_bucket}",
                    headers={
                        "apikey": settings.supabase_service_role_key,
                        "Authorization": f"Bearer {settings.supabase_service_role_key}",
                    },
                )
            if response.status_code == 200:
                ok(f"Bucket '{settings.supabase_storage_bucket}' exists")
            elif response.status_code == 404:
                bad(
                    f"Bucket '{settings.supabase_storage_bucket}' not found",
                    "Run migrations/001_baseline.sql (it creates the bucket), or create it "
                    "manually in Supabase → Storage.",
                )
            else:
                warn(f"Bucket check returned {response.status_code}: {response.text[:120]}")
        except Exception as exc:
            warn(f"Could not check the storage bucket: {exc}")
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# 5. JWT round trip
# ---------------------------------------------------------------------------


def check_jwt(settings) -> None:
    print("\n[5] JWT verification")

    if not settings.supabase_jwt_secret:
        warn("Skipped — using JWKS mode, which needs a real token to test")
        return
    if settings.supabase_jwt_secret.strip().upper() in {"HS256", "RS256", "ES256"}:
        warn("Skipped — SUPABASE_JWT_SECRET is invalid (see above)")
        return

    import jwt

    from app.core.security import SupabaseJWTVerifier

    # Mimic a token Supabase would issue for a signed-in user.
    token = jwt.encode(
        {
            "sub": "00000000-0000-0000-0000-000000000000",
            "email": "preflight@example.com",
            "role": "authenticated",
            "aud": "authenticated",
            "exp": int(time.time()) + 60,
        },
        settings.supabase_jwt_secret,
        algorithm="HS256",
    )

    try:
        user = SupabaseJWTVerifier(settings).verify(token)
        ok(f"Round trip succeeded (sub={user.id[:8]}…)")
        print(
            "         Note: this proves the secret is usable. If real logins still 401, "
            "the secret does not match your project."
        )
    except Exception as exc:
        bad(f"Verification failed: {exc}", "Re-copy the JWT Secret from Supabase.")


# ---------------------------------------------------------------------------
# 6. Qdrant
# ---------------------------------------------------------------------------


async def check_qdrant(settings) -> None:
    print("\n[6] Qdrant")

    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=10
    )
    try:
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        ok(f"Reachable at {settings.qdrant_url}")

        if settings.qdrant_collection in names:
            info = await client.get_collection(settings.qdrant_collection)
            params = info.config.params.vectors
            size = getattr(params, "size", "?")
            ok(
                f"Collection '{settings.qdrant_collection}' exists "
                f"(dim={size}, points={info.points_count})"
            )
            if size != settings.embedding_dimension:
                bad(
                    f"Dimension mismatch: collection={size}, "
                    f"EMBEDDING_DIMENSION={settings.embedding_dimension}",
                    "Either set QDRANT_COLLECTION to a new name or delete the collection.",
                )
        else:
            print(
                f"         Collection '{settings.qdrant_collection}' does not exist yet "
                f"— the API creates it on startup."
            )
    except Exception as exc:
        bad(
            f"Cannot reach Qdrant at {settings.qdrant_url} ({type(exc).__name__})",
            "Start it with: docker compose up -d qdrant   "
            "(Docker Desktop must be running.)",
        )
    finally:
        try:
            await client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 7. OpenRouter
# ---------------------------------------------------------------------------


async def check_openrouter(settings, *, skip_llm: bool) -> None:
    print("\n[7] OpenRouter")

    if not settings.openrouter_api_key:
        warn("Skipped — no API key")
        return

    import httpx

    from app.ai.llm import openrouter_headers

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        **openrouter_headers(settings),
    }

    # Key validity, with no token spend.
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            response = await http.get(
                f"{settings.openrouter_base_url}/key", headers=headers
            )
        if response.status_code == 200:
            data = response.json().get("data", {})
            limit = data.get("limit")
            usage = data.get("usage")
            ok("API key is valid")
            if limit is not None:
                print(f"         credit limit: {limit}, used: {usage}")
            elif usage is not None:
                print(f"         usage so far: {usage} (no hard limit set)")
        elif response.status_code == 401:
            bad(
                "OpenRouter rejected the API key",
                "Generate a new key at https://openrouter.ai/keys",
            )
            return
        else:
            warn(f"Key check returned {response.status_code}: {response.text[:150]}")
    except Exception as exc:
        bad(
            f"Cannot reach OpenRouter ({type(exc).__name__})",
            "Check your internet connection and OPENROUTER_BASE_URL.",
        )
        return

    if skip_llm:
        print("         Skipping the live completion (--skip-llm).")
        return

    # Smallest possible real completion, to prove the default model resolves.
    try:
        from app.domain.models import ModelSpec
        from app.services.openrouter_service import OpenRouterService

        service = OpenRouterService(settings)
        answer = await service.acomplete(
            [{"role": "user", "content": "Reply with the single word: ready"}],
            ModelSpec(
                model=settings.openrouter_default_model, temperature=0.0, max_tokens=5
            ),
        )
        ok(
            f"Model '{settings.openrouter_default_model}' responded: "
            f"{answer[:40]!r}"
        )
    except Exception as exc:
        bad(
            f"Completion failed: {str(exc)[:200]}",
            "If this says the model is unknown, pick a slug from "
            "https://openrouter.ai/models",
        )


# ---------------------------------------------------------------------------


async def main() -> int:
    skip_llm = "--skip-llm" in sys.argv

    print("=" * 72)
    print("RAG System API — preflight check")
    print("=" * 72)

    settings = check_config()
    if settings is None:
        print("\nCannot continue without valid settings.")
        return 1

    await check_supabase(settings)
    check_jwt(settings)
    await check_qdrant(settings)
    await check_openrouter(settings, skip_llm=skip_llm)

    print("\n" + "=" * 72)
    if problems:
        print(f"{len(problems)} problem(s) must be fixed before the system will work:\n")
        for item in problems:
            print(f"  - {item}")
        if warnings_:
            print(f"\nPlus {len(warnings_)} warning(s).")
        return 1

    if warnings_:
        print(f"Ready, with {len(warnings_)} warning(s):\n")
        for item in warnings_:
            print(f"  - {item}")
        return 0

    print("Everything checks out. Start the API with:")
    print("  .venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
