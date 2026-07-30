"""Async Supabase client over PostgREST + Storage REST.

Written directly on httpx rather than using supabase-py so that connection
pooling, timeouts, retries and error translation are all explicit and shared
with the rest of the app. Repositories are the only callers.

Authenticated with the service_role key, which bypasses RLS — every repository
method therefore takes and filters on `user_id` explicitly. Ownership is
enforced in code, never assumed.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal, Sequence
from urllib.parse import quote

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import Settings
from app.core.errors import NotFoundError, UpstreamError
from app.core.logging import get_logger

logger = get_logger(__name__)

Row = dict[str, Any]
Order = tuple[str, Literal["asc", "desc"]]

_RETRYABLE = (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout)


class SupabaseClient:
    """Low-level Supabase access. One instance per process, owned by the container."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._rest_url = settings.rest_url
        self._storage_url = settings.storage_url
        key = settings.supabase_service_role_key

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.supabase_timeout_seconds),
            limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
            },
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # ======================================================================
    # Table operations (PostgREST)
    # ======================================================================

    async def select(
        self,
        table: str,
        *,
        columns: str = "*",
        eq: dict[str, Any] | None = None,
        in_: dict[str, Sequence[Any]] | None = None,
        is_null: Sequence[str] | None = None,
        order: Order | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[Row]:
        params: list[tuple[str, str]] = [("select", columns)]
        params.extend(self._eq_params(eq))
        params.extend(self._in_params(in_))
        for column in is_null or ():
            params.append((column, "is.null"))
        if order:
            params.append(("order", f"{order[0]}.{order[1]}"))
        if limit is not None:
            params.append(("limit", str(limit)))
        if offset is not None:
            params.append(("offset", str(offset)))

        data = await self._request("GET", self._table_url(table), params=params)
        return data if isinstance(data, list) else []

    async def select_one(
        self,
        table: str,
        *,
        columns: str = "*",
        eq: dict[str, Any] | None = None,
    ) -> Row | None:
        rows = await self.select(table, columns=columns, eq=eq, limit=1)
        return rows[0] if rows else None

    async def insert(self, table: str, payload: Row | list[Row]) -> list[Row]:
        data = await self._request(
            "POST",
            self._table_url(table),
            json_body=payload,
            headers={"Prefer": "return=representation"},
        )
        return self._as_rows(data)

    async def insert_one(self, table: str, payload: Row) -> Row:
        rows = await self.insert(table, payload)
        if not rows:
            raise UpstreamError(f"Insert into {table} returned no row.")
        return rows[0]

    async def update(self, table: str, payload: Row, *, eq: dict[str, Any]) -> list[Row]:
        if not eq:
            raise ValueError("update() requires at least one filter — refusing full-table update")
        data = await self._request(
            "PATCH",
            self._table_url(table),
            params=self._eq_params(eq),
            json_body=payload,
            headers={"Prefer": "return=representation"},
        )
        return self._as_rows(data)

    async def update_one(self, table: str, payload: Row, *, eq: dict[str, Any]) -> Row:
        rows = await self.update(table, payload, eq=eq)
        if not rows:
            raise NotFoundError(f"No {table} row matched the update filter.")
        return rows[0]

    async def upsert(
        self,
        table: str,
        payload: Row | list[Row],
        *,
        on_conflict: str,
    ) -> list[Row]:
        data = await self._request(
            "POST",
            self._table_url(table),
            params=[("on_conflict", on_conflict)],
            json_body=payload,
            headers={"Prefer": "resolution=merge-duplicates,return=representation"},
        )
        return self._as_rows(data)

    async def delete(self, table: str, *, eq: dict[str, Any]) -> list[Row]:
        if not eq:
            raise ValueError("delete() requires at least one filter — refusing full-table delete")
        data = await self._request(
            "DELETE",
            self._table_url(table),
            params=self._eq_params(eq),
            headers={"Prefer": "return=representation"},
        )
        return self._as_rows(data)

    async def count(self, table: str, *, eq: dict[str, Any] | None = None) -> int:
        params: list[tuple[str, str]] = [("select", "id")]
        params.extend(self._eq_params(eq))
        response = await self._send(
            "GET",
            self._table_url(table),
            params=params,
            headers={"Prefer": "count=exact", "Range": "0-0"},
        )
        # PostgREST reports the total in Content-Range as "0-0/123".
        content_range = response.headers.get("content-range", "")
        total = content_range.split("/")[-1] if "/" in content_range else "0"
        return int(total) if total.isdigit() else 0

    # ======================================================================
    # Storage operations
    # ======================================================================

    async def storage_upload(
        self,
        bucket: str,
        path: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
        upsert: bool = True,
    ) -> None:
        url = f"{self._storage_url}/object/{bucket}/{self._encode_path(path)}"
        try:
            response = await self._client.post(
                url,
                content=content,
                headers={
                    "Content-Type": content_type,
                    "x-upsert": "true" if upsert else "false",
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise UpstreamError(
                f"Storage upload failed ({exc.response.status_code}): {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Storage upload failed: {exc}") from exc

    async def storage_download(self, bucket: str, path: str) -> bytes:
        url = f"{self._storage_url}/object/{bucket}/{self._encode_path(path)}"
        try:
            response = await self._client.get(url)
            if response.status_code == 404:
                raise NotFoundError(f"File not found in storage: {path}")
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as exc:
            raise UpstreamError(
                f"Storage download failed ({exc.response.status_code}): {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError(f"Storage download failed: {exc}") from exc

    async def storage_remove(self, bucket: str, paths: Sequence[str]) -> None:
        if not paths:
            return
        url = f"{self._storage_url}/object/{bucket}"
        try:
            response = await self._client.request(
                "DELETE",
                url,
                json={"prefixes": list(paths)},
                headers={"Content-Type": "application/json"},
            )
            # A missing object is not an error for our purposes.
            if response.status_code not in (200, 204, 404):
                response.raise_for_status()
        except httpx.HTTPError as exc:
            # Never let orphaned-blob cleanup break a user-facing delete.
            logger.warning("Storage remove failed for %s: %s", paths, exc)

    async def storage_signed_url(self, bucket: str, path: str, expires_in: int = 3600) -> str:
        url = f"{self._storage_url}/object/sign/{bucket}/{self._encode_path(path)}"
        data = await self._request("POST", url, json_body={"expiresIn": expires_in})
        signed = (data or {}).get("signedURL") if isinstance(data, dict) else None
        if not signed:
            raise UpstreamError("Storage did not return a signed URL.")
        return f"{self._settings.supabase_url}/storage/v1{signed}"

    # ======================================================================
    # Health
    # ======================================================================

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self._rest_url}/", timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    # ======================================================================
    # Internals
    # ======================================================================

    def _table_url(self, table: str) -> str:
        return f"{self._rest_url}/{table}"

    @staticmethod
    def _encode_path(path: str) -> str:
        return quote(path, safe="/")

    @staticmethod
    def _as_rows(data: Any) -> list[Row]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []

    @staticmethod
    def _eq_params(eq: dict[str, Any] | None) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = []
        for column, value in (eq or {}).items():
            if value is None:
                params.append((column, "is.null"))
            elif isinstance(value, bool):
                params.append((column, f"eq.{str(value).lower()}"))
            else:
                params.append((column, f"eq.{value}"))
        return params

    @staticmethod
    def _in_params(in_: dict[str, Sequence[Any]] | None) -> list[tuple[str, str]]:
        params: list[tuple[str, str]] = []
        for column, values in (in_ or {}).items():
            joined = ",".join(f'"{v}"' for v in values)
            params.append((column, f"in.({joined})"))
        return params

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.3, max=3),
        reraise=True,
    )
    async def _send(
        self,
        method: str,
        url: str,
        *,
        params: Iterable[tuple[str, str]] | None = None,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        merged = {"Content-Type": "application/json", **(headers or {})}
        response = await self._client.request(
            method,
            url,
            params=list(params) if params else None,
            json=json_body,
            headers=merged,
        )
        if response.status_code >= 400:
            raise UpstreamError(
                f"Supabase {method} {url.rsplit('/', 1)[-1]} failed "
                f"({response.status_code}): {self._extract_message(response)}"
            )
        return response

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: Iterable[tuple[str, str]] | None = None,
        json_body: Any | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        try:
            response = await self._send(
                method, url, params=params, json_body=json_body, headers=headers
            )
        except _RETRYABLE as exc:
            raise UpstreamError(f"Supabase is unreachable: {exc}") from exc

        if not response.content:
            return None
        try:
            return response.json()
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _extract_message(response: httpx.Response) -> str:
        try:
            body = response.json()
        except json.JSONDecodeError:
            return response.text[:300]
        if isinstance(body, dict):
            return str(body.get("message") or body.get("error") or body)[:300]
        return str(body)[:300]
