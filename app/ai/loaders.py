"""LangChain document loaders.

Files arrive as bytes from Supabase Storage, so each loader is fed via a
NamedTemporaryFile. Adding a format later means one entry in `_LOADERS`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import Any

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from app.core.errors import UnsupportedMediaTypeError, ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt", ".md": "md", ".markdown": "md"}


def _load_pdf(path: str) -> list[Document]:
    # PyPDFLoader yields one Document per page and sets metadata["page"].
    return PyPDFLoader(path).load()


def _load_text(path: str) -> list[Document]:
    return TextLoader(path, encoding="utf-8", autodetect_encoding=True).load()


_LOADERS: dict[str, tuple[Callable[[str], list[Document]], str]] = {
    "pdf": (_load_pdf, ".pdf"),
    "txt": (_load_text, ".txt"),
    "md": (_load_text, ".md"),
}


def detect_file_type(filename: str) -> str:
    """Map a filename to one of our supported file types."""
    _, ext = os.path.splitext(filename.lower())
    file_type = SUPPORTED_EXTENSIONS.get(ext)
    if not file_type:
        raise UnsupportedMediaTypeError(
            f"Unsupported file type '{ext or filename}'. Allowed: PDF, TXT, MD.",
            details={"supported": sorted(SUPPORTED_EXTENSIONS)},
        )
    return file_type


def load_documents(
    content: bytes,
    *,
    file_type: str,
    metadata: dict[str, Any] | None = None,
) -> list[Document]:
    """Parse raw bytes into LangChain Documents.

    Synchronous and CPU/IO bound — callers run it in a worker thread.
    """
    entry = _LOADERS.get(file_type)
    if entry is None:
        raise UnsupportedMediaTypeError(f"No loader registered for file type '{file_type}'.")
    loader_fn, suffix = entry

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        documents = loader_fn(tmp_path)
    except UnsupportedMediaTypeError:
        raise
    except Exception as exc:
        raise ValidationError(f"Could not parse the document: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:  # pragma: no cover - Windows file-lock edge case
                logger.debug("Could not remove temp file %s", tmp_path)

    # The temp path is meaningless downstream; replace it with real metadata.
    for document in documents:
        document.metadata.pop("source", None)
        document.metadata.update(metadata or {})

    non_empty = [d for d in documents if d.page_content and d.page_content.strip()]
    if not non_empty:
        raise ValidationError("The document contains no extractable text.")
    return non_empty
