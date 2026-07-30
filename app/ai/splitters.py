"""LangChain text splitters.

Markdown gets a structure-aware splitter; PDF and plain text get the recursive
character splitter. Adding a semantic or token-aware splitter later is one
entry in `_build_splitter`.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_splitter(file_type: str, chunk_size: int, chunk_overlap: int) -> TextSplitter:
    if file_type == "md":
        return MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def split_documents(
    documents: list[Document],
    *,
    file_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into chunks, tagging each with its ordinal index."""
    splitter = _build_splitter(file_type, chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    kept: list[Document] = []
    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue
        chunk.page_content = text
        chunk.metadata["chunk_index"] = len(kept)
        kept.append(chunk)

    logger.debug(
        "Split %d document(s) into %d chunk(s) [type=%s size=%d overlap=%d]",
        len(documents),
        len(kept),
        file_type,
        chunk_size,
        chunk_overlap,
    )
    return kept
