"""LangChain output parsers.

Summarisation and titling produce free text that still needs normalising —
models like to wrap answers in quotes, prefix them with "Summary:", or exceed a
length budget. Structured (JSON / Pydantic) parsers would be added here too.
"""

from __future__ import annotations

import re

from langchain_core.output_parsers import BaseOutputParser, StrOutputParser

from app.core.logging import get_logger

logger = get_logger(__name__)

_LEADING_LABEL = re.compile(
    r"^\s*(updated\s+)?(summary|profile|title|answer)\s*:\s*", re.IGNORECASE
)
_SURROUNDING_QUOTES = re.compile(r'^["\'“‘](.*)["\'”’]$', re.DOTALL)


def _strip_boilerplate(text: str) -> str:
    cleaned = _LEADING_LABEL.sub("", (text or "").strip())
    match = _SURROUNDING_QUOTES.match(cleaned.strip())
    if match:
        cleaned = match.group(1)
    return cleaned.strip()


class SummaryOutputParser(BaseOutputParser[str]):
    """Cleans a generated summary and enforces a hard character budget."""

    max_chars: int = 2000

    def parse(self, text: str) -> str:
        cleaned = _strip_boilerplate(text)
        if len(cleaned) <= self.max_chars:
            return cleaned
        # Truncate on a sentence boundary when there is one nearby.
        window = cleaned[: self.max_chars]
        cut = max(window.rfind(". "), window.rfind("\n"))
        return (window[: cut + 1] if cut > self.max_chars * 0.6 else window).strip()

    @property
    def _type(self) -> str:
        return "summary"


class TitleOutputParser(BaseOutputParser[str]):
    """Cleans a generated conversation title down to a single short line."""

    max_chars: int = 60

    def parse(self, text: str) -> str:
        cleaned = _strip_boilerplate(text).splitlines()[0] if text.strip() else ""
        cleaned = cleaned.strip().rstrip(".")
        return cleaned[: self.max_chars] or "New conversation"

    @property
    def _type(self) -> str:
        return "title"


def summary_parser(max_chars: int) -> SummaryOutputParser:
    return SummaryOutputParser(max_chars=max_chars)


def title_parser() -> TitleOutputParser:
    return TitleOutputParser()


def text_parser() -> StrOutputParser:
    return StrOutputParser()
