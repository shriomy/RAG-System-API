"""Logging setup.

Kept dependency-free on purpose: `get_logger` is imported by nearly every
module, so it must not pull in settings or third-party packages.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "hpack",
    "urllib3",
    "qdrant_client",
    "fastembed",
    "onnxruntime",
)


def configure_logging(level: str = "INFO") -> None:
    """Install a single stdout handler on the root logger. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S"))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
