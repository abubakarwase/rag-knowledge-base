"""Structured logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    parts = [f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={value!r}")
    logger.info(" ".join(parts))
