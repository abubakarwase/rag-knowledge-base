"""Progress helpers and scale notes for large ingest runs."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_kb.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class IngestProgress:
    total_files: int = 0
    completed_files: int = 0
    skipped_files: int = 0
    chunks_written: int = 0
    last_source: str = ""

    @property
    def fraction(self) -> float:
        if self.total_files <= 0:
            return 0.0
        return self.completed_files / self.total_files


def progress_path() -> Path:
    settings = get_settings()
    return Path(settings.chroma_path).parent / "ingest_progress.json"


def write_progress(progress: IngestProgress) -> None:
    path = progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(progress), indent=2), encoding="utf-8")
    logger.info(
        "event=ingest_progress completed=%s total=%s skipped=%s chunks=%s last=%r",
        progress.completed_files,
        progress.total_files,
        progress.skipped_files,
        progress.chunks_written,
        progress.last_source,
    )


def read_progress() -> IngestProgress | None:
    path = progress_path()
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return IngestProgress(**data)
