"""PDF ingest: per-page load, semantic chunk, idempotent Chroma upsert."""

from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from tenacity import retry, stop_after_attempt, wait_exponential

from rag_kb.chunking import semantic_split_text
from rag_kb.config import Settings, get_settings
from rag_kb.logging_utils import log_event
from rag_kb.models import ensure_dirs, get_embeddings
from rag_kb.progress import IngestProgress, write_progress
from rag_kb.store import add_documents, delete_by_source, get_collection

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    scanned: int = 0
    skipped: int = 0
    indexed: int = 0
    chunks_written: int = 0
    empty_pages: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def list_pdfs(root: Path | None = None) -> list[Path]:
    settings = get_settings()
    root = Path(root or settings.knowledge_base_path)
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def _already_indexed(source: str, content_hash: str) -> bool:
    collection = get_collection()
    existing = collection.get(
        where={"$and": [{"source": source}, {"content_hash": content_hash}]},
        limit=1,
        include=[],
    )
    return bool(existing.get("ids"))


def load_pdf_pages(path: Path) -> tuple[list[Document], int]:
    """Load one Document per non-empty page (1-indexed). Returns (pages, total_pages)."""
    reader = PdfReader(str(path))
    pages: list[Document] = []
    source = path.name
    total_pages = len(reader.pages)
    for index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        page_number = index + 1
        if not text:
            continue
        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page_number,
                    "page_start": page_number,
                    "page_end": page_number,
                },
            )
        )
    return pages, total_pages


def semantic_chunk_pages(
    pages: list[Document],
    settings: Settings | None = None,
) -> list[Document]:
    """Semantic-chunk within each page so page attribution stays exact."""
    settings = settings or get_settings()
    if not pages:
        return []

    embeddings = None

    def _embeddings():
        nonlocal embeddings
        if embeddings is None:
            embeddings = get_embeddings()
        return embeddings

    chunks: list[Document] = []
    for page_doc in pages:
        # Very short pages: keep as a single chunk without paying for breakpoints.
        if len(page_doc.page_content.split()) < 40:
            pieces = [page_doc.page_content]
        else:
            try:
                pieces = semantic_split_text(
                    page_doc.page_content,
                    _embeddings(),
                    breakpoint_percentile=settings.semantic_breakpoint_threshold,
                    buffer_size=settings.semantic_buffer_size,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "semantic_chunk_fallback source=%s page=%s error=%s",
                    page_doc.metadata.get("source"),
                    page_doc.metadata.get("page"),
                    exc,
                )
                pieces = [page_doc.page_content]

        for order, text in enumerate(pieces):
            text = text.strip()
            if not text:
                continue
            source = str(page_doc.metadata["source"])
            page = int(page_doc.metadata["page"])
            chunk_id = hashlib.sha256(f"{source}:{page}:{order}:{text}".encode()).hexdigest()[:32]
            meta = dict(page_doc.metadata)
            meta["chunk_id"] = chunk_id
            meta["chunk_index"] = order
            chunks.append(Document(page_content=text, metadata=meta))
    return chunks


@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(5))
def _upsert_with_retry(chunks: list[Document]) -> int:
    return add_documents(chunks)


def ingest_pdf(
    path: Path, settings: Settings | None = None
) -> tuple[int, int, str | None, bool]:
    """
    Index one PDF. Returns (chunks_written, empty_pages, error, skipped).
    Skips when content hash already present.
    """
    settings = settings or get_settings()
    source = path.name
    content_hash = file_content_hash(path)

    if _already_indexed(source, content_hash):
        return 0, 0, None, True

    # Replace prior vectors for this source (content changed or first index).
    delete_by_source(source)

    pages, total_pages = load_pdf_pages(path)
    empty_pages = max(0, total_pages - len(pages))
    if not pages:
        return 0, empty_pages, f"{source}: no extractable text (scan/OCR?)", False

    chunks = semantic_chunk_pages(pages, settings)
    for chunk in chunks:
        chunk.metadata["content_hash"] = content_hash
        chunk.metadata["source_path"] = str(path)

    written = _upsert_with_retry(chunks)
    return written, empty_pages, None, False


def ingest_knowledge_base(
    root: Path | None = None,
    settings: Settings | None = None,
) -> IngestReport:
    """Scan knowledge-base, index new/changed PDFs with bounded parallelism."""
    settings = settings or get_settings()
    ensure_dirs(settings)
    started = time.perf_counter()
    report = IngestReport()
    pdfs = list_pdfs(root)
    report.scanned = len(pdfs)

    if not pdfs:
        report.duration_s = time.perf_counter() - started
        log_event(logger, "ingest_complete", **report.__dict__)
        return report

    workers = max(1, settings.ingest_max_workers)
    progress = IngestProgress(total_files=len(pdfs))

    def _work(pdf: Path) -> tuple[Path, int, int, str | None, bool]:
        written, empty_pages, error, skipped = ingest_pdf(pdf, settings)
        return pdf, written, empty_pages, error, skipped

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_work, pdf): pdf for pdf in pdfs}
        for future in as_completed(futures):
            try:
                pdf, written, empty_pages, error, skipped = future.result()
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{futures[future].name}: {exc}")
                progress.completed_files += 1
                write_progress(progress)
                continue
            report.empty_pages += empty_pages
            progress.completed_files += 1
            progress.last_source = pdf.name
            if skipped:
                report.skipped += 1
                progress.skipped_files += 1
                write_progress(progress)
                continue
            if error:
                report.errors.append(error)
                write_progress(progress)
                continue
            if written:
                report.indexed += 1
                report.chunks_written += written
                progress.chunks_written += written
            write_progress(progress)

    report.duration_s = round(time.perf_counter() - started, 3)
    log_event(
        logger,
        "ingest_complete",
        scanned=report.scanned,
        skipped=report.skipped,
        indexed=report.indexed,
        chunks_written=report.chunks_written,
        empty_pages=report.empty_pages,
        errors=len(report.errors),
        duration_s=report.duration_s,
    )
    return report


def main() -> None:
    """CLI: load PDFs from knowledge-base into Chroma."""
    import sys

    from rag_kb.logging_utils import setup_logging
    from rag_kb.models import ensure_dirs
    from rag_kb.visualize import index_stats

    setup_logging()
    settings = get_settings()
    ensure_dirs(settings)

    if not settings.has_openai_key:
        print("ERROR: Set OPENAI_API_KEY in .env before ingesting.", file=sys.stderr)
        raise SystemExit(1)

    # Fail fast on dead keys / exhausted quota instead of retrying per page.
    try:
        get_embeddings().embed_query("preflight")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: OpenAI embedding preflight failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Knowledge base: {settings.knowledge_base_path}")
    print(f"Chroma path:    {settings.chroma_path}")
    print("Indexing PDFs (unchanged files are skipped)...")
    report = ingest_knowledge_base()

    print()
    print(f"Scanned:          {report.scanned}")
    print(f"Skipped:          {report.skipped}")
    print(f"Newly indexed:    {report.indexed}")
    print(f"Chunks written:   {report.chunks_written}")
    print(f"Empty pages:      {report.empty_pages}")
    print(f"Duration (s):     {report.duration_s}")
    if report.errors:
        print("Errors:")
        for err in report.errors[:50]:
            print(f"  - {err}")
        raise SystemExit(2)

    stats = index_stats()
    print(f"Vectors in Chroma: {stats['vector_count']}")
    print("Done. Start the chatbot with: uv run rag-kb-chat")
    print("(Local only: no Docker required.)")


if __name__ == "__main__":
    main()
