"""Retrieval with score threshold and exact-identifier filtering."""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field

from langchain_core.documents import Document

from rag_kb.config import get_settings
from rag_kb.logging_utils import log_event
from rag_kb.store import query_similar

logger = logging.getLogger(__name__)

# Policy-style identifiers (e.g. AUTO-385891-52, CYBER-481442-63). Dense
# embeddings do not discriminate exact identifiers between near-duplicate
# documents, so identifier queries get a document-content filter first.
_IDENTIFIER = re.compile(r"\b[A-Z]{2,8}-\d{3,8}-\d{1,4}\b", re.IGNORECASE)


def extract_identifiers(query: str) -> list[str]:
    """Uppercased identifier tokens found in the query, order-preserving."""
    seen: list[str] = []
    for match in _IDENTIFIER.findall(query):
        token = match.upper()
        if token not in seen:
            seen.append(token)
    return seen


@dataclass
class RetrievalResult:
    request_id: str
    documents: list[Document]
    scores: list[float]
    abstain: bool
    latency_s: float
    identifiers: list[str] = field(default_factory=list)
    filter_used: bool = False


def retrieve(
    query: str,
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> RetrievalResult:
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.retrieval_top_k
    score_threshold = (
        score_threshold
        if score_threshold is not None
        else settings.retrieval_score_threshold
    )
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()

    identifiers = extract_identifiers(query)
    filter_used = False
    hits: list[tuple[Document, float]] = []

    if identifiers:
        # Query per identifier so multi-policy questions get chunks from each
        # document; scores share one query embedding so they merge cleanly.
        merged: dict[str, tuple[Document, float]] = {}
        for ident in identifiers[:3]:
            for doc, score in query_similar(
                query,
                top_k=top_k,
                where_document={"$contains": ident},
            ):
                key = str(doc.metadata.get("chunk_id", id(doc)))
                if key not in merged or score > merged[key][1]:
                    merged[key] = (doc, score)
        hits = sorted(merged.values(), key=lambda pair: pair[1], reverse=True)[:top_k]
        filter_used = bool(hits)

    if not hits:
        # No identifier in the query, or the filter matched nothing
        # (e.g. a mistyped policy number): fall back to plain dense search.
        hits = query_similar(query, top_k=top_k)

    filtered = [(doc, score) for doc, score in hits if score >= score_threshold]
    latency = round(time.perf_counter() - started, 4)
    abstain = len(filtered) == 0

    sources = [
        f"{d.metadata.get('source')}:p{d.metadata.get('page')}"
        for d, _ in filtered
    ]
    log_event(
        logger,
        "retrieve",
        request_id=request_id,
        top_k=top_k,
        threshold=score_threshold,
        identifiers=identifiers,
        filter_used=filter_used,
        hits=len(hits),
        kept=len(filtered),
        abstain=abstain,
        sources=sources,
        latency_s=latency,
    )

    return RetrievalResult(
        request_id=request_id,
        documents=[d for d, _ in filtered],
        scores=[s for _, s in filtered],
        abstain=abstain,
        latency_s=latency,
        identifiers=identifiers,
        filter_used=filter_used,
    )
