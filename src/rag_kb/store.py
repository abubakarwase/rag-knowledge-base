"""Embedded persistent Chroma vector store helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_kb.config import Settings, get_settings
from rag_kb.models import ensure_dirs, get_embeddings


# One client/collection per path per process: concurrent PersistentClient
# construction from worker threads races inside Chroma's rust bindings.
@lru_cache(maxsize=4)
def _cached_client(path: str) -> chromadb.PersistentClient:
    Path(path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=path)


@lru_cache(maxsize=8)
def _cached_collection(name: str, path: str) -> Collection:
    return _cached_client(path).get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_chroma_client(persist_directory: Path | None = None) -> chromadb.PersistentClient:
    settings = get_settings()
    return _cached_client(str(Path(persist_directory or settings.chroma_path)))


def get_collection(
    name: str | None = None,
    persist_directory: Path | None = None,
) -> Collection:
    settings = get_settings()
    return _cached_collection(
        name or settings.chroma_collection,
        str(Path(persist_directory or settings.chroma_path)),
    )


def collection_count(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    ensure_dirs(settings)
    return get_collection().count()


def add_documents(
    documents: list[Document],
    embeddings: Embeddings | None = None,
    batch_size: int | None = None,
) -> int:
    """Embed and upsert documents. Returns number of vectors written."""
    if not documents:
        return 0

    settings = get_settings()
    embeddings = embeddings or get_embeddings()
    batch_size = batch_size or settings.ingest_embed_batch_size
    collection = get_collection()

    written = 0
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        ids = [str(doc.metadata["chunk_id"]) for doc in batch]
        texts = [doc.page_content for doc in batch]
        metadatas = [_sanitize_metadata(doc.metadata) for doc in batch]
        vectors = embeddings.embed_documents(texts)
        collection.upsert(ids=ids, documents=texts, embeddings=vectors, metadatas=metadatas)
        written += len(batch)
    return written


def delete_by_source(source: str) -> int:
    collection = get_collection()
    existing = collection.get(where={"source": source}, include=[])
    ids = existing.get("ids") or []
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def query_similar(
    query: str,
    top_k: int | None = None,
    embeddings: Embeddings | None = None,
    where_document: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """Return (Document, cosine_similarity) pairs. Higher score is better."""
    settings = get_settings()
    top_k = top_k or settings.retrieval_top_k
    embeddings = embeddings or get_embeddings()
    collection = get_collection()
    if collection.count() == 0:
        return []

    vector = embeddings.embed_query(query)
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        where_document=where_document,
        include=["documents", "metadatas", "distances"],
    )

    docs: list[tuple[Document, float]] = []
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for text, meta, dist in zip(documents, metadatas, distances, strict=False):
        # Cosine distance -> similarity
        similarity = 1.0 - float(dist)
        docs.append((Document(page_content=text or "", metadata=dict(meta or {})), similarity))
    return docs


def fetch_all_embeddings(
    limit: int | None = None,
) -> tuple[list[str], list[list[float]], list[dict[str, Any]], list[str]]:
    """Return ids, embeddings, metadatas, documents (optionally capped).

    Samples ids first, then fetches only the sampled rows, so memory stays
    bounded by `limit` rather than the collection size.
    """
    collection = get_collection()
    if collection.count() == 0:
        return [], [], [], []

    ids = list(collection.get(include=[]).get("ids") or [])
    if limit is not None and len(ids) > limit:
        # Evenly spaced sample for stability across refreshes
        step = len(ids) / limit
        ids = [ids[int(i * step)] for i in range(limit)]

    data = collection.get(ids=ids, include=["embeddings", "metadatas", "documents"])
    ids = list(data.get("ids") or [])
    # Chroma returns embeddings as a numpy array; `or []` on it raises.
    raw_embeddings = data.get("embeddings")
    raw_embeddings = [] if raw_embeddings is None else list(raw_embeddings)
    embeddings = [
        list(map(float, row)) if row is not None and len(row) else []
        for row in raw_embeddings
    ]
    metadatas = [dict(m or {}) for m in (data.get("metadatas") or [])]
    documents = list(data.get("documents") or [])

    # Drop rows without embeddings
    kept = [(i, e, m, d) for i, e, m, d in zip(ids, embeddings, metadatas, documents, strict=False) if e]
    if not kept:
        return [], [], [], []
    ids, embeddings, metadatas, documents = map(list, zip(*kept, strict=False))
    return list(ids), list(embeddings), list(metadatas), list(documents)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean
