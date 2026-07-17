"""Integration tests against a real temp Chroma store (no OpenAI calls).

These exercise the store -> retrieve -> viz path with fake embeddings, the
layer the mock-only unit tests cannot see (e.g. Chroma returning numpy arrays).
"""

from __future__ import annotations

import hashlib

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag_kb import store
from rag_kb.config import get_settings


class FakeEmbeddings(Embeddings):
    """Deterministic, cheap embeddings keyed on token overlap."""

    dim = 32

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode()).digest()
            for i in range(self.dim):
                vec[i] += (digest[i % len(digest)] - 128) / 128.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "test_kb")
    get_settings.cache_clear()
    yield FakeEmbeddings()
    get_settings.cache_clear()


def _docs() -> list[Document]:
    rows = [
        ("Policy AUTO-111111-11 total premium is $1,000.", "a.pdf", 1),
        ("Policy AUTO-222222-22 total premium is $2,000.", "b.pdf", 1),
        ("Collision coverage pays for accidental loss to your auto.", "a.pdf", 2),
        ("The named insured is Jordan Example.", "b.pdf", 2),
    ]
    return [
        Document(
            page_content=text,
            metadata={
                "source": source,
                "page": page,
                "chunk_id": hashlib.sha256(text.encode()).hexdigest()[:32],
            },
        )
        for text, source, page in rows
    ]


def test_add_query_fetch_roundtrip(temp_store: FakeEmbeddings) -> None:
    written = store.add_documents(_docs(), embeddings=temp_store)
    assert written == 4
    assert store.collection_count() == 4

    hits = store.query_similar("total premium", top_k=2, embeddings=temp_store)
    assert len(hits) == 2
    assert all(isinstance(score, float) for _, score in hits)

    # Regression: fetch_all_embeddings must survive Chroma's numpy arrays
    ids, embeddings, metadatas, documents = store.fetch_all_embeddings(limit=3)
    assert len(ids) == 3
    assert len(embeddings[0]) == FakeEmbeddings.dim
    assert all("source" in m for m in metadatas)


def test_where_document_filter_isolates_policy(temp_store: FakeEmbeddings) -> None:
    store.add_documents(_docs(), embeddings=temp_store)
    hits = store.query_similar(
        "total premium of AUTO-222222-22",
        top_k=4,
        embeddings=temp_store,
        where_document={"$contains": "AUTO-222222-22"},
    )
    assert len(hits) == 1
    assert hits[0][0].metadata["source"] == "b.pdf"


def test_build_vector_plots_runs_on_real_store(temp_store: FakeEmbeddings) -> None:
    from rag_kb.visualize import build_vector_plots

    store.add_documents(_docs(), embeddings=temp_store)
    fig2, fig3, status = build_vector_plots(sample_size=4, n_clusters=2, method="kmeans")
    assert "Plotted 4 chunks" in status
    assert fig2.data and fig3.data


def test_delete_by_source_removes_only_that_source(temp_store: FakeEmbeddings) -> None:
    store.add_documents(_docs(), embeddings=temp_store)
    removed = store.delete_by_source("a.pdf")
    assert removed == 2
    assert store.collection_count() == 2
