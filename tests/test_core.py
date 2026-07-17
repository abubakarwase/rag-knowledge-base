"""Unit tests that avoid live OpenAI calls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from langchain_core.documents import Document

from rag_kb.cluster import cluster_embeddings
from rag_kb.generate import ABSTAIN_MESSAGE, format_citations_markdown, Citation
from rag_kb.ingest import file_content_hash, semantic_chunk_pages
from rag_kb.store import _sanitize_metadata


def test_semantic_split_text_cuts_on_topic_shift() -> None:
    from rag_kb.chunking import semantic_split_text

    class FakeEmb:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            # First half similar, second half far away
            out = []
            for text in texts:
                if "cats" in text.lower() or "dogs" in text.lower():
                    out.append([1.0, 0.0, 0.0])
                else:
                    out.append([0.0, 1.0, 0.0])
            return out

    text = (
        "Cats are animals. Dogs are animals too. "
        "Quantum physics studies particles. Relativity is another theory."
    )
    pieces = semantic_split_text(text, FakeEmb(), breakpoint_percentile=50.0, buffer_size=0)
    assert len(pieces) >= 2


def test_file_content_hash_stable(tmp_path: Path) -> None:
    path = tmp_path / "a.pdf"
    path.write_bytes(b"%PDF-demo-content")
    assert file_content_hash(path) == file_content_hash(path)


def test_sanitize_metadata_filters_none() -> None:
    clean = _sanitize_metadata({"source": "a.pdf", "page": 1, "skip": None, "flag": True})
    assert clean == {"source": "a.pdf", "page": 1, "flag": True}


def test_semantic_chunk_short_page_keeps_page() -> None:
    pages = [
        Document(
            page_content="Short page about cats and dogs.",
            metadata={"source": "pets.pdf", "page": 2, "page_start": 2, "page_end": 2},
        )
    ]
    chunks = semantic_chunk_pages(pages)
    assert len(chunks) == 1
    assert chunks[0].metadata["page"] == 2
    assert chunks[0].metadata["source"] == "pets.pdf"
    assert "chunk_id" in chunks[0].metadata


def test_cluster_kmeans_labels() -> None:
    rng = np.random.default_rng(0)
    # Two clear blobs
    a = rng.normal(0, 0.1, size=(20, 8))
    b = rng.normal(5, 0.1, size=(20, 8))
    matrix = np.vstack([a, b])
    labels = cluster_embeddings(matrix.tolist(), method="kmeans", n_clusters=2)
    assert labels.shape == (40,)
    assert set(labels.tolist()) <= {0, 1}


def test_format_citations_markdown() -> None:
    text = format_citations_markdown(
        [Citation(source="a.pdf", page=3, snippet="hello world", score=0.9)]
    )
    assert "a.pdf" in text
    assert "page 3" in text


def test_answer_abstains_when_retrieve_empty() -> None:
    from rag_kb.generate import answer_question
    from rag_kb.retrieve import RetrievalResult

    fake = RetrievalResult(
        request_id="abc",
        documents=[],
        scores=[],
        abstain=True,
        latency_s=0.01,
    )
    with patch("rag_kb.generate.retrieve", return_value=fake):
        result = answer_question("What is quantum foam?")
    assert result.abstained is True
    assert result.answer == ABSTAIN_MESSAGE


def test_answer_flags_conflict_from_model_output() -> None:
    from unittest.mock import MagicMock

    from rag_kb.generate import answer_question
    from rag_kb.retrieve import RetrievalResult

    docs = [
        Document(page_content="Deductible: $500", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Deductible: $750", metadata={"source": "b.pdf", "page": 1}),
    ]
    fake_retrieval = RetrievalResult(
        request_id="abc",
        documents=docs,
        scores=[0.8, 0.79],
        abstain=False,
        latency_s=0.01,
    )
    fake_response = MagicMock()
    fake_response.content = (
        "Conflict found: a.pdf p1 states $500 while b.pdf p1 states $750."
    )
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_response
    with (
        patch("rag_kb.generate.retrieve", return_value=fake_retrieval),
        patch("rag_kb.generate.get_chat_model", return_value=fake_llm),
    ):
        result = answer_question("What is the deductible?")
    assert result.conflict is True
    assert result.abstained is False
    assert len(result.citations) == 2


def test_ingest_idempotent_skip(tmp_path: Path) -> None:
    from rag_kb import ingest as ingest_mod

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    with (
        patch.object(ingest_mod, "_already_indexed", return_value=True),
        patch.object(ingest_mod, "delete_by_source") as delete_mock,
        patch.object(ingest_mod, "load_pdf_pages") as load_mock,
    ):
        written, empty, error, skipped = ingest_mod.ingest_pdf(pdf)
    assert written == 0
    assert error is None
    assert skipped is True
    delete_mock.assert_not_called()
    load_mock.assert_not_called()


def test_retrieve_filters_by_threshold() -> None:
    from rag_kb.retrieve import retrieve

    docs = [
        (Document(page_content="a", metadata={"source": "a.pdf", "page": 1}), 0.9),
        (Document(page_content="b", metadata={"source": "b.pdf", "page": 1}), 0.1),
    ]
    with patch("rag_kb.retrieve.query_similar", return_value=docs):
        result = retrieve("q", top_k=5, score_threshold=0.25)
    assert len(result.documents) == 1
    assert result.documents[0].metadata["source"] == "a.pdf"
    assert result.abstain is False
