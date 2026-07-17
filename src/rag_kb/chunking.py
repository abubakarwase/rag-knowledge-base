"""Semantic text splitting via embedding similarity breakpoints."""

from __future__ import annotations

import re

import numpy as np
from langchain_core.embeddings import Embeddings


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def semantic_split_text(
    text: str,
    embeddings: Embeddings,
    *,
    breakpoint_percentile: float = 95.0,
    buffer_size: int = 1,
) -> list[str]:
    """
    Split text into semantically coherent chunks.

    Embeds sentences, finds distances between consecutive groups, and cuts
    where distance exceeds a percentile threshold (same idea as LangChain's
    SemanticChunker, without the experimental package).
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    # Group sentences with a small sliding buffer for stabler embeddings.
    groups: list[str] = []
    for index, sentence in enumerate(sentences):
        start = max(0, index - buffer_size)
        end = min(len(sentences), index + buffer_size + 1)
        groups.append(" ".join(sentences[start:end]))

    vectors = np.asarray(embeddings.embed_documents(groups), dtype=np.float64)
    # Cosine distance between consecutive group embeddings
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    unit = vectors / norms
    distances = 1.0 - np.sum(unit[1:] * unit[:-1], axis=1)

    if distances.size == 0:
        return [" ".join(sentences)]

    threshold = float(np.percentile(distances, breakpoint_percentile))
    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    for index, distance in enumerate(distances, start=1):
        if distance > threshold and current:
            chunks.append(" ".join(current))
            current = [sentences[index]]
        else:
            current.append(sentences[index])
    if current:
        chunks.append(" ".join(current))
    return chunks
