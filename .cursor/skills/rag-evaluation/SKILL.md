---
name: rag-evaluation
description: >-
  When and how to evaluate retrieval and answer quality for this RAG app.
  Use before claiming better embeddings, chunking, or prompts.
---

# RAG Evaluation

## Before claiming "better"

1. Keep a small gold set (20-50 Q&A grounded in your PDFs)
2. Measure retrieval hit-rate @k (does the right doc/page appear?)
3. Spot-check faithfulness (answer supported by cited chunks?)
4. Change one variable at a time

## Do not

- Swap to `text-embedding-3-large`, hybrid search, or a reranker without metrics
- Treat vibe checks as the only signal

## Minimal harness

Use `scripts/eval_smoke.py` or `tests/` gold fixtures. Deeper RAGAS can wait until the smoke path is green.
