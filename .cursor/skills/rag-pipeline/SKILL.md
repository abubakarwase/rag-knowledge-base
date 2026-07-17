---
name: rag-pipeline
description: >-
  Architecture and conventions for this RAG knowledge-base app (ingest,
  retrieve, generate, visualize). Use when changing chunking, embeddings,
  Chroma, citations, abstain behavior, or vector-space viz.
---

# RAG Pipeline

## Boundaries

- `ingest.py`: PDF load (per page), semantic chunk, embed, upsert Chroma
- `retrieve.py`: similarity search + score threshold
- `generate.py`: LLM answer + citations + abstain
- `cluster.py` / `visualize.py`: sampled PCA + cluster colors
- `app.py`: Gradio UI only (thin)

## Hard rules

- English-only corpus and prompts in v1
- Chunk **per page** with semantic breakpoints so every chunk keeps an exact `page`
- Metadata must include: `source`, `page` (or `page_start`/`page_end`), `chunk_id`
- Do not use fixed-size-only chunking without documenting why in the PR/README
- One collection, one embedding model, one chat model unless config says otherwise
- Abstain when no chunk clears the retrieval score threshold
- Chroma is embedded `PersistentClient` under `data/chroma/` (volume-mounted)
- Scale path: embedded Chroma -> Chroma service -> Qdrant (only when measured need)
- At ~400k docs expect millions of chunks; document that honestly

## Config keys (env)

- `OPENAI_API_KEY`
- `EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `CHAT_MODEL` (default `gpt-4o-mini`)
- `CHROMA_PATH`, `KNOWLEDGE_BASE_PATH`
- `RETRIEVAL_TOP_K`, `RETRIEVAL_SCORE_THRESHOLD`
- `SEMANTIC_BREAKPOINT_THRESHOLD`
