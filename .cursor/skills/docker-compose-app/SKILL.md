---
name: docker-compose-app
description: >-
  Optional Docker Compose packaging for production/deploy of this RAG app.
  Local day-to-day work uses uv, not Docker.
---

# Docker Compose App (deploy only)

## Local vs Docker

- **Local (default):** `uv run rag-kb-ingest` then `uv run rag-kb-chat`
- **Docker:** only for production-style deploys or portable packaging. Do not use Docker for routine local development (resource heavy).

## Deploy contract

1. Ingest: `docker compose run --rm ingest`
2. Chat UI: `docker compose up --build chat`

- Mount `./knowledge-base` and `./data`
- Pass `OPENAI_API_KEY` via `.env`
- Gradio on `7860`

## Do not

- Bake API keys into the image
- Tell contributors that Docker is required to develop locally
- Rely on ephemeral container FS for the vector index
