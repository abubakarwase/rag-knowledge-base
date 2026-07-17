FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml README.md ./
COPY src ./src

RUN uv pip install --system .

RUN mkdir -p /app/knowledge-base /app/data/chroma

ENV KNOWLEDGE_BASE_PATH=/app/knowledge-base \
    CHROMA_PATH=/app/data/chroma \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "-m", "rag_kb.app"]
