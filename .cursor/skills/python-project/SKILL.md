---
name: python-project
description: >-
  Python package layout, typing, uv, env, and logging conventions for
  rag-knowledge-base. Use when adding modules, deps, or tests.
---

# Python Project

## Layout

- Package: `src/rag_kb/`
- Tests: `tests/`
- Manage deps with `uv` and `pyproject.toml`
- Secrets only via `.env` (gitignored); ship `.env.example`

## Standards

- Type hints on public functions
- Prefer small modules over god-files
- Structured logging via the stdlib `logging` module (JSON-ish key=value messages OK)
- No secrets in git, logs, or screenshots
- Run tests with `pytest`
- **Local run path is `uv`, not Docker.** Docker is optional packaging for production/deploy.
