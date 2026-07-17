"""
Minimal retrieval smoke eval against a JSONL gold set.

Gold file format (one JSON object per line):
  {"question": "...", "expected_source": "file.pdf", "expected_page": 1}

Usage:
  python -m rag_kb.eval_smoke --gold eval/gold.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from rag_kb.logging_utils import log_event, setup_logging
from rag_kb.retrieve import retrieve

logger = logging.getLogger(__name__)


def load_gold(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(gold_path: Path, top_k: int = 5) -> dict:
    rows = load_gold(gold_path)
    hits = 0
    for row in rows:
        result = retrieve(row["question"], top_k=top_k, score_threshold=0.0)
        expected_source = row.get("expected_source")
        expected_page = row.get("expected_page")
        matched = False
        for doc in result.documents:
            source_ok = expected_source is None or doc.metadata.get("source") == expected_source
            page_ok = expected_page is None or int(doc.metadata.get("page", -1)) == int(
                expected_page
            )
            if source_ok and page_ok:
                matched = True
                break
        if matched:
            hits += 1
    total = len(rows) or 1
    report = {
        "total": len(rows),
        "hits": hits,
        "hit_rate_at_k": round(hits / total, 4),
        "top_k": top_k,
    }
    log_event(logger, "eval_smoke", **report)
    return report


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Smoke eval for retrieval hit-rate")
    parser.add_argument("--gold", type=Path, default=Path("eval/gold.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if not args.gold.exists():
        raise SystemExit(f"Gold file not found: {args.gold}")
    report = evaluate(args.gold, top_k=args.top_k)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
