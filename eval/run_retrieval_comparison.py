#!/usr/bin/env python3
"""Compares the backend's real RAG (semantic) retrieval against a naive
keyword/substring search over the same PDB corpus — for both recall@k and
latency.

This is **not** a comparison against a human manually browsing GIMP's menus
(no such timing data exists, and fabricating one would be dishonest) — it's
a comparison between two computational search strategies over the same
procedure corpus, which is something we can actually measure.

Costs real Google embedding API calls (one per case, via the same
`backend.rag.search` the live backend uses). Run via the backend's venv:

    cd backend && uv run python ../eval/run_retrieval_comparison.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from cases import CASES

from backend.rag import search as rag_search
from backend.shared.config import get_settings
from backend.shared.schemas import PDBProcedure

RESULTS_DIR = Path(__file__).parent / "results"
TOP_K = 8


def load_procedures(path: Path) -> list[PDBProcedure]:
    procedures = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                procedures.append(PDBProcedure.model_validate_json(line))
    return procedures


def naive_keyword_search(query: str, procedures: list[PDBProcedure], top_k: int) -> list[str]:
    """Score each procedure by how many query words appear in its
    embedding text, and return the top-k names — a simple linear-scan
    keyword baseline with no vectors/embeddings involved.
    """
    words = [w for w in query.lower().replace(",", " ").replace(".", " ").split() if w]
    scored = []
    for proc in procedures:
        haystack = proc.embedding_text.lower()
        score = sum(1 for w in words if w in haystack)
        scored.append((score, proc.name))
    scored.sort(key=lambda pair: -pair[0])
    return [name for _score, name in scored[:top_k]]


def run() -> dict:
    settings = get_settings()
    procedures = load_procedures(settings.pdb_export_path)

    rows = []
    for case in CASES:
        if not case.expected_procedures:
            continue  # no_tool_call cases have nothing to retrieve for
        expected = case.expected_procedures[0]

        start = time.monotonic()
        rag_results = rag_search(case.message, top_k=TOP_K, settings=settings)
        rag_latency = time.monotonic() - start
        rag_names = [r.procedure.name for r in rag_results]
        rag_hit = any(name in rag_names for name in expected)

        start = time.monotonic()
        naive_names = naive_keyword_search(case.message, procedures, TOP_K)
        naive_latency = time.monotonic() - start
        naive_hit = any(name in naive_names for name in expected)

        rows.append(
            {
                "case": case.id,
                "message": case.message,
                "expected": expected,
                "rag_hit": rag_hit,
                "rag_latency_s": round(rag_latency, 3),
                "rag_top_k": rag_names,
                "naive_hit": naive_hit,
                "naive_latency_s": round(naive_latency, 4),
                "naive_top_k": naive_names,
            }
        )

    total = len(rows)
    rag_hits = sum(1 for r in rows if r["rag_hit"])
    naive_hits = sum(1 for r in rows if r["naive_hit"])
    rag_latencies = [r["rag_latency_s"] for r in rows]
    naive_latencies = [r["naive_latency_s"] for r in rows]

    summary = {
        "total_queries": total,
        "rag_recall_at_k": round(rag_hits / total, 3) if total else None,
        "naive_recall_at_k": round(naive_hits / total, 3) if total else None,
        "rag_avg_latency_s": round(sum(rag_latencies) / total, 3) if total else None,
        "naive_avg_latency_s": round(sum(naive_latencies) / total, 4) if total else None,
        "top_k": TOP_K,
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    result = run()

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"retrieval_comparison_{timestamp}.json"
    output_path.write_text(json.dumps(result, indent=2))

    print(json.dumps(result["summary"], indent=2))
    print(f"\nFull results written to {output_path}")


if __name__ == "__main__":
    main()
