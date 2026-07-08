"""CLI for building/refreshing the vector index and querying it.

    uv run python -m backend.rag ingest
    uv run python -m backend.rag search "change color"
"""

from __future__ import annotations

import argparse

from backend.rag.ingest import ensure_index, get_table
from backend.rag.retrieval import search
from backend.shared.config import get_settings


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("ingest", help="Build/refresh the vector index")

    search_parser = subparsers.add_parser("search", help="Similarity-search the index")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()

    if args.command == "ingest":
        settings = get_settings()
        try:
            table = ensure_index(settings)
            print(f"Index ready: {table.count_rows()} procedures.")
        except Exception as error:
            table = get_table(settings)
            print(
                f"Ingestion stopped early ({error}).\n"
                f"{table.count_rows()} procedures indexed so far and searchable now "
                f"— rerun `ingest` later to pick up where it left off."
            )
    elif args.command == "search":
        results = search(args.query, top_k=args.top_k)
        for r in results:
            print(f"{r.distance:.4f}  {r.procedure.name}  — {r.procedure.blurb[:80]}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
