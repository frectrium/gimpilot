"""Similarity search over the LanceDB procedures table built by `ingest.py`."""

from __future__ import annotations

import json

from backend.rag.ingest import _embeddings_client, get_table
from backend.shared.config import Settings, get_settings
from backend.shared.schemas import PDBProcedure, ScoredProcedure


def _row_to_scored_procedure(row: dict) -> ScoredProcedure:
    row = dict(row)
    distance = row.pop("_distance")
    row.pop("vector", None)
    row["args"] = json.loads(row["args"])
    row["return_values"] = json.loads(row["return_values"])
    return ScoredProcedure(procedure=PDBProcedure.model_validate(row), distance=distance)


def search(
    query: str, top_k: int | None = None, settings: Settings | None = None
) -> list[ScoredProcedure]:
    settings = settings or get_settings()
    top_k = top_k or settings.rag_top_k

    table = get_table(settings)
    embeddings = _embeddings_client(settings)
    query_vector = embeddings.embed_query(query)

    rows = table.search(query_vector).limit(top_k).to_list()
    return [_row_to_scored_procedure(row) for row in rows]
