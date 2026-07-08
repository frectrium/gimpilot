"""Shared pydantic models for PDB procedure records.

Mirrors the JSONL schema produced by ``pdb-tools/export_pdb.py`` — this is
the one place both ingestion (`rag/ingest.py`) and the agent graph
(`conversation/`) import PDB-record shapes from, so the two never drift apart.
"""

from __future__ import annotations

from pydantic import BaseModel


class PDBArgument(BaseModel):
    name: str
    type: str
    nick: str = ""
    description: str = ""


class PDBProcedure(BaseModel):
    name: str
    proc_type: str
    blurb: str = ""
    help: str = ""
    menu_label: str = ""
    authors: str = ""
    copyright: str = ""
    date: str = ""
    args: list[PDBArgument] = []
    return_values: list[PDBArgument] = []
    deprecated: bool = False
    embedding_text: str = ""


class ScoredProcedure(BaseModel):
    """A `PDBProcedure` as returned from a similarity search."""

    procedure: PDBProcedure
    distance: float
    """Raw vector distance from the query (L2 — lower means more similar)."""
