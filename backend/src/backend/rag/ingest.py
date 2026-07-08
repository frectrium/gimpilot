"""Embed `pdb-tools/export_pdb.py`'s JSONL output into a local LanceDB table.

Ingestion is content-hash gated: re-running it is a no-op unless the JSONL
export or the embedding model has changed, so a normal backend startup never
re-embeds ~1000 procedures against Google's API for nothing.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import lancedb
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.shared.config import Settings, get_settings
from backend.shared.schemas import PDBProcedure

TABLE_NAME = "procedures"
MANIFEST_NAME = "ingest_manifest.json"
PARTIAL_MANIFEST_NAME = "ingest_partial.json"

# Google AI Studio's free tier throttles embedContent to ~100 items/minute.
# The client library batches internally but fires all batches back-to-back,
# blowing through that in seconds — so we pace it ourselves, one chunk per
# minute-ish window, rather than depending on retry/backoff to survive 429s.
EMBED_CHUNK_SIZE = 90
EMBED_CHUNK_PAUSE_SECONDS = 62


def _embeddings_client(settings: Settings) -> GoogleGenerativeAIEmbeddings:  # pragma: no cover
    # Real network-calling client — tests patch this out entirely via the
    # `fake_embeddings`/`patch_embeddings_client` fixtures, so this body never
    # runs under test.
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
    )


def load_procedures(jsonl_path: Path) -> list[PDBProcedure]:
    procedures = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                procedures.append(PDBProcedure.model_validate_json(line))
    return procedures


def _source_fingerprint(jsonl_path: Path, embedding_model: str) -> str:
    """Hash of everything that should invalidate the index if it changes."""
    digest = hashlib.sha256()
    digest.update(jsonl_path.read_bytes())
    digest.update(embedding_model.encode("utf-8"))
    return digest.hexdigest()


def _manifest_path(settings: Settings) -> Path:
    return settings.lancedb_path / MANIFEST_NAME


def _read_manifest(settings: Settings) -> dict | None:
    path = _manifest_path(settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_manifest(settings: Settings, fingerprint: str, count: int) -> None:
    _manifest_path(settings).write_text(
        json.dumps({"source_hash": fingerprint, "count": count})
    )


def _partial_path(settings: Settings) -> Path:
    return settings.lancedb_path / PARTIAL_MANIFEST_NAME


def _read_partial(settings: Settings) -> dict | None:
    path = _partial_path(settings)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_partial(settings: Settings, fingerprint: str, done_names: set[str]) -> None:
    _partial_path(settings).write_text(
        json.dumps({"source_hash": fingerprint, "done_names": sorted(done_names)})
    )


def _clear_partial(settings: Settings) -> None:
    _partial_path(settings).unlink(missing_ok=True)


def _table_exists(db: lancedb.DBConnection) -> bool:
    return TABLE_NAME in db.list_tables().tables


def _procedure_row(procedure: PDBProcedure, vector: list[float]) -> dict:
    row = procedure.model_dump()
    row["args"] = json.dumps(row["args"])
    row["return_values"] = json.dumps(row["return_values"])
    row["vector"] = vector
    return row


def ensure_index(settings: Settings | None = None, force: bool = False) -> lancedb.table.Table:
    """Open the procedures table, (re)building it first if it's stale.

    Stale = no table yet, or the JSONL export / embedding model changed
    since the last build. Otherwise this is just a fast local table open —
    no embedding calls made.
    """
    settings = settings or get_settings()
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)

    fingerprint = _source_fingerprint(settings.pdb_export_path, settings.embedding_model)
    manifest = _read_manifest(settings)
    db = lancedb.connect(settings.lancedb_path)

    if (
        not force
        and manifest is not None
        and manifest.get("source_hash") == fingerprint
        and _table_exists(db)
    ):
        return db.open_table(TABLE_NAME)

    return build_index(settings, db=db, fingerprint=fingerprint, force=force)


def get_table(settings: Settings | None = None) -> lancedb.table.Table:
    """Open the procedures table for querying, without trying to finish an
    interrupted ingestion.

    `ensure_index` (used by `ingest`) will try to top up a partial index —
    which re-hits the API and re-raises the same quota error if it's still
    rate-limited. Searching shouldn't be blocked on that: if a same-source
    table already has *some* rows (a prior `ingest` got partway through
    before a 429), just search those. Only builds from scratch if there's no
    matching table at all yet.
    """
    settings = settings or get_settings()
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)

    fingerprint = _source_fingerprint(settings.pdb_export_path, settings.embedding_model)
    db = lancedb.connect(settings.lancedb_path)

    manifest = _read_manifest(settings)
    partial = _read_partial(settings)
    matches_current_source = (manifest is not None and manifest.get("source_hash") == fingerprint) or (
        partial is not None and partial.get("source_hash") == fingerprint
    )

    if matches_current_source and _table_exists(db):
        return db.open_table(TABLE_NAME)

    return build_index(settings, db=db, fingerprint=fingerprint)


def build_index(
    settings: Settings | None = None,
    db: lancedb.DBConnection | None = None,
    fingerprint: str | None = None,
    force: bool = False,
) -> lancedb.table.Table:
    """Embed the JSONL export and (re)build the table, resumably.

    Google AI Studio's free tier caps embedContent at ~1000 items/day, which
    is right at the size of the whole PDB corpus — a single run can get cut
    off by a 429 before finishing. So this writes each embedded chunk to the
    LanceDB table immediately (rather than batching everything in memory
    until the end) and records progress in a partial-ingest manifest keyed
    by procedure name. Re-running after a quota reset (or a crash) skips
    whatever's already embedded and only pays for the remainder.
    """
    settings = settings or get_settings()
    settings.lancedb_path.mkdir(parents=True, exist_ok=True)
    db = db or lancedb.connect(settings.lancedb_path)
    fingerprint = fingerprint or _source_fingerprint(
        settings.pdb_export_path, settings.embedding_model
    )

    procedures = load_procedures(settings.pdb_export_path)

    partial = None if force else _read_partial(settings)
    resuming = partial is not None and partial.get("source_hash") == fingerprint
    if resuming and _table_exists(db):
        table = db.open_table(TABLE_NAME)
        done_names = set(partial["done_names"])
    else:
        if _table_exists(db):
            db.drop_table(TABLE_NAME)
        _clear_partial(settings)
        table = None
        done_names = set()

    remaining = [p for p in procedures if p.name not in done_names]
    if remaining:
        embeddings = _embeddings_client(settings)
        chunks = [
            remaining[i : i + EMBED_CHUNK_SIZE]
            for i in range(0, len(remaining), EMBED_CHUNK_SIZE)
        ]
        for i, chunk in enumerate(chunks):
            print(f"embedding {len(done_names) + len(chunk)}/{len(procedures)}...")
            vectors = embeddings.embed_documents(
                [p.embedding_text for p in chunk], batch_size=len(chunk)
            )
            rows = [_procedure_row(p, v) for p, v in zip(chunk, vectors)]
            if table is None:
                table = db.create_table(TABLE_NAME, data=rows)
            else:
                table.add(rows)
            done_names.update(p.name for p in chunk)
            _write_partial(settings, fingerprint, done_names)
            if i < len(chunks) - 1:
                time.sleep(EMBED_CHUNK_PAUSE_SECONDS)

    _write_manifest(settings, fingerprint, len(procedures))
    _clear_partial(settings)
    return table
