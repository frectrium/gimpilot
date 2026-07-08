# backend

Local server (localhost-only, for now) that does all the reasoning:

- Receives requests from `gimp-plugin` (the user's NL query + image
  context).
- Retrieves relevant PDB procedures from a vector DB (populated from
  `../pdb-tools/export_pdb.py`'s JSONL output).
- Orchestrates the decision via LangGraph, calling Google GenAI.
- Returns a structured plan (procedure name(s) + args) for the plug-in
  to execute — this server never touches the PDB itself.

See the root [README](../README.md) for the full architecture/API plan.

## Setup

```
cp .env.example .env   # fill in GOOGLE_API_KEY (https://aistudio.google.com/apikey)
uv sync
uv run python -m backend.rag ingest          # embeds data/pdb_export.jsonl into data/lancedb/
uv run python -m backend.rag search "change color"
```

`data/pdb_export.jsonl` is `pdb-tools/export_pdb.py`'s output, committed as
a fixture so the backend doesn't need a live GIMP to build its index.
Re-export it (see that script's docstring) after a GIMP upgrade that
changes the PDB, and re-run `ingest` — it's a no-op if the export and
embedding model haven't changed since the last build.

## Run the server

```
uv run uvicorn backend.main:app --reload --port 8765
```

Only `GET /health` and `POST /refresh-conversation` exist so far (the
latter just mints a `thread_id`, no real conversation state yet). On
startup the app calls `ensure_index()`, so it boots with an up-to-date RAG
index (or falls back to whatever's already indexed if that fails, e.g. on
a quota error). `/converse` and the LangGraph agent land with milestones 4/5.

## Run tests

```
uv run pytest --cov=backend --cov-report=term-missing
```

All embedding calls are faked (see `tests/conftest.py`'s `fake_embeddings`
fixture) — no test hits the real Google API or the real committed index.

## Layout

- `src/backend/shared/` — `config.py` (settings loaded from `.env`) and
  `schemas.py` (shared PDB procedure pydantic models).
- `src/backend/rag/` — `ingest.py` (JSONL -> Google embeddings -> LanceDB,
  content-hash gated, resumable), `retrieval.py` (similarity search),
  `__main__.py` (the `ingest`/`search` CLI).
- `src/backend/conversation/` — reserved for the LangGraph retrieve->agent
  graph (milestone 4); just a placeholder today.
- `src/backend/main.py` — FastAPI app.
- `data/pdb_export.jsonl` — committed PDB export (RAG source corpus).
- `data/lancedb/` — committed, pre-built vector index (Google's free-tier
  embedding quota is ~1000 items/day, right at corpus size, so shipping
  the built index means a fresh checkout doesn't have to re-embed
  everything). Re-run `ingest` after changing the JSONL or embedding
  model — it's hash-gated, so it only embeds what's new.
- `tests/unit/` — mirrors `src/backend/` (`shared/`, `rag/`, `conversation/`).
  `tests/integration/` — exercises the real FastAPI app end to end.

Not built yet: the LangGraph agent graph and `/converse`.
