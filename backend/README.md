# backend

Local server (localhost-only, for now) that does all the reasoning:

- Receives requests from `gimp-pilot-plugin` (the user's NL query + image
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

`GET /health`, `POST /refresh-conversation` (mints a `thread_id`), and
`POST /converse` (see the root README's API section for the request/
response shapes) all exist now. On startup the app calls `ensure_index()`,
so it boots with an up-to-date RAG index (or falls back to whatever's
already indexed if that fails, e.g. on a quota error), and builds the
LangGraph conversation graph (`backend.conversation.build_graph`).

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
- `src/backend/conversation/` — the LangGraph retrieve->agent graph.
  `graph.py` (`build_graph`, one retrieve+agent pass per `/converse` call,
  checkpointed via `MemorySaver`), `tools.py` (turns candidate PDB
  procedures into per-turn Gemini tool schemas), `schemas.py` (`/converse`
  request/response pydantic models).
- `src/backend/main.py` — FastAPI app; `/converse` builds a
  `HumanMessage`/`ToolMessage` from the request and invokes the graph.
- `data/pdb_export.jsonl` — committed PDB export (RAG source corpus).
- `data/lancedb/` — committed, pre-built vector index (Google's free-tier
  embedding quota is ~1000 items/day, right at corpus size, so shipping
  the built index means a fresh checkout doesn't have to re-embed
  everything). Re-run `ingest` after changing the JSONL or embedding
  model — it's hash-gated, so it only embeds what's new.
- `tests/unit/` — mirrors `src/backend/` (`shared/`, `rag/`, `conversation/`).
  `tests/integration/` — exercises the real FastAPI app end to end.

See [`../gimp-pilot-plugin/README.md`](../gimp-pilot-plugin/README.md) for
the GIMP-side client that actually drives `/converse` in a loop and executes
the returned tool calls, and the root [README](../README.md) for the full
project overview and quickstart.
