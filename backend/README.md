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

## Layout

- `src/backend/config.py` — settings loaded from `.env`.
- `src/backend/schemas.py` — shared PDB procedure pydantic models.
- `src/backend/rag.py` — JSONL -> Google embeddings -> LanceDB, and
  similarity search over it.
- `data/pdb_export.jsonl` — committed PDB export (RAG source corpus).
- `data/lancedb/` — committed, pre-built vector index (Google's free-tier
  embedding quota is ~1000 items/day, right at corpus size, so shipping
  the built index means a fresh checkout doesn't have to re-embed
  everything). Re-run `ingest` after changing the JSONL or embedding
  model — it's hash-gated, so it only embeds what's new.

Not built yet: the FastAPI app and the LangGraph agent graph
(`/converse`, `/refresh-conversation`).
