# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A GIMP plugin that uses RAG + an LLM agent (LangGraph) to turn natural
language requests into GIMP PDB procedure calls. Three components:

- **`gimp-pilot-plugin/`** — GIMP-side plug-in (Python-fu). Runs inside
  GIMP, presents the chat UI, and is the *only* thing with a live PDB
  handle (`Gimp.get_pdb().lookup_procedure(...)`). Not built yet beyond a
  README — see milestone 6 below.
- **`backend/`** — local HTTP server (own venv/port, `uv`-managed). Owns the
  RAG index over PDB procedures and (eventually) the LangGraph agent.
  Never touches the PDB directly — only ever returns *proposed* procedure
  calls for the plug-in to execute. This is where almost all current work
  lives.
- **`pdb-tools/export_pdb.py`** — run once per GIMP version, inside GIMP's
  headless batch interpreter, to dump the full PDB to JSONL. Feeds the
  backend's vector store. `gimp_mcp_bridge.py` / `mcp_client_example.py` in
  that dir are leftovers from an earlier standalone-bridge design,
  superseded now — kept only as reference for `to_pdb_value`/`from_pdb_value`
  type-coercion logic that still needs porting into `gimp-pilot-plugin`'s
  executor before those two files can be deleted.

Full architecture/API design (the `/converse` request/response shapes, why
the tool-call loop is split across one HTTP round trip per PDB call): see
the root [README.md](README.md).

## Commands (all run from `backend/`)

```
cp .env.example .env && $EDITOR .env   # fill in GOOGLE_API_KEY

uv sync                                          # install deps (+ dev group: pytest, pytest-cov, httpx)
uv run uvicorn backend.main:app --reload --port 8765   # run the server
uv run python -m backend.rag ingest              # (re)build the vector index — no-op if unchanged
uv run python -m backend.rag search "change color"     # similarity-search the index

uv run pytest --cov=backend --cov-report=term-missing --cov-fail-under=95   # full suite + coverage gate
uv run pytest tests/unit/rag/test_ingest.py::test_build_index_embeds_all_procedures_in_chunks  # single test
```

There is no lint/format command configured yet.

## Backend architecture

```
pdb-tools/export_pdb.py  --(jsonl)-->  backend ingestion  --(embed, cache)-->  LanceDB (backend/data/)
                                                                                      |
gimp-pilot-plugin  --HTTP-->  FastAPI (/converse, /refresh-conversation)  --> LangGraph app
                                                                                      |
                                                                     retrieve node -> agent node (Gemini)
```

Package layout under `src/backend/` (mirrored by `tests/unit/`):

- **`shared/`** — `config.py` (`Settings`, a `pydantic-settings` model read
  from `backend/.env`; every function that needs config takes a `Settings`
  object explicitly rather than reaching for the global, so tests can
  inject an isolated one) and `schemas.py` (`PDBProcedure`/`PDBArgument`/
  `ScoredProcedure` — the one place both ingestion and, later, the agent
  graph import PDB-record shapes from, so they can't drift apart).
- **`rag/`** — `ingest.py` (JSONL -> Google embeddings -> LanceDB),
  `retrieval.py` (`search()`), `__main__.py` (the `ingest`/`search` CLI, run
  via `python -m backend.rag`). `rag/__init__.py` re-exports the public
  surface (`ensure_index`, `get_table`, `build_index`, `search`,
  `load_procedures`) so callers don't need to know the submodule split.
- **`conversation/`** — reserved for the LangGraph retrieve->agent graph
  (milestone 4). Currently just a docstring placeholder — no logic yet.
- **`main.py`** — FastAPI app. Only `/health` and `/refresh-conversation`
  (mints a `thread_id`, no real state) exist so far. Startup lifespan calls
  `ensure_index()`; if that raises (e.g. quota exhausted), falls back to
  `get_table()` so the server still boots and serves whatever's already
  indexed.

### The ingestion design is shaped by Google's free-tier embedding quota

`embedContent` is capped at roughly 1000 items/day, and the PDB corpus is
~1023 procedures — a single ingestion run can come up just short. Because of
that, `rag/ingest.py`'s `build_index()`:

- Writes each embedded chunk to the LanceDB table **immediately** (not
  batched in memory until the end), pacing itself (`EMBED_CHUNK_SIZE`,
  `EMBED_CHUNK_PAUSE_SECONDS`) rather than depending on retry/backoff.
- Records progress in a partial-ingest manifest keyed by procedure name
  (`ingest_partial.json`), so a re-run after a quota reset/crash only pays
  for the remainder.
- Is content-hash gated end to end (`ensure_index()`): the fingerprint is a
  hash of the JSONL export bytes + embedding model name, stored in
  `ingest_manifest.json` once a run completes fully. A normal backend
  startup is a fast no-op unless the export or model changed.
- `get_table()` (used for plain searching, e.g. by the CLI's `search`
  command and `main.py`'s startup fallback) deliberately does **not** try to
  finish an interrupted ingestion — it just opens whatever's there so
  searching isn't blocked on hitting the same quota error again.

`backend/data/lancedb/` (the built table) **is committed to git**, not
gitignored — shipping the pre-built index means a fresh checkout doesn't
have to re-embed ~1000 procedures against that same daily quota. Only
`data/lancedb/*.lock` is ignored. Note: the repo's root `.gitignore` has a
generic `*.manifest` rule (for PyInstaller) that collides with LanceDB's own
`.manifest` version files — `backend/.gitignore` has a negation
(`!data/lancedb/**/*.manifest`) to un-ignore those; keep that in mind if
LanceDB data ever silently fails to get staged.

### Testing conventions

- Every test that touches embeddings uses the `fake_embeddings` (or
  `patch_embeddings_client` / `make_fake_embeddings_client`) fixture from
  `tests/conftest.py` — no test may hit the real Google API. `sample_settings`
  gives an isolated `Settings` pointed at a `tmp_path` LanceDB dir and the
  small fixture corpus at `tests/fixtures/sample_pdb_export.jsonl` (5
  procedures), fully bypassing the real `.env` and the real committed index.
- `tests/unit/` mirrors `src/backend/`'s package layout 1:1
  (`shared/`, `rag/`, `conversation/`). `tests/integration/` boots the real
  FastAPI app via `TestClient` against a tmp RAG index.
- Coverage target is 95%+ (`--cov-fail-under=95`), currently at 100%. Lines
  that are impractical to hit honestly (the real network-calling body of
  `_embeddings_client`, `if __name__ == "__main__"` guards) are marked
  `# pragma: no cover` rather than given a contrived test.

## Milestones (see root README for the full list)

1. Repo restructure — done.
2. Backend skeleton (FastAPI boots, health check) — done.
3. RAG ingestion (all ~1023 procedures embedded, committed index) — done.
4. LangGraph agent (retrieve + Gemini agent node, checkpointed) — not started.
5. Endpoints (`/converse` wired to the graph) — not started; `/refresh-conversation` stub exists.
6. Plug-in (executor port, chat UI, execute-then-continue loop) — not started.
7. Cleanup (delete `pdb-tools/gimp_mcp_bridge.py` etc.) — not started.
8. End-to-end pass with a real GIMP instance — not started.

Update this file as milestones progress or the architecture shifts.
