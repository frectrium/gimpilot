# GIMPilot

A GIMP plugin that uses RAG + an LLM agent (LangGraph + Gemini) to turn
natural-language image-editing requests into real GIMP PDB procedure calls.
Ask for what you want in plain English; it finds the relevant GIMP
procedures, calls them one step at a time, and tells you what it did.

## Components

- **`gimp-pilot-plugin/`** — the GIMP-side plug-in (Python-fu). Runs
  inside GIMP, presents a chat window, and is the *only* thing with a live
  PDB handle. Thin client + executor: it calls the backend, and it's the
  one that actually invokes `Gimp.get_pdb().lookup_procedure(...)`.
- **`backend/`** — local HTTP server (its own venv/port). Owns the RAG
  index over PDB procedures and the LangGraph agent. Never touches the
  PDB directly — it only ever returns *proposed* procedure calls for the
  plug-in to execute.
- **`pdb-tools/export_pdb.py`** — run once (per GIMP version) inside
  GIMP's headless batch interpreter to dump the full PDB to JSONL. Feeds
  the backend's vector store.

## Prerequisites

- GIMP 3.x (developed/tested against 3.2.4)
- [`uv`](https://docs.astral.sh/uv/) for the backend's Python environment
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (used for
  both the RAG embeddings and the Gemini chat model)

## Quickstart

**1. Backend** — from `backend/`:
```
cp .env.example .env && $EDITOR .env   # fill in GOOGLE_API_KEY
uv sync
uv run uvicorn backend.main:app --port 8765
```
The RAG index (`backend/data/lancedb/`) is pre-built and committed, so a
fresh checkout doesn't need a separate ingestion step — see
[`backend/README.md`](backend/README.md) if you ever need to rebuild it
(e.g. after a GIMP PDB upgrade).

**2. Plug-in** — install `gimp-pilot-plugin/` into GIMP's plug-ins
directory and restart GIMP. See
[`gimp-pilot-plugin/README.md`](gimp-pilot-plugin/README.md) for the exact
per-OS install path and commands.

**3. Use it** — in GIMP, open an image, then **Filters > GIMP Pilot...**.
Type a request (e.g. *"sharpen this a little and crop the edges a tiny
bit"*) and hit Enter. The plug-in executes whatever procedures the backend
proposes automatically, showing each call/result in a "Tool Activity" pane,
and prints a final summary once the request is done.

## Why the tool loop is split across an HTTP round trip

The backend can propose a PDB call, but only the plug-in can run it (it
has the live `Gimp.get_pdb()` handle; the backend is a separate process
that may not even be on the same machine). So the agent loop is **one
step per HTTP call**:

1. Plug-in calls `POST /converse` with the user's message.
2. Backend runs one LangGraph turn: retrieve candidate PDB procedures →
   ask Gemini → Gemini either answers directly, or asks to call *one*
   procedure. Backend returns that to the plug-in and pauses (the graph
   state is checkpointed against a `thread_id`).
3. If a tool call came back, the plug-in executes it against the PDB and
   calls `POST /converse` again, this time including the tool result
   (success + return values, or an error string).
4. Backend resumes the checkpointed graph with that result appended as a
   `ToolMessage`, and Gemini decides again: call another procedure, or
   respond with a final answer. Repeat until the response has no tool
   call — that's the end of this user turn.

This means the agent always sees the *real* outcome of a step (e.g. the
id of a newly created layer) before deciding the next one, instead of
committing to a blind multi-step plan.

## API

Both endpoints live on the backend, port configurable via env/CLI (not
hardcoded, so it can run alongside other local services).

### `POST /refresh-conversation`

Starts a brand new conversation (a "New Chat"): drops any existing
LangGraph thread and history.

Request: `{}`
Response: `{ "thread_id": "..." }`

### `POST /converse`

Request:
```jsonc
{
  "thread_id": "...",           // from /refresh-conversation
  "message": "make the background blue",   // omit if this call is only carrying a tool_result
  "context": {                  // fresh snapshot every call, since backend has no PDB access
    "image_id": 1,
    "width": 1024, "height": 768,   // confirmed live: without these Gemini invents crop/resize numbers that violate GIMP's own bounds checks
    "selection": { ... },
    "layers": [ ... ]
  },
  "tool_result": {               // present only when replying to a prior tool_call
    "procedure": "gimp-image-select-rectangle",
    "ok": true,
    "result": [ ... ],           // or "error": "..." if ok is false
  }
}
```

Response:
```jsonc
{
  "thread_id": "...",
  "message": "Selecting the background layer...",  // shown in the UI verbatim
  "tool_call": {                 // omitted when the turn is finished
    "procedure": "gimp-image-select-rectangle",
    "args": { "image": 1, "operation": 2, "x": 0, "y": 0, "width": 800, "height": 200 }
  },
  "done": false                  // true once there's no tool_call left to run
}
```

Argument values follow the same convention as `gimp_mcp_bridge.py`'s
original `to_pdb_value`/`from_pdb_value`: core object types (`GimpImage`,
`GimpLayer`, `GimpDrawable`, ...) are plain integer ids, `GFile` args are
path strings, everything else passes through as-is. That coercion logic now
lives in `gimp-pilot-plugin/pdb_bridge.py`, the thing that actually executes
calls — including a best-effort fixup for PDB enum args (e.g. `GimpRunMode`)
that the backend's tool schema represents as a JSON string but the PDB
itself expects as an int; see that file's docstring and the plug-in
README's "Bugs found via live GIMP testing" for the details.

## Backend architecture

```
pdb-tools/export_pdb.py  --(jsonl)-->  backend ingestion  --(embed, cache)-->  LanceDB (backend/data/)
                                                                                      |
gimp-pilot-plugin  --HTTP-->  FastAPI (/converse, /refresh-conversation)  --> LangGraph app
                                                                                      |
                                                                     retrieve node -> agent node (Gemini)
```

- **RAG / vector store**: [LanceDB](https://lancedb.github.io/lancedb/)
  (embedded, file-based, no server process — table lives under
  `backend/data/`). Ingestion step reads `pdb-tools`' JSONL export,
  embeds each procedure (name + blurb + help + arg descriptions) with a
  Google `text-embedding` model, and upserts into the table. Guarded by
  a content hash of the JSONL file stored alongside the table: if the
  hash matches, ingestion is skipped entirely (no re-embedding on every
  startup). GIMP's PDB is a few thousand procedures at most, so a top-k
  similarity query is cheap.
- **Agent turn**: a "retrieve" node queries LanceDB for the top-k
  procedures relevant to the user's message (+ recent conversation
  context), then an "agent" node calls Gemini (`langchain-google-genai`)
  with those top-k procedures bound as *dynamically constructed* tools
  for that single invocation (one LangChain tool per candidate
  procedure, with a args schema built from that procedure's real
  PDB arg spec — not one static giant tool list, and not a single
  generic "call any procedure" tool — so Gemini gets real per-argument
  types/descriptions/constraints instead of hallucinating a payload
  shape). This is what gets bound fresh each turn since the candidate
  set changes with retrieval.
- **State / checkpointing**: LangGraph's in-memory checkpointer
  (`MemorySaver`), keyed by `thread_id`. No persistence to disk — this is
  a local single-user dev tool; losing conversation history on a backend
  restart is an acceptable tradeoff for the simplicity of not managing a
  second on-disk store next to the vector DB. `/refresh-conversation`
  just mints a new `thread_id` (old one is abandoned, garbage collected
  with the process).
- **Tooling**: `uv`-managed `pyproject.toml`. Google AI Studio API key (used
  for both the `text-embedding` model and the Gemini chat model) read
  from env (`GOOGLE_API_KEY`), never hardcoded.

## Repository layout

```
backend/
  pyproject.toml
  src/backend/
    main.py          # FastAPI app: /health, /refresh-conversation, /converse
    shared/
      config.py      # port, API key, paths (env-driven)
      schemas.py     # PDB procedure pydantic models
    rag/
      ingest.py      # JSONL -> embeddings -> LanceDB, hash-gated, resumable
      retrieval.py   # top-k query against LanceDB
      __main__.py    # `ingest`/`search` CLI
    conversation/    # LangGraph graph: retrieve -> agent, checkpointed
      graph.py       # build_graph, one retrieve+agent pass per /converse call
      tools.py       # candidate PDB procedures -> per-turn Gemini tool schemas
      schemas.py     # /converse request/response pydantic models
  tests/
    unit/            # mirrors src/backend/: shared/, rag/, conversation/
    integration/     # exercises the real FastAPI app end to end
  data/              # committed: jsonl export, pre-built lancedb table + hash marker

gimp-pilot-plugin/
  gimp-pilot-plugin.py   # Gimp.PlugIn entry point, opens the chat window
  chat_window.py         # GTK chat window (transcript + tool-activity pane + input)
  conversation.py        # ConversationController: drives the tool-call loop
  backend_client.py      # urllib-based client for /converse, /refresh-conversation
  pdb_bridge.py          # to_pdb_value/from_pdb_value/call_procedure (ported from pdb-tools)
  context.py             # gather_context(): {image_id, width, height, selection, layers}
  tests/                 # unit tests for everything except the GTK/GIMP glue
```

## Development

Each component has its own `uv`-managed environment and test suite:

```
cd backend && uv sync && uv run pytest --cov=backend --cov-report=term-missing --cov-fail-under=95
cd gimp-pilot-plugin && uv sync && uv run pytest
```

See each component's README for details on what's unit tested vs. only
verifiable by actually running the plug-in in GIMP.

CI (`.github/workflows/ci.yml`) runs both of the commands above on every
push to `main` and every pull request — no secrets required, since the test
suites never make a real API call or need a real `GOOGLE_API_KEY`.
Packaging/Docker image stages are planned but not built yet.

## Eval / Benchmarks

`eval/` benchmarks the backend's RAG + agent pipeline directly (no real
GIMP execution — tool results are simulated, matching what the plug-in
sends back after actually running a procedure). See
[`eval/README.md`](eval/README.md) for full methodology and limitations.
Latest run (19 hand-written cases, `gemini-3.1-flash-lite`,
`models/gemini-embedding-2`, 2026-07-08):

| Metric | Result |
| --- | --- |
| Tool-call accuracy (single-step) | 15/15 (100%) |
| Multi-step completion rate | 2/2 (100%) |
| No-tool-call correctness | 1/1 (100%) |
| Hallucinated-success rate | 0/1 (0%) |
| Avg. latency per `/converse` call | 3.2s |
| RAG recall@8 vs. naive keyword search | **100%** vs. 39% |
| RAG vs. naive keyword search latency | 0.76s vs. 0.008s |

Two things worth calling out honestly rather than glossing over:

- **This is a small, single run** (19 cases, one snapshot in time against a
  non-deterministic model) — a real signal, not a rigorous statistical
  claim. See `eval/README.md`'s Limitations section.
- **RAG is not faster than keyword search — it's more accurate.** Semantic
  retrieval costs a real embedding API round trip per query, so it's ~100x
  slower than an in-process keyword scan. What it buys is recall: 11 of
  19 requests (e.g. "sharpen this image" → `script-fu-unsharp-mask`, "make
  this image black and white" → `gimp-drawable-desaturate`) share no literal
  words with the procedure name/description a keyword search would need to
  match, and it found all of them anyway. There's no real "X% faster than
  a manual GIMP workflow" claim here — no human-timing baseline exists to
  compare against, and inventing one would be misleading.

Raw results for this run: [`eval/results/`](eval/results/).

## License

[GPL-3.0](LICENSE), matching GIMP's own license (this project loads and
calls into `libgimp` via a GIMP plug-in).
