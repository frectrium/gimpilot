# gimpilot

A GIMP plugin that uses RAG + an LLM agent (LangGraph) to turn natural
language requests into GIMP PDB procedure calls.

## Components

- **`gimp-pilot-plugin/`** — the GIMP-side plug-in (Python-fu). Runs
  inside GIMP, presents the chat UI, and is the *only* thing with a live
  PDB handle. Thin client + executor: it calls the backend, and it's the
  one that actually invokes `Gimp.get_pdb().lookup_procedure(...)`.
- **`backend/`** — local HTTP server (its own venv/port). Owns the RAG
  index over PDB procedures and the LangGraph agent. Never touches the
  PDB directly — it only ever returns *proposed* procedure calls for the
  plug-in to execute.
- **`pdb-tools/export_pdb.py`** — run once (per GIMP version) inside
  GIMP's headless batch interpreter to dump the full PDB to JSONL. Feeds
  the backend's vector store. (`gimp_mcp_bridge.py` and
  `mcp_client_example.py` in this directory are leftovers from an earlier
  standalone-bridge design and are superseded — the PDB-calling/type-
  coercion logic in `gimp_mcp_bridge.py` still needs to be ported into
  `gimp-pilot-plugin`'s executor before those two files are deleted.)

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
`to_pdb_value`/`from_pdb_value`: core object types (`GimpImage`,
`GimpLayer`, `GimpDrawable`, ...) are plain integer ids, `GFile` args are
path strings, everything else passes through as-is. That coercion logic
moves into `gimp-pilot-plugin` as the thing that actually executes calls.

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
- **Tooling**: `uv`-managed `pyproject.toml` — `uv venv`, `uv add
  fastapi uvicorn langgraph langchain-google-genai lancedb pydantic
  python-dotenv`, `uv run uvicorn ...`. Google AI Studio API key (used
  for both the `text-embedding` model and the Gemini chat model) read
  from env (`GOOGLE_API_KEY`), never hardcoded.

## Planned layout

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
  # existing thin-client responsibilities, plus:
  # - executor module (ported from pdb-tools/gimp_mcp_bridge.py's
  #   to_pdb_value/from_pdb_value + call_procedure logic)
  # - HTTP client for /converse and /refresh-conversation
  # - chat UI (message list + input box) inside a GIMP dock/dialog
```

## Milestones

1. **Repo restructure** — done: `gimp-plugin/` → `gimp-pilot-plugin/`.
2. **Backend skeleton** — done: `pyproject.toml`, FastAPI app boots on a
   configurable port, health check + `/refresh-conversation` stub.
3. **RAG ingestion** — done: all ~1023 PDB procedures embedded into the
   committed LanceDB table, with the hash-gated skip-if-unchanged behavior.
4. **LangGraph agent** — done: retrieve node + Gemini (`gemini-3.1-flash-lite`)
   agent node with per-turn dynamic tool binding, checkpointed via
   `MemorySaver`.
5. **Endpoints** — done: real `/converse` wired to the graph, matching the
   request/response shapes above.
6. **Plug-in** — port the executor/type-coercion logic into
   `gimp-pilot-plugin`, build the chat UI, wire the execute-then-
   continue loop against the backend.
7. **Cleanup** — delete `pdb-tools/gimp_mcp_bridge.py` and
   `mcp_client_example.py` once their logic has moved.
8. **End-to-end pass** — real GIMP instance, real PDB, a handful of
   natural-language requests exercised manually.
