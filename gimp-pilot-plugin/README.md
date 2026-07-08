# gimp-pilot-plugin

The GIMP-side plug-in. Runs inside GIMP (Python-fu), presents a chat window
for natural-language requests, and is the only thing with a live PDB handle.

Responsibilities:
- Show a chat window; collect the user's request plus fresh context (image
  id, dimensions, selection, layers) and POST it to the local `backend`
  server (`/converse`, `/refresh-conversation` — see the root README's API
  section).
- Whatever the backend proposes (a `tool_call`), execute it against the PDB
  directly (`Gimp.get_pdb().lookup_procedure(...)`), automatically feeding
  the result back and repeating until the backend says the turn is `done`
  — no user input needed in between.
- Nothing here calls Google GenAI or a vector DB directly — that's the
  backend's job. This plug-in is a thin client + executor.

Requires GIMP 3.x and a running `backend` (see [`../backend/README.md`](../backend/README.md)
and the root [README](../README.md) for the full setup).

## Install

GIMP 3.x plug-ins are one-folder-per-plug-in, folder name matching the main
script's name, dropped into GIMP's per-user plug-ins directory. Copy this
whole directory there and make the entry point executable:

**macOS**
```
cp -r gimp-pilot-plugin "$HOME/Library/Application Support/GIMP/3.2/plug-ins/"
chmod +x "$HOME/Library/Application Support/GIMP/3.2/plug-ins/gimp-pilot-plugin/gimp-pilot-plugin.py"
```

**Linux**
```
cp -r gimp-pilot-plugin "$HOME/.config/GIMP/3.2/plug-ins/"
chmod +x "$HOME/.config/GIMP/3.2/plug-ins/gimp-pilot-plugin/gimp-pilot-plugin.py"
```

**Windows** (PowerShell)
```
Copy-Item -Recurse gimp-pilot-plugin "$env:APPDATA\GIMP\3.2\plug-ins\"
```

Adjust the `3.2` version folder to match your install (confirmed against
GIMP's own `gimp_directory()` source: it's the app's `major.minor` version,
not the `3.0` typelib ABI version used in `gi.require_version` calls). If
unsure, check GIMP's **Edit > Preferences > Folders > Plug-Ins** dialog,
which lists the exact directory your install actually searches — that's the
authoritative source, more reliable than guessing the path.

The dev-only files (`tests/`, `pyproject.toml`, `uv.lock`, `.venv/`) don't
need to be copied — GIMP only ever imports `gimp-pilot-plugin.py` and its
sibling `.py` modules — but copying them along is harmless if you used a
plain `cp -r`/`Copy-Item`.

Restart GIMP (a fresh plug-in install requires a restart to be picked up).
The plug-in appears as **Filters > GIMP Pilot...**, and opens a chat window
that starts a fresh conversation automatically.

## Run

1. Start the backend first (from `../backend`):
   ```
   uv run uvicorn backend.main:app --port 8765
   ```
2. Open GIMP, open or create an image, then **Filters > GIMP Pilot...**.
3. Type a request (e.g. *"sharpen this image a little and crop the edges a
   tiny bit"*) and hit Enter/Send. Watch the **Tool Activity** pane for each
   procedure call and its result as the plug-in executes them automatically;
   the main transcript shows your messages and the AI's final summary once
   the turn is done. Click **+** any time to start a fresh conversation.

By default the plug-in talks to `http://127.0.0.1:8765`. Override with the
`GIMP_PILOT_BACKEND_URL` environment variable if the backend runs on a
different port/host — set it in the environment GIMP itself is launched
from (a shell env var won't reach GIMP if it's launched via Finder/the Start
Menu/`open -a`; launch GIMP's own binary directly from that same shell if
you need this).

## Testing

```
uv sync
uv run pytest
```

`gi`/`gi.repository.Gimp` only exist inside GIMP's own bundled Python
runtime — there is no way to import them standalone (confirmed: invoking
GIMP's bundled interpreter outside the app's own process crashes). So
`tests/conftest.py` installs minimal fake `gi.repository.{GObject,Gio,Gimp}`
modules into `sys.modules` before any plug-in module is imported, giving
`pdb_bridge.py`/`context.py`'s real code just enough duck-typed surface to
run against in plain Python. `conversation.py` and `backend_client.py` don't
need that at all (they take an injected `pdb`/hit a real ephemeral local
HTTP server, respectively). `gimp-pilot-plugin.py` and `chat_window.py` are
GIMP/GTK glue and are **not** unit tested — no contrived GTK test harness;
they're verified by actually running the plug-in in real GIMP.

## Layout

- `gimp-pilot-plugin.py` — the `Gimp.PlugIn` entry point: registers the
  plug-in, opens the chat window on invocation. GIMP/GTK glue only, not
  unit tested (see Testing above) — exercised by actually running it.
- `chat_window.py` — the GTK window: transcript pane (user messages + the
  AI's final summary), a separate "Tool Activity" pane (each `tool_call` +
  its result, as they happen), an input box, and a "+" button that starts a
  fresh conversation (also fired once automatically on open). Runs
  background work (HTTP calls, PDB execution) off a `threading.Thread` and
  marshals UI updates back via `GLib.idle_add`, so the window stays
  responsive during a multi-second Gemini round trip. Thin glue, not unit
  tested.
- `conversation.py` — `ConversationController`: the actual orchestration
  (start a thread, send a message, then automatically drive the tool-call
  loop against the PDB until `done`). No GTK/GIMP-registration code — unit
  tested with fakes.
- `backend_client.py` — `BackendClient`, a stdlib-`urllib`-only HTTP client
  for `/refresh-conversation` and `/converse` (GIMP's bundled Python doesn't
  ship `requests`). Reads `GIMP_PILOT_BACKEND_URL` env var, defaults to
  `http://127.0.0.1:8765`. Unit tested against a real ephemeral local
  `http.server`.
- `pdb_bridge.py` — `to_pdb_value`/`from_pdb_value`/`call_procedure`, ported
  from `../pdb-tools/gimp_mcp_bridge.py` (see that file's docstring for the
  original argument conventions) and adapted to be called in-process rather
  than over a socket. Unit tested with fake GI objects (see Testing).
- `context.py` — `gather_context()`: builds the `{image_id, width, height,
  selection, layers}` snapshot sent with every `/converse` call. Unit tested.

## Bugs found via live GIMP testing

None of these were caught by the (fully mocked) unit test suite — only by
actually running the plug-in in GIMP. If a new PDB-call error shows up,
suspect the same class of issue: something about the real GIMP API surface
that the fake `gi` shim in `tests/conftest.py` doesn't (and structurally
can't fully) reproduce.

- **PDB enum args as strings.** The backend's tool schema represents
  unrecognized PDB enum types (e.g. `GimpRunMode`) as a plain JSON string,
  and Gemini duly returned values like `"run-mode": "RUN-NONINTERACTIVE"` —
  a string nick, not the int GIMP's PDB wants. `pdb_bridge.to_pdb_value`
  resolves such strings via the enum pytype's real `__enum_values__` dict
  (case/hyphen/underscore-insensitive match against the nick or the tail of
  the value name). An earlier attempt at this used a nonexistent
  `GObject.enum_list_values(gtype)` function, caught live via
  `AttributeError`.
- **`GimpCoreObjectArray` args** (e.g. `drawables`) can't go through
  `config.set_property()` — confirmed against
  `../gimp/libgimp/gimpprocedureconfig.h`, they need
  `config.set_core_object_array()` instead. The backend's schema also
  needed a fix so Gemini emits an array (`[4]`) instead of a bare id
  (`"4"`).
- **Return-value unpacking off-by-one.** `call_procedure` used to size its
  read from `len(proc.get_return_values())`, but GIMP's real `Gimp.ValueArray`
  always has the run status at index 0 first (not counted in
  `get_return_values()`). Every procedure with zero declared return values
  (e.g. `gimp-image-crop`) read `status = None` and was reported as failed
  even when it had actually succeeded. Fixed by reading the array's real
  length (`result.length()`) instead of trusting the declared count.
- **Missing image dimensions.** `context.gather_context()` didn't send the
  image's width/height, so Gemini had no way to pick sensible crop/resize
  numbers and invented ones that violated GIMP's own `0 <= offset <=
  (dimension - new_dimension)` bounds check (a legitimate `EXECUTION_ERROR`,
  not a coercion bug). Now included in the context snapshot.
