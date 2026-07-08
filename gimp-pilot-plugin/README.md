# gimp-plugin

The GIMP-side plug-in. Runs inside GIMP (Python-fu), presents the UI for
natural-language requests, and is the only thing with a live PDB handle.

Responsibilities:
- Collect the user's request plus relevant context (active image id,
  selection, layers, ...) and POST it to the local `backend` server.
- Receive back a plan (procedure name(s) + args) and execute it against
  the PDB directly (`Gimp.get_pdb().lookup_procedure(...)`).
- Nothing here calls Google GenAI or a vector DB directly — that's the
  backend's job. This plug-in is a thin client + executor.

The PDB-calling/type-coercion logic in `../pdb-tools/gimp_mcp_bridge.py`
was written for an earlier "headless bridge" design and needs to move
here as the actual executor module, adapted to be invoked from the
plugin's own request handling rather than a standalone socket server.
