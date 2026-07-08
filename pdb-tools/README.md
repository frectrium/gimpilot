# pdb-tools

- `export_pdb.py` — run inside GIMP's headless Python-fu batch
  interpreter to dump the full PDB (name, blurb, help, args, return
  values) to JSONL. Feeds the backend's vector DB. Still current.

- `gimp_mcp_bridge.py`, `mcp_client_example.py` — from an earlier design
  where a standalone server called into a persistent headless GIMP over
  a JSON socket. That's superseded now that the plug-in itself lives
  inside GIMP and executes procedures directly. Kept here as reference
  for the PDB-calling/type-coercion logic (see `gimp-plugin/README.md`)
  until it's adapted and moved into `../gimp-plugin`.
