# pdb-tools

- `export_pdb.py` — run inside GIMP's headless Python-fu batch
  interpreter to dump the full PDB (name, blurb, help, args, return
  values) to JSONL. Feeds the backend's vector DB. Still current.

`gimp_mcp_bridge.py` and `mcp_client_example.py` used to live here (an
earlier design where a standalone server called into a persistent headless
GIMP over a JSON socket, superseded once the plug-in itself moved inside
GIMP and started executing procedures directly). Their PDB-calling/type-
coercion logic (`to_pdb_value`/`from_pdb_value`/`call_procedure`) was ported
into `../gimp-pilot-plugin/pdb_bridge.py`, adapted to run in-process rather
than over a socket, and the two original files have since been deleted.
