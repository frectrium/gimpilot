#!/usr/bin/env python3
"""
GIMP plug-in: a local JSON-RPC bridge onto the PDB, for an MCP server to
call into a running headless GIMP.

Install: drop this file (executable, `chmod +x`) into a directory on
GIMP's plug-in path (e.g. `~/GIMP/3.0/plug-ins/gimp_mcp_bridge/
gimp_mcp_bridge.py` — same-named folder as the file, GIMP 3 plug-ins are
one-file-per-folder) and restart GIMP once so it's registered.

Run headless, one persistent GIMP process per bridge:

    gimp-console-3.0 -idf --batch-interpreter=plug-in-script-fu-eval \
        -b '(python-fu-mcp-bridge "127.0.0.1" 10088)'

(the batch interpreter must be set explicitly to the Script-Fu one — the
default depends on plug-in load order otherwise.) This call never returns — it blocks GIMP in an accept() loop, exactly like
GIMP's built-in Script-Fu server (plug-ins/script-fu/server). Kill the
process (or send {"procedure": "gimp-quit", "args": {"exit-status": 0}})
to stop it.

Wire protocol: newline-delimited JSON over TCP, one request/response pair
per line (single-threaded — requests queue and run one at a time, since
the GIMP core isn't safe to hit concurrently from two PDB calls at once).

Request:
    {"procedure": "gimp-image-scale", "args": {"image": 1, "new-width": 800, "new-height": 600}}

Response:
    {"ok": true, "result": {"status": "SUCCESS"}}
    {"ok": false, "error": "no such procedure: ..."}

Argument conventions (matched against the arg "type" field from
export_pdb.py's output so the MCP layer knows how to fill args in):
  - GimpImage / GimpItem / GimpDrawable / GimpLayer / GimpDisplay / ...:
    pass the integer id (`image.get_id()` et al when GIMP hands you one
    back); the bridge resolves it with e.g. `Gimp.Image.get_by_id(id)`.
  - GFile: pass a path string.
  - everything else (int, double, string, bool, string arrays): passed
    through as-is; PyGObject coerces it onto the GParamSpec's GType.
Return values are converted back the other way (core objects -> id,
GFile -> path) so the response is plain JSON.
"""

import gi
gi.require_version('Gimp', '3.0')
from gi.repository import Gimp
from gi.repository import GObject
from gi.repository import Gio
from gi.repository import GLib

import json
import socketserver
import sys


def to_pdb_value(pspec, value):
    """Coerce a JSON-decoded value onto the GType a PDB argument expects."""
    if value is None:
        return None

    gtype = pspec.value_type
    pytype = gtype.pytype

    if pytype is not None and hasattr(pytype, 'get_by_id'):
        return pytype.get_by_id(int(value))

    if gtype.name == 'GFile':
        return Gio.File.new_for_path(value)

    return value


def from_pdb_value(value):
    """Turn a PDB return value into something json.dumps can handle."""
    if hasattr(value, 'get_id'):
        return value.get_id()
    if isinstance(value, Gio.File):
        path = value.peek_path()
        return path if path is not None else value.get_uri()
    if isinstance(value, GObject.GEnum):
        return int(value)
    return value


def call_procedure(pdb, name, args):
    proc = pdb.lookup_procedure(name)
    if proc is None:
        return {'ok': False, 'error': f'no such PDB procedure: {name}'}

    config = proc.create_config()
    for arg_name, raw_value in (args or {}).items():
        pspec = proc.find_argument(arg_name)
        if pspec is None:
            return {'ok': False, 'error': f'{name}: no such argument "{arg_name}"'}
        config.set_property(arg_name, to_pdb_value(pspec, raw_value))

    result = proc.run(config)

    n_return_values = len(proc.get_return_values() or [])
    values = [from_pdb_value(result.index(i)) for i in range(n_return_values)]

    status = values[0] if values else None
    if status == Gimp.PDBStatusType.SUCCESS:
        return {'ok': True, 'result': values[1:]}

    error = values[1] if len(values) > 1 else None
    return {'ok': False, 'error': str(error) if error else f'{name} failed: {status}'}


class BridgeHandler(socketserver.StreamRequestHandler):
    def handle(self):
        pdb = Gimp.get_pdb()
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = call_procedure(pdb, request.get('procedure'), request.get('args'))
            except Exception as error:
                response = {'ok': False, 'error': str(error)}
            self.wfile.write((json.dumps(response) + '\n').encode('utf-8'))


class BridgeServer(socketserver.TCPServer):
    allow_reuse_address = True


def serve(host, port):
    with BridgeServer((host, port), BridgeHandler) as server:
        sys.stderr.write(f'gimp-mcp-bridge: listening on {host}:{port}\n')
        sys.stderr.flush()
        server.serve_forever()


def run(procedure, config, run_data):
    host = config.get_property('host')
    port = config.get_property('port')
    serve(host, port)
    # unreachable: serve_forever() never returns; GIMP is killed to stop it
    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


class McpBridge(Gimp.PlugIn):
    def do_query_procedures(self):
        return ['python-fu-mcp-bridge']

    def do_create_procedure(self, name):
        procedure = Gimp.Procedure.new(self, name, Gimp.PDBProcType.PLUGIN, run, None)

        procedure.set_documentation(
            "Start a local JSON-RPC bridge onto the GIMP PDB",
            "Listens on a TCP socket and executes newline-delimited JSON "
            "requests ({\"procedure\": <pdb-name>, \"args\": {...}}) against "
            "the PDB, for an external MCP server to drive this GIMP "
            "instance. Blocks forever once started.",
            name)
        procedure.set_attribution("", "", "2026")

        procedure.add_string_argument(
            "host", "Host", "Address to listen on", "127.0.0.1",
            GObject.ParamFlags.READWRITE)
        procedure.add_int_argument(
            "port", "Port", "TCP port to listen on", 1, 65535, 10088,
            GObject.ParamFlags.READWRITE)

        return procedure


Gimp.main(McpBridge.__gtype__, sys.argv)
