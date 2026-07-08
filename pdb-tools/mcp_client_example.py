"""
Example of the client side: what your MCP server does to call into the
running headless GIMP via gimp_mcp_bridge.py. No `gi` import needed here —
this is plain Python, runs in the MCP server process, not inside GIMP.
"""

import json
import socket


class GimpBridge:
    def __init__(self, host="127.0.0.1", port=10088):
        self.sock = socket.create_connection((host, port))
        self.rfile = self.sock.makefile("r", encoding="utf-8")

    def call(self, procedure, **args):
        request = json.dumps({"procedure": procedure, "args": args}) + "\n"
        self.sock.sendall(request.encode("utf-8"))
        response = json.loads(self.rfile.readline())
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    gimp = GimpBridge()

    # e.g. an MCP "open_image" tool -> gimp-file-load, args per export_pdb.py's
    # recorded arg names/types for that procedure.
    result = gimp.call("gimp-version")
    print("GIMP version:", result)

    image_id, = gimp.call(
        "gimp-file-load",
        run_mode=1,  # Gimp.RunMode.NONINTERACTIVE
        file="/path/to/input.png",
    )
    gimp.call("gimp-image-scale", image=image_id, new_width=800, new_height=600)
    gimp.call(
        "file-png-export",
        run_mode=1,
        image=image_id,
        file="/path/to/output.png",
    )

    gimp.close()
