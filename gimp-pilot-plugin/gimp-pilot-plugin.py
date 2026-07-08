#!/usr/bin/env python3
"""GIMP Pilot: a chat-driven assistant that turns natural-language image-
editing requests into GIMP PDB procedure calls, via a local backend server
(see `../backend`). This file only registers the plug-in and opens the chat
window — see `chat_window.py`, `conversation.py`, `backend_client.py`,
`pdb_bridge.py`, and `context.py` for the actual logic.

Install: this directory, as-is (folder name matching this file's name, per
GIMP 3's one-plug-in-per-folder convention), dropped into GIMP's plug-ins
folder — see README.md for exact paths and how to make this file executable.
"""

import sys

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp, GLib, GObject  # noqa: E402

PROC_NAME = "python-fu-gimp-pilot"


def run(procedure, config, data):
    gi.require_version("GimpUi", "3.0")
    gi.require_version("Gtk", "3.0")
    from gi.repository import GimpUi

    GimpUi.init(PROC_NAME)

    from chat_window import ChatWindow

    ChatWindow(Gimp.get_pdb()).run()

    return procedure.new_return_values(Gimp.PDBStatusType.SUCCESS, GLib.Error())


class GimpPilot(Gimp.PlugIn):
    def do_query_procedures(self):
        return [PROC_NAME]

    def do_create_procedure(self, name):
        procedure = Gimp.Procedure.new(self, name, Gimp.PDBProcType.PLUGIN, run, None)

        procedure.set_menu_label("_GIMP Pilot...")
        procedure.set_documentation(
            "Chat with GIMP Pilot",
            "Opens a chat window that turns natural-language requests into "
            "GIMP PDB procedure calls via a local backend server.",
            name,
        )
        procedure.set_attribution("", "", "2026")
        procedure.add_menu_path("<Image>/Filters")

        procedure.add_enum_argument(
            "run-mode",
            "Run mode",
            "The run mode",
            Gimp.RunMode,
            Gimp.RunMode.INTERACTIVE,
            GObject.ParamFlags.READWRITE,
        )

        return procedure


Gimp.main(GimpPilot.__gtype__, sys.argv)
