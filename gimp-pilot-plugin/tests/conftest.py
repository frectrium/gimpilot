"""Test harness for GIMP-plugin-side code.

`gi`/`gi.repository.Gimp` only exists inside GIMP's own bundled Python
runtime (confirmed: invoking GIMP's bundled interpreter standalone crashes
outside the app's process). So before any plugin module is imported, this
conftest installs minimal fake `gi`/`gi.repository.{GObject,Gio,Gimp}`
modules into `sys.modules` — enough duck-typed surface for `pdb_bridge.py`
and `context.py`'s real code to run against, without a real GIMP process.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Plugin modules are flat files (no package) next to this tests/ dir, since
# GIMP imports gimp-pilot-plugin.py directly and relies on sibling imports
# via sys.path[0] — mirror that here so `import pdb_bridge` etc. works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _install_fake_gi() -> None:
    if "gi" in sys.modules and getattr(sys.modules["gi"], "_gimpilot_fake", False):
        return  # already installed (e.g. re-imported across test modules)

    gi_module = types.ModuleType("gi")
    gi_module._gimpilot_fake = True
    gi_module.require_version = lambda name, version: None
    sys.modules["gi"] = gi_module

    repository = types.ModuleType("gi.repository")
    sys.modules["gi.repository"] = repository
    gi_module.repository = repository

    # -- GObject: just enough for enum handling + ParamFlags --
    gobject_module = types.ModuleType("gi.repository.GObject")

    class GEnum:
        """Base class fake enum PDB arg types can subclass in tests.

        Real GI-generated enum classes carry a `__enum_values__` dict (int ->
        member instance, each with `.value_name`/`.value_nick`) — tests build
        that same shape on their fake subclasses.
        """

    class ParamFlags:
        READWRITE = 3

    gobject_module.GEnum = GEnum
    gobject_module.ParamFlags = ParamFlags
    sys.modules["gi.repository.GObject"] = gobject_module
    repository.GObject = gobject_module

    # -- Gio: just GFile construction/inspection --
    gio_module = types.ModuleType("gi.repository.Gio")

    class File:
        def __init__(self, path=None, uri=None):
            self._path = path
            self._uri = uri

        @classmethod
        def new_for_path(cls, path):
            return cls(path=path)

        def peek_path(self):
            return self._path

        def get_uri(self):
            return self._uri or (f"file://{self._path}" if self._path else None)

    gio_module.File = File
    sys.modules["gi.repository.Gio"] = gio_module
    repository.Gio = gio_module

    # -- Gimp: just PDBStatusType, enough for call_procedure's status check --
    gimp_module = types.ModuleType("gi.repository.Gimp")

    class PDBStatusType:
        SUCCESS = "SUCCESS"
        EXECUTION_ERROR = "EXECUTION_ERROR"
        CALLING_ERROR = "CALLING_ERROR"
        CANCEL = "CANCEL"

    class Selection:
        @staticmethod
        def bounds(image):
            return (True, False, 0, 0, 0, 0)

    class Item:
        @classmethod
        def get_by_id(cls, item_id):
            return ("item", item_id)

    gimp_module.PDBStatusType = PDBStatusType
    gimp_module.Selection = Selection
    gimp_module.Item = Item
    gimp_module.get_images = lambda: []
    sys.modules["gi.repository.Gimp"] = gimp_module
    repository.Gimp = gimp_module


_install_fake_gi()
