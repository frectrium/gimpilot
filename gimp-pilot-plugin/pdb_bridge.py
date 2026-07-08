"""Coerce values between JSON-ish Python and the real GIMP PDB, and call a
named PDB procedure with them.

Ported from `pdb-tools/gimp_mcp_bridge.py`'s `to_pdb_value`/`from_pdb_value`/
`call_procedure` (that file's docstring documents the original argument
conventions this still follows), adapted to be called in-process from the
plug-in rather than over a socket.
"""

from __future__ import annotations

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp, GObject, Gio  # noqa: E402


def _resolve_enum(pytype, value: str):
    """Match a string enum value (e.g. "RUN-NONINTERACTIVE") against the
    enum pytype's real members, case/hyphen/underscore-insensitively.

    The backend's tool schema (see `backend/.../conversation/tools.py`)
    represents PDB enum args as a plain JSON string, and live-testing showed
    Gemini return the value's nick as a string (e.g. "RUN-NONINTERACTIVE"),
    not the int GIMP's PDB actually expects for that property.

    `pytype.__enum_values__` (a dict of int -> enum instance, each carrying
    `.value_name`/`.value_nick`) is PyGObject's real introspection surface
    for a GI-generated enum class — confirmed live after `GObject.
    enum_list_values(gtype)` (an earlier, wrong guess) raised `AttributeError`.
    """
    normalized = value.strip().lower().replace("_", "-")
    for member in pytype.__enum_values__.values():
        nick = (getattr(member, "value_nick", "") or "").lower()
        name = (getattr(member, "value_name", "") or "").lower().replace("_", "-")
        if normalized == nick or name.endswith(normalized):
            return int(member)
    raise ValueError(f"unknown enum value {value!r} for {pytype.__name__}")


def to_pdb_value(pspec, value):
    """Coerce a JSON-decoded value onto the GType a PDB argument expects."""
    if value is None:
        return None

    gtype = pspec.value_type
    pytype = gtype.pytype

    if pytype is not None and hasattr(pytype, "get_by_id"):
        return pytype.get_by_id(int(value))

    if gtype.name == "GFile":
        return Gio.File.new_for_path(value)

    if isinstance(pytype, type) and issubclass(pytype, GObject.GEnum) and isinstance(value, str):
        return _resolve_enum(pytype, value)

    return value


def from_pdb_value(value):
    """Turn a PDB return value into something JSON-serializable."""
    if isinstance(value, (list, tuple)):
        return [from_pdb_value(v) for v in value]
    if hasattr(value, "get_id"):
        return value.get_id()
    if isinstance(value, Gio.File):
        path = value.peek_path()
        return path if path is not None else value.get_uri()
    if isinstance(value, GObject.GEnum):
        return int(value)
    return value


def _resolve_core_object_array(value):
    """Resolve a JSON list of ids (or a bare id) into actual PDB objects.

    `GimpCoreObjectArray`-typed args (e.g. script-fu-unsharp-mask's
    `drawables`) can't go through `config.set_property()` — confirmed live:
    it raises "could not convert '4' to type 'GimpCoreObjectArray'". They
    need `config.set_core_object_array()` instead (see
    `libgimp/gimpprocedureconfig.h`'s `gimp_procedure_config_set_core_object_array`,
    and `python-console.py`'s own generated-code template for the same
    special case). `Gimp.Item` is the common base of Drawable/Layer/Channel/
    Vectors, so `Item.get_by_id` resolves any of them to the right runtime type.
    """
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [Gimp.Item.get_by_id(int(v)) for v in value]


def call_procedure(pdb, name: str, args: dict | None) -> dict:
    """Run a named PDB procedure with JSON-ish args; never raises — reports
    failures as `{"ok": False, "error": ...}` so the caller can feed them
    back to the backend as a `tool_result` instead of crashing the plug-in.
    """
    proc = pdb.lookup_procedure(name)
    if proc is None:
        return {"ok": False, "error": f"no such PDB procedure: {name}"}

    try:
        config = proc.create_config()
        for arg_name, raw_value in (args or {}).items():
            pspec = proc.find_argument(arg_name)
            if pspec is None:
                return {"ok": False, "error": f'{name}: no such argument "{arg_name}"'}
            if pspec.value_type.name == "GimpCoreObjectArray":
                config.set_core_object_array(arg_name, _resolve_core_object_array(raw_value))
            else:
                config.set_property(arg_name, to_pdb_value(pspec, raw_value))

        result = proc.run(config)

        # Index 0 is always the implicit run status (never counted in
        # `proc.get_return_values()`); declared return values follow at
        # index 1+ on success, or a single error message does on failure.
        # Use the *actual* result length, not the declared-return-value
        # count, or a status-only failure (e.g. gimp-image-crop, which
        # declares zero return values) reads as `status = None` always.
        values = [from_pdb_value(result.index(i)) for i in range(result.length())]
    except Exception as error:
        return {"ok": False, "error": f"{name} raised {error}"}

    status = values[0] if values else None
    if status == Gimp.PDBStatusType.SUCCESS:
        return {"ok": True, "result": values[1:]}

    error = values[1] if len(values) > 1 else None
    return {"ok": False, "error": str(error) if error else f"{name} failed: {status}"}
