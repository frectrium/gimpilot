"""Build per-turn LangChain/Gemini tool schemas from candidate PDB procedures.

Each candidate procedure retrieved by RAG becomes one tool, with a JSON-schema
`parameters` built straight from that procedure's real PDB arg spec — not a
single generic "call any procedure" tool — so Gemini sees real per-argument
descriptions/constraints instead of hallucinating a payload shape.

Arg-type mapping here is a deliberately simple heuristic, not a full PDB type
system: exact argument coercion (e.g. turning a `GimpImage` JSON int back into
whatever the PDB actually expects) is the plug-in's job at execution time
(see the root README's `to_pdb_value` note). This only needs to be good
enough for Gemini to produce plausible arguments, and each arg's `description`
(GIMP's `param_spec_get_desc` already bakes in things like valid enum
members) carries most of that signal regardless of the declared JSON type.
"""

from __future__ import annotations

from backend.shared.schemas import PDBProcedure

_INT_TYPES = {
    "gint",
    "gint8",
    "gint16",
    "gint32",
    "gint64",
    "guint",
    "guint8",
    "guint16",
    "guint32",
    "guint64",
}
_FLOAT_TYPES = {"gdouble", "gfloat"}

# GIMP core object "handle" types — passed as plain integer ids per the
# convention documented in the root README.
_HANDLE_TYPES = {
    "GimpImage",
    "GimpItem",
    "GimpLayer",
    "GimpChannel",
    "GimpDrawable",
    "GimpSelection",
    "GimpVectors",
    "GimpDisplay",
    "GimpBrush",
    "GimpFont",
    "GimpGradient",
    "GimpPalette",
    "GimpPattern",
    "GimpResource",
    "GimpLayerMask",
}


def _json_schema_type(pdb_type: str) -> dict:
    if pdb_type in _INT_TYPES or pdb_type in _HANDLE_TYPES:
        return {"type": "integer"}
    if pdb_type in _FLOAT_TYPES:
        return {"type": "number"}
    if pdb_type == "gboolean":
        return {"type": "boolean"}
    if pdb_type == "GStrv":
        return {"type": "array", "items": {"type": "string"}}
    # gchararray, GFile, enums (GimpRunMode, ...), structs (GimpRGB, ...),
    # and anything else unrecognized: fall back to string.
    return {"type": "string"}


def build_tool_schema(procedure: PDBProcedure) -> dict:
    """A bare `{"name", "description", "parameters"}` dict describing one
    procedure as a callable tool — accepted directly by
    `ChatGoogleGenerativeAI.bind_tools` (confirmed against
    `langchain_google_genai._function_utils`).
    """
    properties: dict[str, dict] = {}
    required: list[str] = []
    for arg in procedure.args:
        schema = _json_schema_type(arg.type)
        description = arg.description or arg.nick
        if description:
            schema["description"] = description
        properties[arg.name] = schema
        required.append(arg.name)

    description = procedure.blurb or procedure.help or procedure.name
    return {
        "name": procedure.name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }
