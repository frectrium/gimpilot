from __future__ import annotations

import pytest

from backend.conversation.tools import _json_schema_type, build_tool_schema
from backend.shared.schemas import PDBArgument, PDBProcedure


@pytest.mark.parametrize(
    ("pdb_type", "expected"),
    [
        ("gint", {"type": "integer"}),
        ("gint32", {"type": "integer"}),
        ("guint64", {"type": "integer"}),
        ("gdouble", {"type": "number"}),
        ("gfloat", {"type": "number"}),
        ("gboolean", {"type": "boolean"}),
        ("GStrv", {"type": "array", "items": {"type": "string"}}),
        ("GimpCoreObjectArray", {"type": "array", "items": {"type": "integer"}}),
        ("GimpImage", {"type": "integer"}),
        ("GimpLayer", {"type": "integer"}),
        ("GimpDrawable", {"type": "integer"}),
        ("gchararray", {"type": "string"}),
        ("GFile", {"type": "string"}),
        ("GimpRunMode", {"type": "string"}),  # unrecognized enum -> fallback
        ("GimpRGB", {"type": "string"}),  # unrecognized struct -> fallback
    ],
)
def test_json_schema_type_mapping(pdb_type, expected):
    assert _json_schema_type(pdb_type) == expected


def test_build_tool_schema_shape():
    procedure = PDBProcedure(
        name="gimp-image-select-rectangle",
        proc_type="PLUGIN",
        blurb="Select a rectangle on the specified image.",
        args=[
            PDBArgument(name="image", type="GimpImage", description="The image"),
            PDBArgument(name="width", type="gdouble", description="Width of the rectangle"),
        ],
    )

    schema = build_tool_schema(procedure)

    assert schema["name"] == "gimp-image-select-rectangle"
    assert schema["description"] == "Select a rectangle on the specified image."
    assert schema["parameters"]["type"] == "object"
    assert schema["parameters"]["properties"]["image"] == {
        "type": "integer",
        "description": "The image",
    }
    assert schema["parameters"]["properties"]["width"] == {
        "type": "number",
        "description": "Width of the rectangle",
    }
    assert schema["parameters"]["required"] == ["image", "width"]


def test_build_tool_schema_falls_back_to_nick_then_name_for_description():
    procedure_with_nick = PDBProcedure(
        name="gimp-noop",
        proc_type="PLUGIN",
        args=[PDBArgument(name="x", type="gint", nick="X coordinate")],
    )
    schema = build_tool_schema(procedure_with_nick)
    assert schema["parameters"]["properties"]["x"]["description"] == "X coordinate"

    procedure_with_no_blurb = PDBProcedure(name="gimp-noop", proc_type="PLUGIN")
    schema = build_tool_schema(procedure_with_no_blurb)
    assert schema["description"] == "gimp-noop"


def test_build_tool_schema_handles_no_args():
    procedure = PDBProcedure(name="gimp-image-flatten", proc_type="PLUGIN", blurb="Flatten.")

    schema = build_tool_schema(procedure)

    assert schema["parameters"]["properties"] == {}
    assert schema["parameters"]["required"] == []
