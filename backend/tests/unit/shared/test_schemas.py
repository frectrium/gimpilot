from __future__ import annotations

from pathlib import Path

from backend.shared.schemas import PDBArgument, PDBProcedure, ScoredProcedure

SAMPLE_JSONL = Path(__file__).parents[2] / "fixtures" / "sample_pdb_export.jsonl"


def test_pdb_argument_defaults():
    arg = PDBArgument(name="image", type="GimpImage")

    assert arg.nick == ""
    assert arg.description == ""


def test_pdb_procedure_defaults():
    proc = PDBProcedure(name="gimp-noop", proc_type="PLUGIN")

    assert proc.blurb == ""
    assert proc.args == []
    assert proc.return_values == []
    assert proc.deprecated is False


def test_pdb_procedure_parses_real_export_line():
    first_line = SAMPLE_JSONL.read_text(encoding="utf-8").splitlines()[0]

    proc = PDBProcedure.model_validate_json(first_line)

    assert proc.name == "gimp-image-select-rectangle"
    assert proc.proc_type == "PLUGIN"
    assert len(proc.args) == 1
    assert proc.args[0].name == "image"


def test_scored_procedure_wraps_procedure_with_distance():
    proc = PDBProcedure(name="gimp-noop", proc_type="PLUGIN")

    scored = ScoredProcedure(procedure=proc, distance=0.42)

    assert scored.procedure.name == "gimp-noop"
    assert scored.distance == 0.42
