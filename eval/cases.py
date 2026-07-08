"""Eval test cases for the gimpilot backend's RAG + agent pipeline.

Every expected procedure name here was checked against the real, committed
`backend/data/pdb_export.jsonl` before being written down (not guessed) —
see `eval/README.md` for the grep commands used.

Categories:
- "single_step": exactly one procedure call is expected, then the turn
  should end (`done: true`).
- "multi_step": an ordered sequence of procedure calls is expected, each
  fed back as a successful `tool_result` before checking the next step.
- "no_tool_call": the backend should answer directly, without proposing a
  procedure at all (a question, not an editing request).
- "hallucination_check": like `single_step`, but after the proposed tool
  call, the eval deliberately feeds back a *failed* `tool_result` and
  checks that the final message acknowledges the failure rather than
  falsely claiming success.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_CONTEXT = {
    "image_id": 1,
    "width": 1024,
    "height": 768,
    "selection": None,
    "layers": [{"id": 1, "name": "Background"}],
}


@dataclass
class EvalCase:
    id: str
    message: str
    category: str
    # One entry per expected step, in order; each entry is a set of
    # acceptable alternative procedure names for that step (more than one
    # real PDB procedure can satisfy the same request).
    expected_procedures: list[list[str]] = field(default_factory=list)
    context: dict = field(default_factory=lambda: dict(DEFAULT_CONTEXT))


CASES: list[EvalCase] = [
    EvalCase("sharpen", "Sharpen this image a little bit.", "single_step", [["script-fu-unsharp-mask"]]),
    EvalCase("crop", "Crop 20 pixels off each edge of the image.", "single_step", [["gimp-image-crop"]]),
    EvalCase("scale", "Resize this image to 500x500 pixels.", "single_step", [["gimp-image-scale"]]),
    EvalCase("rotate", "Rotate the image 90 degrees.", "single_step", [["gimp-image-rotate"]]),
    EvalCase("flip", "Flip the image horizontally.", "single_step", [["gimp-image-flip"]]),
    EvalCase(
        "grayscale",
        "Make this image black and white.",
        "single_step",
        [["gimp-drawable-desaturate", "gimp-image-convert-grayscale"]],
    ),
    EvalCase("flatten", "Flatten all the layers into one.", "single_step", [["gimp-image-flatten"]]),
    EvalCase("invert", "Invert the colors of this image.", "single_step", [["gimp-drawable-invert"]]),
    EvalCase(
        "brightness", "Increase the brightness a bit.", "single_step", [["gimp-drawable-brightness-contrast"]]
    ),
    EvalCase("new_layer", "Add a new blank layer to the image.", "single_step", [["gimp-layer-new"]]),
    EvalCase(
        "select_rectangle",
        "Select a rectangle from (10,10) to (100,100).",
        "single_step",
        [["gimp-image-select-rectangle"]],
    ),
    EvalCase("select_all", "Select the entire image.", "single_step", [["gimp-selection-all"]]),
    EvalCase("saturation", "Increase the color saturation.", "single_step", [["gimp-drawable-hue-saturation"]]),
    EvalCase("posterize", "Posterize this image.", "single_step", [["gimp-drawable-posterize"]]),
    EvalCase("export_png", "Export this image as a PNG file.", "single_step", [["file-png-export"]]),
    EvalCase(
        "sharpen_then_crop",
        "Sharpen this image a little and then crop 20 pixels off each edge.",
        "multi_step",
        [["script-fu-unsharp-mask"], ["gimp-image-crop"]],
    ),
    EvalCase(
        "grayscale_then_flatten",
        "Make this image black and white, then flatten the layers.",
        "multi_step",
        [["gimp-drawable-desaturate", "gimp-image-convert-grayscale"], ["gimp-image-flatten"]],
    ),
    EvalCase("question_no_action", "What does the unsharp mask filter do?", "no_tool_call", []),
    EvalCase(
        "hallucination_sharpen",
        "Sharpen this image a little bit.",
        "hallucination_check",
        [["script-fu-unsharp-mask"]],
    ),
]
