"""Gather the fresh GIMP-side snapshot (`context` field) sent with every
`/converse` call — see the root README's API section.

GIMP doesn't expose a PDB call for "the currently active image" (confirmed
against `../gimp/libgimp/gimpimage_pdb.h`); `Gimp.get_images()` returns all
open images, so the first one is taken as a documented simplifying
assumption (matching GIMP 2.x's `gimp.image_list()[0]` convention).

Includes the image's width/height — confirmed live that without them,
Gemini has no way to pick sensible crop/resize parameters and invents
plausible-looking but wrong numbers (e.g. a crop offset that violates
GIMP's own `0 <= offset <= (dimension - new_dimension)` bounds check,
which GIMP then legitimately rejects as an execution error).
"""

from __future__ import annotations

import gi

gi.require_version("Gimp", "3.0")
from gi.repository import Gimp  # noqa: E402


def gather_context() -> dict:
    images = Gimp.get_images() or []
    if not images:
        return {"image_id": None, "width": None, "height": None, "selection": None, "layers": None}

    image = images[0]
    layers = [
        {"id": layer.get_id(), "name": layer.get_name()} for layer in (image.get_layers() or [])
    ]

    _, non_empty, x1, y1, x2, y2 = Gimp.Selection.bounds(image)
    selection = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1} if non_empty else None

    return {
        "image_id": image.get_id(),
        "width": image.get_width(),
        "height": image.get_height(),
        "selection": selection,
        "layers": layers,
    }
