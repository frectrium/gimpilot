from __future__ import annotations

from gi.repository import Gimp

import context


class FakeLayer:
    def __init__(self, layer_id, name):
        self._id = layer_id
        self._name = name

    def get_id(self):
        return self._id

    def get_name(self):
        return self._name


class FakeImage:
    def __init__(self, image_id, layers, width=800, height=600):
        self._id = image_id
        self._layers = layers
        self._width = width
        self._height = height

    def get_id(self):
        return self._id

    def get_layers(self):
        return self._layers

    def get_width(self):
        return self._width

    def get_height(self):
        return self._height


def test_gather_context_returns_empty_snapshot_when_no_images_open(monkeypatch):
    monkeypatch.setattr(Gimp, "get_images", lambda: [])

    assert context.gather_context() == {
        "image_id": None,
        "width": None,
        "height": None,
        "selection": None,
        "layers": None,
    }


def test_gather_context_reports_image_id_dimensions_and_layers(monkeypatch):
    image = FakeImage(1, [FakeLayer(10, "Background"), FakeLayer(11, "Sky")], width=1024, height=768)
    monkeypatch.setattr(Gimp, "get_images", lambda: [image])
    monkeypatch.setattr(Gimp.Selection, "bounds", staticmethod(lambda img: (True, False, 0, 0, 0, 0)))

    result = context.gather_context()

    assert result["image_id"] == 1
    assert result["width"] == 1024
    assert result["height"] == 768
    assert result["layers"] == [{"id": 10, "name": "Background"}, {"id": 11, "name": "Sky"}]
    assert result["selection"] is None


def test_gather_context_reports_selection_bounds_when_non_empty(monkeypatch):
    image = FakeImage(1, [])
    monkeypatch.setattr(Gimp, "get_images", lambda: [image])
    monkeypatch.setattr(
        Gimp.Selection, "bounds", staticmethod(lambda img: (True, True, 5, 10, 55, 60))
    )

    result = context.gather_context()

    assert result["selection"] == {"x": 5, "y": 10, "width": 50, "height": 50}


def test_gather_context_uses_first_image_when_multiple_open(monkeypatch):
    first = FakeImage(1, [])
    second = FakeImage(2, [])
    monkeypatch.setattr(Gimp, "get_images", lambda: [first, second])
    monkeypatch.setattr(Gimp.Selection, "bounds", staticmethod(lambda img: (True, False, 0, 0, 0, 0)))

    assert context.gather_context()["image_id"] == 1
