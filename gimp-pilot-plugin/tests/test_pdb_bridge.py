from __future__ import annotations

import pytest
from gi.repository import Gimp, Gio, GObject

import pdb_bridge


class FakeGType:
    def __init__(self, name, pytype=None):
        self.name = name
        self.pytype = pytype


class FakePSpec:
    def __init__(self, value_type):
        self.value_type = value_type


class FakeEnumValue:
    """Mimics one member of a real GI enum's `__enum_values__` dict."""

    def __init__(self, value, value_name, value_nick):
        self.value = value
        self.value_name = value_name
        self.value_nick = value_nick

    def __int__(self):
        return self.value


class FakeGimpImage:
    @classmethod
    def get_by_id(cls, image_id):
        return ("image", image_id)


class FakeRunMode(GObject.GEnum):
    # Real GI-generated enum classes carry this same `__enum_values__` shape
    # (int -> member instance) — see `pdb_bridge._resolve_enum`'s docstring
    # for why this replaced an earlier, wrong guess at the API.
    __enum_values__ = {
        1: FakeEnumValue(1, "GIMP_RUN_NONINTERACTIVE", "run-noninteractive"),
        0: FakeEnumValue(0, "GIMP_RUN_INTERACTIVE", "run-interactive"),
    }


# -- to_pdb_value --------------------------------------------------------


def test_to_pdb_value_passes_through_none():
    assert pdb_bridge.to_pdb_value(FakePSpec(FakeGType("gint")), None) is None


def test_to_pdb_value_resolves_handle_types_by_id():
    pspec = FakePSpec(FakeGType("GimpImage", pytype=FakeGimpImage))

    assert pdb_bridge.to_pdb_value(pspec, 7) == ("image", 7)


def test_to_pdb_value_builds_gfile_from_path():
    pspec = FakePSpec(FakeGType("GFile"))

    result = pdb_bridge.to_pdb_value(pspec, "/tmp/out.png")

    assert isinstance(result, Gio.File)
    assert result.peek_path() == "/tmp/out.png"


def test_to_pdb_value_resolves_string_enum_by_nick():
    pspec = FakePSpec(FakeGType("GimpRunMode", pytype=FakeRunMode))

    assert pdb_bridge.to_pdb_value(pspec, "RUN-NONINTERACTIVE") == 1


def test_to_pdb_value_resolves_string_enum_case_and_separator_insensitively():
    pspec = FakePSpec(FakeGType("GimpRunMode", pytype=FakeRunMode))

    assert pdb_bridge.to_pdb_value(pspec, "run_interactive") == 0


def test_to_pdb_value_raises_for_unresolvable_enum_string():
    pspec = FakePSpec(FakeGType("GimpRunMode", pytype=FakeRunMode))

    with pytest.raises(ValueError, match="unknown enum value"):
        pdb_bridge.to_pdb_value(pspec, "not-a-real-value")


def test_to_pdb_value_passes_through_int_enum_value_unchanged():
    pspec = FakePSpec(FakeGType("GimpRunMode", pytype=FakeRunMode))

    assert pdb_bridge.to_pdb_value(pspec, 1) == 1


def test_to_pdb_value_passes_through_plain_types():
    pspec = FakePSpec(FakeGType("gint"))

    assert pdb_bridge.to_pdb_value(pspec, 42) == 42
    assert pdb_bridge.to_pdb_value(FakePSpec(FakeGType("gchararray")), "hello") == "hello"


# -- from_pdb_value -------------------------------------------------------


def test_from_pdb_value_uses_get_id_for_core_objects():
    class FakeLayer:
        def get_id(self):
            return 3

    assert pdb_bridge.from_pdb_value(FakeLayer()) == 3


def test_from_pdb_value_converts_gfile_to_path():
    gfile = Gio.File.new_for_path("/tmp/x.png")

    assert pdb_bridge.from_pdb_value(gfile) == "/tmp/x.png"


def test_from_pdb_value_falls_back_to_uri_when_gfile_has_no_path():
    gfile = Gio.File(uri="memory://buffer")

    assert pdb_bridge.from_pdb_value(gfile) == "memory://buffer"


def test_from_pdb_value_converts_genum_to_int():
    class FakeStatus(GObject.GEnum):
        def __int__(self):
            return 5

    assert pdb_bridge.from_pdb_value(FakeStatus()) == 5


def test_from_pdb_value_passes_through_plain_values():
    assert pdb_bridge.from_pdb_value("plain string") == "plain string"
    assert pdb_bridge.from_pdb_value(42) == 42


def test_from_pdb_value_recurses_into_lists():
    class FakeLayer:
        def __init__(self, layer_id):
            self._id = layer_id

        def get_id(self):
            return self._id

    assert pdb_bridge.from_pdb_value([FakeLayer(1), FakeLayer(2)]) == [1, 2]


# -- call_procedure --------------------------------------------------------


class FakeConfig:
    def __init__(self):
        self.properties: dict[str, object] = {}
        self.core_object_arrays: dict[str, list] = {}

    def set_property(self, name, value):
        self.properties[name] = value

    def set_core_object_array(self, name, value):
        self.core_object_arrays[name] = value


class FakeProcedure:
    def __init__(self, args, run_result, raise_on_run=None):
        """`args`: dict of arg name -> FakePSpec (or a list of names, each
        defaulting to a plain `gint` pspec, for tests that don't care).
        """
        if isinstance(args, dict):
            self._args = args
        else:
            self._args = {a: FakePSpec(FakeGType("gint")) for a in args}
        self._run_result = run_result
        self._raise_on_run = raise_on_run
        self.config: FakeConfig | None = None

    def create_config(self):
        self.config = FakeConfig()
        return self.config

    def find_argument(self, name):
        return self._args.get(name)

    def run(self, config):
        if self._raise_on_run:
            raise self._raise_on_run
        return self._run_result


class FakeResult:
    """A fake `Gimp.ValueArray`. Real ones always hold the run status at
    index 0, then any declared return values on success, or a single error
    message on failure — `values` here should be given in that same shape.
    """

    def __init__(self, values):
        self._values = values

    def index(self, i):
        return self._values[i]

    def length(self):
        return len(self._values)


class FakePdb:
    def __init__(self, procedures: dict[str, FakeProcedure]):
        self._procedures = procedures

    def lookup_procedure(self, name):
        return self._procedures.get(name)


def test_call_procedure_success_with_no_declared_return_values():
    # e.g. gimp-image-crop: declares zero return values, so the real
    # ValueArray is just [status] — this used to be misread as status=None.
    proc = FakeProcedure(args=["image", "radius"], run_result=FakeResult(["SUCCESS"]))
    pdb = FakePdb({"gimp-sharpen": proc})

    outcome = pdb_bridge.call_procedure(pdb, "gimp-sharpen", {"image": 1, "radius": 2})

    assert outcome == {"ok": True, "result": []}
    assert proc.config.properties == {"image": 1, "radius": 2}


def test_call_procedure_returns_extra_return_values_on_success():
    proc = FakeProcedure(args=[], run_result=FakeResult(["SUCCESS", 99]))
    pdb = FakePdb({"gimp-layer-new": proc})

    outcome = pdb_bridge.call_procedure(pdb, "gimp-layer-new", {})

    assert outcome == {"ok": True, "result": [99]}


def test_call_procedure_unknown_procedure():
    pdb = FakePdb({})

    outcome = pdb_bridge.call_procedure(pdb, "does-not-exist", {})

    assert outcome == {"ok": False, "error": "no such PDB procedure: does-not-exist"}


def test_call_procedure_unknown_argument():
    proc = FakeProcedure(args=["image"], run_result=FakeResult(["SUCCESS"]))
    pdb = FakePdb({"gimp-thing": proc})

    outcome = pdb_bridge.call_procedure(pdb, "gimp-thing", {"nonexistent": 1})

    assert outcome == {"ok": False, "error": 'gimp-thing: no such argument "nonexistent"'}


def test_call_procedure_failure_status_with_no_declared_return_values():
    # e.g. gimp-image-crop failing: the real ValueArray on failure is
    # [status, error_message] regardless of the declared return-value count.
    proc = FakeProcedure(args=[], run_result=FakeResult(["EXECUTION_ERROR", "boom"]))
    pdb = FakePdb({"gimp-thing": proc})

    outcome = pdb_bridge.call_procedure(pdb, "gimp-thing", {})

    assert outcome == {"ok": False, "error": "boom"}


def test_call_procedure_failure_status_without_error_message():
    proc = FakeProcedure(args=[], run_result=FakeResult(["CALLING_ERROR"]))
    pdb = FakePdb({"gimp-thing": proc})

    outcome = pdb_bridge.call_procedure(pdb, "gimp-thing", {})

    assert outcome == {"ok": False, "error": "gimp-thing failed: CALLING_ERROR"}


def test_call_procedure_catches_exceptions_during_run():
    proc = FakeProcedure(args=[], run_result=None, raise_on_run=RuntimeError("kaboom"))
    pdb = FakePdb({"gimp-thing": proc})

    outcome = pdb_bridge.call_procedure(pdb, "gimp-thing", {})

    assert outcome == {"ok": False, "error": "gimp-thing raised kaboom"}


def test_call_procedure_uses_set_core_object_array_for_that_arg_type(monkeypatch):
    monkeypatch.setattr(Gimp.Item, "get_by_id", classmethod(lambda cls, item_id: ("item", item_id)))
    proc = FakeProcedure(
        args={"drawables": FakePSpec(FakeGType("GimpCoreObjectArray"))},
        run_result=FakeResult(["SUCCESS"]),
    )
    pdb = FakePdb({"script-fu-unsharp-mask": proc})

    outcome = pdb_bridge.call_procedure(pdb, "script-fu-unsharp-mask", {"drawables": "4"})

    assert outcome == {"ok": True, "result": []}
    assert proc.config.properties == {}
    assert proc.config.core_object_arrays == {"drawables": [("item", 4)]}


def test_resolve_core_object_array_wraps_bare_value_in_a_list(monkeypatch):
    monkeypatch.setattr(Gimp.Item, "get_by_id", classmethod(lambda cls, item_id: ("item", item_id)))

    assert pdb_bridge._resolve_core_object_array("4") == [("item", 4)]


def test_resolve_core_object_array_resolves_each_element_of_a_list(monkeypatch):
    monkeypatch.setattr(Gimp.Item, "get_by_id", classmethod(lambda cls, item_id: ("item", item_id)))

    assert pdb_bridge._resolve_core_object_array([1, 2]) == [("item", 1), ("item", 2)]
