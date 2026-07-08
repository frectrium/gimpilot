from __future__ import annotations

import pytest

import conversation as conversation_module
from conversation import ConversationController


class FakeClient:
    def __init__(self, converse_responses):
        self.refresh_calls = 0
        self.converse_calls: list[dict] = []
        self._responses = list(converse_responses)

    def refresh_conversation(self):
        self.refresh_calls += 1
        return f"thread-{self.refresh_calls}"

    def converse(self, thread_id, *, message=None, context=None, tool_result=None):
        self.converse_calls.append(
            {"thread_id": thread_id, "message": message, "context": context, "tool_result": tool_result}
        )
        return self._responses.pop(0)


@pytest.fixture
def gather_context_calls():
    calls = []
    return calls


@pytest.fixture
def make_controller(gather_context_calls):
    def _make(responses):
        client = FakeClient(responses)

        def gather_context():
            gather_context_calls.append(True)
            return {"image_id": 1}

        controller = ConversationController(client, pdb=object(), gather_context=gather_context)
        return controller, client

    return _make


def test_start_new_conversation_stores_thread_id(make_controller):
    controller, client = make_controller([])

    thread_id = controller.start_new_conversation()

    assert thread_id == "thread-1"
    assert controller.thread_id == "thread-1"
    assert client.refresh_calls == 1


def test_send_message_without_pending_conversation_raises(make_controller):
    controller, _ = make_controller([])

    with pytest.raises(RuntimeError, match="start_new_conversation"):
        controller.send_message("hello")


def test_send_message_returns_final_message_when_no_tool_call_needed(make_controller):
    controller, client = make_controller(
        [{"done": True, "tool_call": None, "message": "Sure, here's the answer."}]
    )
    controller.start_new_conversation()

    result = controller.send_message("what's this filter do?")

    assert result == "Sure, here's the answer."
    assert client.converse_calls == [
        {"thread_id": "thread-1", "message": "what's this filter do?", "context": {"image_id": 1}, "tool_result": None}
    ]


def test_send_message_drives_sharpen_then_crop_tool_loop(monkeypatch, make_controller):
    call_procedure_calls = []

    def fake_call_procedure(pdb, name, args):
        call_procedure_calls.append((pdb, name, args))
        return {"ok": True, "result": [42]}

    monkeypatch.setattr(conversation_module, "call_procedure", fake_call_procedure)

    controller, client = make_controller(
        [
            {"done": False, "tool_call": {"procedure": "gimp-image-flatten", "args": {"image": 1}}, "message": ""},
            {"done": False, "tool_call": {"procedure": "gimp-image-resize", "args": {"image": 1}}, "message": ""},
            {"done": True, "tool_call": None, "message": "All done — flattened and resized."},
        ]
    )
    controller.start_new_conversation()

    tool_calls_seen = []
    tool_results_seen = []
    result = controller.send_message(
        "flatten then resize",
        on_tool_call=lambda proc, args: tool_calls_seen.append((proc, args)),
        on_tool_result=lambda proc, outcome: tool_results_seen.append((proc, outcome)),
    )

    assert result == "All done — flattened and resized."
    assert tool_calls_seen == [
        ("gimp-image-flatten", {"image": 1}),
        ("gimp-image-resize", {"image": 1}),
    ]
    assert tool_results_seen == [
        ("gimp-image-flatten", {"ok": True, "result": [42]}),
        ("gimp-image-resize", {"ok": True, "result": [42]}),
    ]
    assert [c[1] for c in call_procedure_calls] == ["gimp-image-flatten", "gimp-image-resize"]

    # 1 initial message + 2 tool-result continuations
    assert len(client.converse_calls) == 3
    assert client.converse_calls[1]["tool_result"] == {
        "procedure": "gimp-image-flatten",
        "ok": True,
        "result": [42],
    }
    assert client.converse_calls[2]["tool_result"] == {
        "procedure": "gimp-image-resize",
        "ok": True,
        "result": [42],
    }


def test_send_message_sends_error_tool_result_when_procedure_fails(monkeypatch, make_controller):
    monkeypatch.setattr(
        conversation_module,
        "call_procedure",
        lambda pdb, name, args: {"ok": False, "error": "no active image"},
    )

    controller, client = make_controller(
        [
            {"done": False, "tool_call": {"procedure": "gimp-image-flatten", "args": {}}, "message": ""},
            {"done": True, "tool_call": None, "message": "Sorry, that failed."},
        ]
    )
    controller.start_new_conversation()

    result = controller.send_message("flatten it")

    assert result == "Sorry, that failed."
    assert client.converse_calls[1]["tool_result"] == {
        "procedure": "gimp-image-flatten",
        "ok": False,
        "error": "no active image",
    }


def test_send_message_gathers_fresh_context_every_step(monkeypatch, make_controller, gather_context_calls):
    monkeypatch.setattr(
        conversation_module, "call_procedure", lambda pdb, name, args: {"ok": True, "result": []}
    )

    controller, _ = make_controller(
        [
            {"done": False, "tool_call": {"procedure": "gimp-x", "args": {}}, "message": ""},
            {"done": True, "tool_call": None, "message": "done"},
        ]
    )
    controller.start_new_conversation()

    controller.send_message("do the thing")

    # once for the initial message, once for the tool-result continuation
    assert len(gather_context_calls) == 2
