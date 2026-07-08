"""The chat window's GTK shell.

Thin glue only: builds widgets and marshals `ConversationController`'s
(background-thread) work back onto GTK's main loop via `GLib.idle_add`. All
the actual conversation/tool-loop logic lives in `conversation.py` (unit
tested); this file is exercised by running the plug-in in real GIMP.
"""

from __future__ import annotations

import threading

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402

from backend_client import BackendClient, BackendError
from context import gather_context
from conversation import ConversationController


class ChatWindow(Gtk.Window):
    def __init__(self, pdb):
        super().__init__(title="GIMP Pilot")
        self.set_default_size(480, 640)
        self.connect("destroy", lambda *_args: Gtk.main_quit())

        self.controller = ConversationController(BackendClient(), pdb, gather_context)

        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        outer.set_border_width(8)
        self.add(outer)

        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        title_label = Gtk.Label(label="GIMP Pilot")
        title_label.set_halign(Gtk.Align.START)
        header.pack_start(title_label, True, True, 0)
        new_conversation_button = Gtk.Button(label="+")
        new_conversation_button.set_tooltip_text("Start a new conversation")
        new_conversation_button.connect("clicked", self._on_new_conversation_clicked)
        header.pack_end(new_conversation_button, False, False, 0)
        outer.pack_start(header, False, False, 0)

        self.transcript_view = self._make_text_view()
        outer.pack_start(self._wrap_in_scroller(self.transcript_view, height=320), True, True, 0)

        tool_activity_label = Gtk.Label(label="Tool Activity")
        tool_activity_label.set_halign(Gtk.Align.START)
        outer.pack_start(tool_activity_label, False, False, 0)

        self.tool_activity_view = self._make_text_view()
        outer.pack_start(self._wrap_in_scroller(self.tool_activity_view, height=140), False, True, 0)

        input_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.entry = Gtk.Entry()
        self.entry.set_placeholder_text("Ask GIMP Pilot to do something...")
        self.entry.connect("activate", self._on_send_clicked)
        input_row.pack_start(self.entry, True, True, 0)
        self.send_button = Gtk.Button(label="Send")
        self.send_button.connect("clicked", self._on_send_clicked)
        input_row.pack_start(self.send_button, False, False, 0)
        outer.pack_start(input_row, False, False, 0)

        self._start_new_conversation()

    # -- widget helpers ----------------------------------------------------

    @staticmethod
    def _make_text_view() -> Gtk.TextView:
        view = Gtk.TextView()
        view.set_editable(False)
        view.set_cursor_visible(False)
        view.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        return view

    @staticmethod
    def _wrap_in_scroller(view: Gtk.TextView, height: int) -> Gtk.ScrolledWindow:
        scroller = Gtk.ScrolledWindow()
        scroller.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.ALWAYS)
        scroller.set_min_content_height(height)
        scroller.add(view)
        return scroller

    @staticmethod
    def _append_text(view: Gtk.TextView, text: str) -> None:
        buffer = view.get_buffer()
        buffer.insert(buffer.get_end_iter(), text)
        view.scroll_to_iter(buffer.get_end_iter(), 0.0, False, 0, 0)

    def _append_transcript(self, text: str) -> None:
        self._append_text(self.transcript_view, text + "\n\n")

    def _append_tool_activity(self, text: str) -> None:
        self._append_text(self.tool_activity_view, text + "\n")

    def _clear_views(self) -> None:
        self.transcript_view.get_buffer().set_text("")
        self.tool_activity_view.get_buffer().set_text("")

    def _set_input_enabled(self, enabled: bool) -> None:
        self.entry.set_sensitive(enabled)
        self.send_button.set_sensitive(enabled)

    # -- conversation actions (each kicks off a background thread) --------

    def _on_new_conversation_clicked(self, _button) -> None:
        self._start_new_conversation()

    def _start_new_conversation(self) -> None:
        self._clear_views()
        self._set_input_enabled(False)

        def work() -> None:
            try:
                self.controller.start_new_conversation()
                GLib.idle_add(self._on_new_conversation_ready)
            except BackendError as error:
                GLib.idle_add(self._on_error, str(error))

        threading.Thread(target=work, daemon=True).start()

    def _on_new_conversation_ready(self) -> bool:
        self._append_transcript("New conversation started. How can I help?")
        self._set_input_enabled(True)
        return False

    def _on_send_clicked(self, _widget) -> None:
        message = self.entry.get_text().strip()
        if not message:
            return
        self.entry.set_text("")
        self._append_transcript(f"You: {message}")
        self._set_input_enabled(False)

        def work() -> None:
            try:
                final_message = self.controller.send_message(
                    message,
                    on_tool_call=self._notify_tool_call,
                    on_tool_result=self._notify_tool_result,
                )
                GLib.idle_add(self._on_turn_complete, final_message)
            except BackendError as error:
                GLib.idle_add(self._on_error, str(error))

        threading.Thread(target=work, daemon=True).start()

    def _notify_tool_call(self, procedure: str, args: dict) -> None:
        # Called from the background thread — marshal onto the GTK loop.
        GLib.idle_add(self._append_tool_activity, f"-> {procedure}({args})")

    def _notify_tool_result(self, procedure: str, outcome: dict) -> None:
        if outcome.get("ok"):
            GLib.idle_add(self._append_tool_activity, f"<- {procedure} ok: {outcome.get('result')}")
        else:
            GLib.idle_add(self._append_tool_activity, f"<- {procedure} FAILED: {outcome.get('error')}")

    def _on_turn_complete(self, final_message: str) -> bool:
        self._append_transcript(f"GIMP Pilot: {final_message}")
        self._set_input_enabled(True)
        return False

    def _on_error(self, message: str) -> bool:
        self._append_transcript(f"[error] {message}")
        self._set_input_enabled(True)
        return False

    def run(self) -> None:
        self.show_all()
        Gtk.main()
