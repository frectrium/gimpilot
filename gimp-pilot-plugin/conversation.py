"""Orchestrates one user turn: send a message (or start a fresh thread),
then automatically execute whatever procedures the backend asks for against
the local PDB, feeding each result back, until the backend reports the turn
is `done`. No GTK/GIMP-registration code here — see `chat_window.py` for the
UI that drives this and `gimp-pilot-plugin.py` for the plug-in entry point.
"""

from __future__ import annotations

from typing import Callable

from pdb_bridge import call_procedure

ToolCallHook = Callable[[str, dict], None]
ToolResultHook = Callable[[str, dict], None]


class ConversationController:
    def __init__(self, client, pdb, gather_context: Callable[[], dict]):
        self.client = client
        self.pdb = pdb
        self.gather_context = gather_context
        self.thread_id: str | None = None

    def start_new_conversation(self) -> str:
        self.thread_id = self.client.refresh_conversation()
        return self.thread_id

    def send_message(
        self,
        message: str,
        on_tool_call: ToolCallHook | None = None,
        on_tool_result: ToolResultHook | None = None,
    ) -> str:
        """Send `message`, then keep executing any procedures the backend
        asks for until it responds with `done: True`. Returns the final,
        user-facing AI message.
        """
        if self.thread_id is None:
            raise RuntimeError("start_new_conversation() must be called before send_message()")

        response = self.client.converse(
            self.thread_id, message=message, context=self.gather_context()
        )
        return self._drive_tool_loop(response, on_tool_call, on_tool_result)

    def _drive_tool_loop(
        self, response: dict, on_tool_call: ToolCallHook | None, on_tool_result: ToolResultHook | None
    ) -> str:
        while not response["done"]:
            tool_call = response["tool_call"]
            procedure = tool_call["procedure"]
            args = tool_call["args"]
            if on_tool_call:
                on_tool_call(procedure, args)

            outcome = call_procedure(self.pdb, procedure, args)
            if on_tool_result:
                on_tool_result(procedure, outcome)

            tool_result = {"procedure": procedure, "ok": outcome["ok"]}
            if outcome["ok"]:
                tool_result["result"] = outcome.get("result")
            else:
                tool_result["error"] = outcome.get("error")

            response = self.client.converse(
                self.thread_id, tool_result=tool_result, context=self.gather_context()
            )

        return response["message"]
