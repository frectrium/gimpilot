"""HTTP client for the local gimpilot backend's `/refresh-conversation` and
`/converse` endpoints (see the root README's API section for the request/
response shapes). Stdlib-only (`urllib.request`) — GIMP's bundled Python
doesn't ship `requests`, matching `pdb-tools/gimp_mcp_bridge.py`'s existing
dependency-free convention.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

DEFAULT_BACKEND_URL = "http://127.0.0.1:8765"


def backend_url() -> str:
    return os.environ.get("GIMP_PILOT_BACKEND_URL", DEFAULT_BACKEND_URL)


class BackendError(Exception):
    """Raised when the backend is unreachable or returns a non-2xx response."""


class BackendClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or backend_url()).rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise BackendError(f"{path} failed ({error.code}): {body}") from error
        except urllib.error.URLError as error:
            raise BackendError(f"{path} failed: {error.reason}") from error

    def refresh_conversation(self) -> str:
        return self._post("/refresh-conversation", {})["thread_id"]

    def converse(
        self,
        thread_id: str,
        *,
        message: str | None = None,
        context: dict | None = None,
        tool_result: dict | None = None,
    ) -> dict:
        payload: dict = {"thread_id": thread_id}
        if message is not None:
            payload["message"] = message
        if context is not None:
            payload["context"] = context
        if tool_result is not None:
            payload["tool_result"] = tool_result
        return self._post("/converse", payload)
