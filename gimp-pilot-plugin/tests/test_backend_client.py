from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import backend_client
from backend_client import BackendClient, BackendError


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        payload = json.loads(body) if body else {}
        self.server.received.append((self.path, payload))

        status, response_body = self.server.responses.get(self.path, (200, {}))
        data = json.dumps(response_body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # silence request logging
        pass


@pytest.fixture
def http_server():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.received = []
    server.responses = {}
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()


def _client_for(server) -> BackendClient:
    return BackendClient(base_url=f"http://127.0.0.1:{server.server_port}")


def test_refresh_conversation(http_server):
    http_server.responses["/refresh-conversation"] = (200, {"thread_id": "abc-123"})

    thread_id = _client_for(http_server).refresh_conversation()

    assert thread_id == "abc-123"
    assert http_server.received == [("/refresh-conversation", {})]


def test_converse_with_message_and_context(http_server):
    http_server.responses["/converse"] = (
        200,
        {
            "thread_id": "t1",
            "message": "",
            "tool_call": {"procedure": "gimp-sharpen", "args": {}},
            "done": False,
        },
    )

    result = _client_for(http_server).converse("t1", message="sharpen it", context={"image_id": 1})

    assert result["tool_call"]["procedure"] == "gimp-sharpen"
    assert result["done"] is False
    path, payload = http_server.received[0]
    assert path == "/converse"
    assert payload == {"thread_id": "t1", "message": "sharpen it", "context": {"image_id": 1}}


def test_converse_with_tool_result_omits_message(http_server):
    http_server.responses["/converse"] = (
        200,
        {"thread_id": "t1", "message": "done", "tool_call": None, "done": True},
    )

    _client_for(http_server).converse(
        "t1", tool_result={"procedure": "x", "ok": True, "result": []}
    )

    _, payload = http_server.received[0]
    assert "message" not in payload
    assert "context" not in payload
    assert payload["tool_result"] == {"procedure": "x", "ok": True, "result": []}


def test_backend_error_on_http_error_status(http_server):
    http_server.responses["/converse"] = (500, {"detail": "boom"})

    with pytest.raises(BackendError, match="500"):
        _client_for(http_server).converse("t1", message="hi")


def test_backend_error_on_connection_failure():
    client = BackendClient(base_url="http://127.0.0.1:1", timeout=2)

    with pytest.raises(BackendError):
        client.refresh_conversation()


def test_backend_url_reads_env_var(monkeypatch):
    monkeypatch.setenv("GIMP_PILOT_BACKEND_URL", "http://example.invalid:9999")

    assert backend_client.backend_url() == "http://example.invalid:9999"
    assert BackendClient().base_url == "http://example.invalid:9999"


def test_default_backend_url(monkeypatch):
    monkeypatch.delenv("GIMP_PILOT_BACKEND_URL", raising=False)

    assert backend_client.backend_url() == backend_client.DEFAULT_BACKEND_URL
