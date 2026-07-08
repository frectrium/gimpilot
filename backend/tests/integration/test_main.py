from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import _message_text, app
from backend.rag import ingest


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("plain string", "plain string"),
        ("", ""),
        (None, ""),
        (
            [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
            "hello world",
        ),
        (
            [{"type": "text", "text": "kept"}, {"type": "signature", "signature": "abc"}],
            "kept",
        ),
        (["plain", "strings", "too"], "plainstringstoo"),
    ],
)
def test_message_text_flattens_gemini_content_blocks(content, expected):
    assert _message_text(content) == expected


def test_health_and_refresh_conversation(sample_settings, fake_embeddings, monkeypatch):
    monkeypatch.setattr("backend.main.get_settings", lambda: sample_settings)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        first = client.post("/refresh-conversation")
        second = client.post("/refresh-conversation")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["thread_id"] != second.json()["thread_id"]


def test_startup_falls_back_to_existing_table_when_ensure_index_fails(
    sample_settings, fake_embeddings, monkeypatch
):
    # Pre-build a searchable table so there's something for the fallback to find.
    ingest.build_index(sample_settings)

    monkeypatch.setattr("backend.main.get_settings", lambda: sample_settings)

    def _boom(settings):
        raise RuntimeError("simulated ensure_index failure (e.g. quota exceeded)")

    monkeypatch.setattr("backend.main.ensure_index", _boom)

    with TestClient(app) as client:
        health = client.get("/health")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
