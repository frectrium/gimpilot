from __future__ import annotations

import sys

from backend.rag import __main__ as cli
from backend.rag import ingest


def test_cli_ingest_success(monkeypatch, capsys, sample_settings, fake_embeddings):
    monkeypatch.setattr(cli, "get_settings", lambda: sample_settings)
    monkeypatch.setattr(sys, "argv", ["prog", "ingest"])

    cli._cli()

    out = capsys.readouterr().out
    assert "Index ready: 5 procedures." in out


def test_cli_ingest_reports_partial_progress_on_failure(
    monkeypatch, capsys, sample_settings, fake_embeddings
):
    ingest.build_index(sample_settings)  # pre-seed a searchable table

    monkeypatch.setattr(cli, "get_settings", lambda: sample_settings)

    def _boom(settings, force=False):
        raise RuntimeError("simulated quota error (429)")

    monkeypatch.setattr(cli, "ensure_index", _boom)
    monkeypatch.setattr(sys, "argv", ["prog", "ingest"])

    cli._cli()

    out = capsys.readouterr().out
    assert "Ingestion stopped early" in out
    assert "5 procedures indexed so far" in out


def test_cli_search(monkeypatch, capsys, sample_settings, fake_embeddings):
    ingest.build_index(sample_settings)
    capsys.readouterr()  # discard ingest's own progress output
    monkeypatch.setattr("backend.rag.retrieval.get_settings", lambda: sample_settings)
    monkeypatch.setattr(sys, "argv", ["prog", "search", "select a rectangle", "--top-k", "2"])

    cli._cli()

    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2
