from __future__ import annotations

import time
from pathlib import Path

import lancedb
import pytest

from backend.rag import ingest

SAMPLE_JSONL = Path(__file__).parents[2] / "fixtures" / "sample_pdb_export.jsonl"


def test_load_procedures_reads_all_lines():
    procedures = ingest.load_procedures(SAMPLE_JSONL)

    assert [p.name for p in procedures] == [
        "gimp-image-select-rectangle",
        "gimp-context-set-foreground",
        "gimp-image-flatten",
        "gimp-layer-new",
        "gimp-image-resize",
    ]


def test_fingerprint_changes_with_file_content(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(SAMPLE_JSONL.read_text())
    b.write_text(SAMPLE_JSONL.read_text() + "\n")

    assert ingest._source_fingerprint(a, "model-x") != ingest._source_fingerprint(b, "model-x")


def test_fingerprint_changes_with_embedding_model(tmp_path):
    a = tmp_path / "a.jsonl"
    a.write_text(SAMPLE_JSONL.read_text())

    assert ingest._source_fingerprint(a, "model-x") != ingest._source_fingerprint(a, "model-y")


def test_fingerprint_stable_for_same_inputs():
    assert ingest._source_fingerprint(SAMPLE_JSONL, "model-x") == ingest._source_fingerprint(
        SAMPLE_JSONL, "model-x"
    )


def test_manifest_roundtrip(sample_settings):
    sample_settings.lancedb_path.mkdir(parents=True)
    assert ingest._read_manifest(sample_settings) is None

    ingest._write_manifest(sample_settings, "abc123", 5)

    assert ingest._read_manifest(sample_settings) == {"source_hash": "abc123", "count": 5}


def test_read_manifest_returns_none_for_corrupted_file(sample_settings):
    sample_settings.lancedb_path.mkdir(parents=True)
    ingest._manifest_path(sample_settings).write_text("not valid json")

    assert ingest._read_manifest(sample_settings) is None


def test_partial_manifest_roundtrip_and_clear(sample_settings):
    sample_settings.lancedb_path.mkdir(parents=True)
    assert ingest._read_partial(sample_settings) is None

    ingest._write_partial(sample_settings, "abc123", {"a", "b"})
    partial = ingest._read_partial(sample_settings)
    assert partial["source_hash"] == "abc123"
    assert sorted(partial["done_names"]) == ["a", "b"]

    ingest._clear_partial(sample_settings)
    assert ingest._read_partial(sample_settings) is None


def test_read_partial_returns_none_for_corrupted_file(sample_settings):
    sample_settings.lancedb_path.mkdir(parents=True)
    ingest._partial_path(sample_settings).write_text("not valid json")

    assert ingest._read_partial(sample_settings) is None


def test_build_index_embeds_all_procedures_in_chunks(sample_settings, fake_embeddings, monkeypatch):
    monkeypatch.setattr(ingest, "EMBED_CHUNK_SIZE", 2)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    table = ingest.build_index(sample_settings)

    assert table.count_rows() == 5
    # 5 procedures in chunks of 2 -> three embed_documents calls (2, 2, 1)
    assert [len(c) for c in fake_embeddings.embed_documents_calls] == [2, 2, 1]

    manifest = ingest._read_manifest(sample_settings)
    assert manifest["count"] == 5
    assert ingest._read_partial(sample_settings) is None


def test_build_index_resumes_after_simulated_quota_error(
    sample_settings, monkeypatch, patch_embeddings_client, make_fake_embeddings_client
):
    monkeypatch.setattr(ingest, "EMBED_CHUNK_SIZE", 2)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    failing_client = patch_embeddings_client(make_fake_embeddings_client(fail_after_calls=1))

    with pytest.raises(RuntimeError, match="simulated quota error"):
        ingest.build_index(sample_settings)

    # First chunk (2 procedures) made it in before the simulated failure.
    partial = ingest._read_partial(sample_settings)
    assert len(partial["done_names"]) == 2
    assert ingest._read_manifest(sample_settings) is None

    db = lancedb.connect(sample_settings.lancedb_path)
    assert db.open_table(ingest.TABLE_NAME).count_rows() == 2

    resuming_client = patch_embeddings_client(make_fake_embeddings_client())

    table = ingest.build_index(sample_settings)

    assert table.count_rows() == 5
    # Only the remaining 3 procedures should have been (re-)embedded.
    total_embedded = sum(len(c) for c in resuming_client.embed_documents_calls)
    assert total_embedded == 3
    assert ingest._read_manifest(sample_settings)["count"] == 5
    assert ingest._read_partial(sample_settings) is None


def test_ensure_index_builds_then_skips_on_second_call(
    sample_settings, monkeypatch, patch_embeddings_client, make_fake_embeddings_client
):
    client = patch_embeddings_client(make_fake_embeddings_client())

    table = ingest.ensure_index(sample_settings)
    assert table.count_rows() == 5
    assert len(client.embed_documents_calls) == 1

    def _boom(settings):
        raise AssertionError("embeddings client should not be constructed on a no-op ensure_index")

    monkeypatch.setattr(ingest, "_embeddings_client", _boom)

    table_again = ingest.ensure_index(sample_settings)
    assert table_again.count_rows() == 5


def test_ensure_index_force_rebuilds_even_when_unchanged(sample_settings, fake_embeddings):
    ingest.ensure_index(sample_settings)
    assert len(fake_embeddings.embed_documents_calls) == 1

    ingest.ensure_index(sample_settings, force=True)
    assert len(fake_embeddings.embed_documents_calls) == 2


def test_get_table_opens_existing_partial_without_embedding(
    sample_settings, monkeypatch, patch_embeddings_client, make_fake_embeddings_client
):
    monkeypatch.setattr(ingest, "EMBED_CHUNK_SIZE", 2)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    patch_embeddings_client(make_fake_embeddings_client(fail_after_calls=1))

    with pytest.raises(RuntimeError):
        ingest.build_index(sample_settings)

    def _boom(settings):
        raise AssertionError("get_table should not need to embed for a matching partial table")

    monkeypatch.setattr(ingest, "_embeddings_client", _boom)

    table = ingest.get_table(sample_settings)
    assert table.count_rows() == 2


def test_get_table_builds_from_scratch_when_nothing_exists(sample_settings, fake_embeddings):
    table = ingest.get_table(sample_settings)

    assert table.count_rows() == 5
    assert ingest._read_manifest(sample_settings)["count"] == 5
