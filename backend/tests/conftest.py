"""Shared fixtures for the backend test suite.

Every test that touches embeddings uses the `fake_embeddings` fixture — no
test may hit the real Google API (costs quota, requires network, is slow).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from backend.shared.config import Settings

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_JSONL = FIXTURES_DIR / "sample_pdb_export.jsonl"

VECTOR_DIM = 8


def fake_vector(text: str) -> list[float]:
    """Deterministic stand-in embedding: a hash of the text, not semantically
    meaningful, but stable across calls/processes so tests are reproducible.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255 for b in digest[:VECTOR_DIM]]


class FakeEmbeddingsClient:
    """Drop-in stand-in for `GoogleGenerativeAIEmbeddings` — no network calls."""

    def __init__(self, fail_after_calls: int | None = None):
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []
        self._fail_after_calls = fail_after_calls

    def embed_documents(self, texts, batch_size=None):
        if self._fail_after_calls is not None and len(self.embed_documents_calls) >= self._fail_after_calls:
            raise RuntimeError("simulated quota error (429)")
        self.embed_documents_calls.append(list(texts))
        return [fake_vector(t) for t in texts]

    def embed_query(self, text):
        self.embed_query_calls.append(text)
        return fake_vector(text)


@pytest.fixture
def patch_embeddings_client(monkeypatch):
    """Returns `patch(client)`, which installs `client` as what
    `_embeddings_client(settings)` returns in both `rag.ingest` and
    `rag.retrieval` — the one seam every test must use instead of the real
    `GoogleGenerativeAIEmbeddings`.
    """

    def _patch(client):
        monkeypatch.setattr("backend.rag.ingest._embeddings_client", lambda settings: client)
        monkeypatch.setattr("backend.rag.retrieval._embeddings_client", lambda settings: client)
        return client

    return _patch


@pytest.fixture
def make_fake_embeddings_client():
    """The `FakeEmbeddingsClient` class, for tests that need a custom
    instance (e.g. one that fails after N calls to simulate a quota error).
    """
    return FakeEmbeddingsClient


@pytest.fixture
def fake_embeddings(patch_embeddings_client, make_fake_embeddings_client):
    """A ready-to-use `FakeEmbeddingsClient`, already patched in — the
    default for tests that don't care about simulating failures.
    """
    return patch_embeddings_client(make_fake_embeddings_client())


@pytest.fixture
def sample_settings(tmp_path) -> Settings:
    """A `Settings` pointed at a tmp LanceDB dir + the fixture JSONL, fully
    isolated from any real `.env` file or the real committed index.
    """
    return Settings(
        _env_file=None,
        google_api_key="test-key",
        pdb_export_path=SAMPLE_JSONL,
        lancedb_path=tmp_path / "lancedb",
    )
