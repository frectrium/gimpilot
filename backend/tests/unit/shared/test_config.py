from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.shared.config import BACKEND_ROOT, Settings, get_settings


def test_backend_root_resolves_to_backend_directory():
    assert BACKEND_ROOT.name == "backend"
    assert (BACKEND_ROOT / "pyproject.toml").exists()


def test_defaults_apply_when_only_required_field_given():
    settings = Settings(_env_file=None, google_api_key="test-key")

    assert settings.embedding_model == "models/gemini-embedding-2"
    assert settings.chat_model == "gemini-2.5-flash"
    assert settings.host == "127.0.0.1"
    assert settings.port == 8765
    assert settings.rag_top_k == 8
    assert settings.pdb_export_path == BACKEND_ROOT / "data" / "pdb_export.jsonl"
    assert settings.lancedb_path == BACKEND_ROOT / "data" / "lancedb"


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("EMBEDDING_MODEL", "models/some-other-model")
    monkeypatch.setenv("PORT", "1234")
    monkeypatch.setenv("RAG_TOP_K", "3")

    settings = Settings(_env_file=None, google_api_key="test-key")

    assert settings.embedding_model == "models/some-other-model"
    assert settings.port == 1234
    assert settings.rag_top_k == 3


def test_missing_google_api_key_raises(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_google_api_key_is_a_secret_str():
    settings = Settings(_env_file=None, google_api_key="super-secret")

    assert "super-secret" not in repr(settings.google_api_key)
    assert settings.google_api_key.get_secret_value() == "super-secret"


def test_get_settings_reads_from_environment_and_caches(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "cached-key")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.google_api_key.get_secret_value() == "cached-key"
        assert get_settings() is settings
    finally:
        get_settings.cache_clear()
