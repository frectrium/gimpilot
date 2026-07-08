"""Backend configuration, loaded from `backend/.env` (see `.env.example`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Google AI Studio — one key, used for both embeddings and the chat model.
    google_api_key: SecretStr
    # text-embedding-004 isn't available for this key's API version; of the
    # embedContent-capable models it does expose (gemini-embedding-001, -2,
    # -2-preview), each free-tier key gets its own ~1000-items/day quota
    # *per model* — right at the size of the whole PDB corpus (~1023 procs),
    # so a single day's ingestion can come up just short. -2 is the freshest
    # bucket. Uses the full 3072-dim output — at this corpus size that's
    # ~12MB, trivial to store/search locally, and avoids the manual
    # L2-renormalization Google recommends when truncating via
    # `output_dimensionality`.
    embedding_model: str = "models/gemini-embedding-2"
    chat_model: str = "gemini-3.1-flash-lite"

    # HTTP server
    host: str = "127.0.0.1"
    port: int = 8765

    # RAG
    pdb_export_path: Path = BACKEND_ROOT / "data" / "pdb_export.jsonl"
    lancedb_path: Path = BACKEND_ROOT / "data" / "lancedb"
    rag_top_k: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
