"""FastAPI entry point.

Only `/health` and `/refresh-conversation` exist so far. `/converse` and the
LangGraph agent (`backend.conversation`) land with milestone 4/5 — see the
root README's Milestones section.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI

from backend.rag import ensure_index, get_table
from backend.shared.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        ensure_index(settings)
    except Exception:
        logger.exception(
            "RAG index build/refresh failed on startup; falling back to "
            "whatever is already indexed and searchable."
        )
        get_table(settings)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/refresh-conversation")
def refresh_conversation() -> dict:
    return {"thread_id": str(uuid4())}


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
