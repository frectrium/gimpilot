"""FastAPI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from langchain_core.messages import HumanMessage, ToolMessage

from backend.conversation import build_graph
from backend.conversation.schemas import (
    ConversationContext,
    ConverseRequest,
    ConverseResponse,
    ToolCallOut,
    ToolResult,
)
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
    # Built fresh per app startup (not at module-import time) so `settings`
    # is always whatever `get_settings()` currently resolves to, and so the
    # MemorySaver checkpointer's lifetime matches the running app's.
    app.state.conversation_graph = build_graph(settings)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/refresh-conversation")
def refresh_conversation() -> dict:
    return {"thread_id": str(uuid4())}


def _format_context_note(context: ConversationContext | None) -> str:
    if context is None:
        return ""
    return f"\n\n[Current GIMP state: {context.model_dump_json(exclude_none=True)}]"


def _format_tool_result(tool_result: ToolResult) -> str:
    if tool_result.ok:
        return f"Tool `{tool_result.procedure}` succeeded. Result: {tool_result.result!r}"
    return f"Tool `{tool_result.procedure}` failed. Error: {tool_result.error}"


def _message_text(content) -> str:
    """`AIMessage.content` is a plain string for most models, but Gemini
    sometimes returns a list of content blocks (`{"type": "text", "text":
    ..., ...}`, plus non-text blocks like signatures) — flatten either shape
    down to the plain text the UI should show verbatim.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def _last_tool_call_id(graph, config: dict) -> str:
    state = graph.get_state(config).values
    messages = state.get("messages", [])
    if messages:
        tool_calls = getattr(messages[-1], "tool_calls", None)
        if tool_calls:
            return tool_calls[0]["id"]
    raise HTTPException(status_code=400, detail="no pending tool_call for this thread_id")


@app.post("/converse", response_model=ConverseResponse)
def converse(payload: ConverseRequest, request: Request) -> ConverseResponse:
    graph = request.app.state.conversation_graph
    config = {"configurable": {"thread_id": payload.thread_id}}

    if payload.tool_result is not None:
        tool_call_id = _last_tool_call_id(graph, config)
        content = _format_tool_result(payload.tool_result) + _format_context_note(payload.context)
        input_message = ToolMessage(content=content, tool_call_id=tool_call_id)
    elif payload.message is not None:
        content = payload.message + _format_context_note(payload.context)
        input_message = HumanMessage(content=content)
    else:
        raise HTTPException(status_code=400, detail="must provide either 'message' or 'tool_result'")

    result = graph.invoke({"messages": [input_message]}, config=config)
    last_message = result["messages"][-1]

    tool_call_out = None
    done = True
    tool_calls = getattr(last_message, "tool_calls", None)
    if tool_calls:
        call = tool_calls[0]
        tool_call_out = ToolCallOut(procedure=call["name"], args=call["args"])
        done = False

    return ConverseResponse(
        thread_id=payload.thread_id,
        message=_message_text(last_message.content),
        tool_call=tool_call_out,
        done=done,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)
