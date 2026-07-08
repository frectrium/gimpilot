"""`/converse` request/response models — see the root README's API section."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConversationContext(BaseModel):
    """Fresh GIMP-side snapshot sent with every `/converse` call."""

    model_config = ConfigDict(extra="allow")

    image_id: int | None = None
    selection: dict | None = None
    layers: list | None = None


class ToolResult(BaseModel):
    """The plug-in's report of executing the procedure from a prior `tool_call`."""

    procedure: str
    ok: bool
    result: list | None = None
    error: str | None = None


class ToolCallOut(BaseModel):
    """A procedure the backend wants the plug-in to execute next."""

    procedure: str
    args: dict


class ConverseRequest(BaseModel):
    thread_id: str
    message: str | None = None
    context: ConversationContext | None = None
    tool_result: ToolResult | None = None


class ConverseResponse(BaseModel):
    thread_id: str
    message: str
    tool_call: ToolCallOut | None = None
    done: bool
