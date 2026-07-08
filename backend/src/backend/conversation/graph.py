"""The retrieve -> agent LangGraph graph — one pass per `/converse` call.

The backend never runs a tool itself (only the plug-in has the live PDB
handle), so this graph doesn't loop internally: each invocation does exactly
one retrieval + one Gemini call, then ends. If Gemini asks to call a
procedure, the HTTP layer returns that `tool_call` and the plug-in drives the
next step by executing it and POSTing the result back to `/converse`, which
resumes this same checkpointed thread with that result appended as a
`ToolMessage`. See the root README's "Why the tool loop is split across an
HTTP round trip" section.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph

from backend.conversation.tools import build_tool_schema
from backend.rag import search
from backend.shared.config import Settings, get_settings

SYSTEM_PROMPT = """\
You are the reasoning core of GIMP Pilot, translating a user's natural \
language image-editing request into calls to GIMP's Procedure Database (PDB).

You are offered a shortlist of PDB procedures as tools for this turn — the \
ones retrieved as most relevant right now, not the whole PDB. Call at most \
one procedure per turn: pick the single next procedure that makes progress \
on the user's request, with correctly typed arguments per the tool's \
parameter schema (descriptions note valid ranges/enum members where \
relevant). Never call a procedure that wasn't offered to you as a tool this \
turn.

If the most recent message is a tool result, use it to judge whether that \
step succeeded and decide the next step.

Once the user's whole request has been carried out — or if no offered \
procedure is relevant and you should just answer directly — respond with a \
plain, user-facing message summarizing what was done, and do not call any \
tool.\
"""


class ConversationState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    candidates: list  # list[ScoredProcedure], ephemeral — recomputed every run


def _chat_client(settings: Settings) -> ChatGoogleGenerativeAI:  # pragma: no cover
    # Real network-calling client — tests patch this out entirely, same as
    # `rag.ingest._embeddings_client`, so this body never runs under test.
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        temperature=0,
    )


def _build_retrieval_query(messages: list[BaseMessage]) -> str:
    """The latest human ask, plus any tool results reported since — so once
    e.g. a sharpen result comes back, retrieval is biased toward whatever's
    still left to do (e.g. crop) rather than re-finding "sharpen" again.
    """
    parts: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            parts = [str(message.content)]
        elif isinstance(message, ToolMessage):
            parts.append(str(message.content))
    return "\n".join(parts)


def retrieve_node(state: ConversationState, *, settings: Settings) -> dict:
    query = _build_retrieval_query(state["messages"])
    candidates = search(query, settings=settings)
    return {"candidates": candidates}


def agent_node(state: ConversationState, *, settings: Settings) -> dict:
    llm = _chat_client(settings)
    tool_schemas = [build_tool_schema(c.procedure) for c in state["candidates"]]
    llm = llm.bind_tools(tool_schemas) if tool_schemas else llm

    messages = [SystemMessage(SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)

    # The API contract offers at most one tool_call per turn; if Gemini ever
    # proposes more, only the first is honored.
    if isinstance(response, AIMessage) and len(response.tool_calls) > 1:
        response.tool_calls = response.tool_calls[:1]

    return {"messages": [response]}


def build_graph(settings: Settings | None = None) -> CompiledStateGraph:
    settings = settings or get_settings()

    graph = StateGraph(ConversationState)
    graph.add_node("retrieve", lambda state: retrieve_node(state, settings=settings))
    graph.add_node("agent", lambda state: agent_node(state, settings=settings))
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "agent")
    graph.add_edge("agent", END)

    return graph.compile(checkpointer=MemorySaver())
