from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.conversation import graph as graph_module
from backend.conversation.graph import _build_retrieval_query, agent_node, build_graph, retrieve_node
from backend.rag import ingest
from backend.shared.schemas import PDBProcedure, ScoredProcedure


def test_build_retrieval_query_uses_latest_human_message():
    messages = [HumanMessage(content="sharpen this a bit")]

    assert _build_retrieval_query(messages) == "sharpen this a bit"


def test_build_retrieval_query_appends_tool_results_after_latest_human_message():
    messages = [
        HumanMessage(content="sharpen and crop"),
        AIMessage(content="", tool_calls=[{"name": "gimp-sharpen", "args": {}, "id": "call_1"}]),
        ToolMessage(content="sharpen succeeded", tool_call_id="call_1"),
    ]

    assert _build_retrieval_query(messages) == "sharpen and crop\nsharpen succeeded"


def test_build_retrieval_query_resets_on_new_human_message():
    messages = [
        HumanMessage(content="sharpen"),
        ToolMessage(content="sharpen succeeded", tool_call_id="call_1"),
        HumanMessage(content="now crop it"),
    ]

    assert _build_retrieval_query(messages) == "now crop it"


def test_retrieve_node_calls_search_and_returns_candidates(sample_settings, fake_embeddings):
    ingest.build_index(sample_settings)
    state = {"messages": [HumanMessage(content="select a rectangle")], "candidates": []}

    result = retrieve_node(state, settings=sample_settings)

    # The fixture corpus only has 5 procedures, fewer than the default top_k.
    assert len(result["candidates"]) == 5
    assert fake_embeddings.embed_query_calls == ["select a rectangle"]


def test_agent_node_binds_tools_from_candidates_and_appends_response(
    sample_settings, patch_chat_client, make_fake_chat_client
):
    response = AIMessage(content="Done!")
    chat = patch_chat_client(make_fake_chat_client([response]))

    state = {
        "messages": [HumanMessage(content="flatten the image")],
        "candidates": [_scored("gimp-image-flatten", "Flatten the image.")],
    }

    result = agent_node(state, settings=sample_settings)

    assert result["messages"] == [response]
    assert len(chat.bind_tools_calls) == 1
    assert chat.bind_tools_calls[0][0]["name"] == "gimp-image-flatten"
    # system prompt is prepended fresh, not persisted into state
    invoked_messages = chat.invoke_calls[0]
    assert invoked_messages[0].content == graph_module.SYSTEM_PROMPT
    assert invoked_messages[1:] == state["messages"]


def test_agent_node_skips_bind_tools_when_no_candidates(
    sample_settings, patch_chat_client, make_fake_chat_client
):
    response = AIMessage(content="I don't have a procedure for that.")
    chat = patch_chat_client(make_fake_chat_client([response]))

    state = {"messages": [HumanMessage(content="what's the weather?")], "candidates": []}

    agent_node(state, settings=sample_settings)

    assert chat.bind_tools_calls == []


def test_agent_node_only_honors_first_tool_call_if_model_proposes_several(
    sample_settings, patch_chat_client, make_fake_chat_client
):
    response = AIMessage(
        content="",
        tool_calls=[
            {"name": "gimp-sharpen", "args": {}, "id": "call_1"},
            {"name": "gimp-crop", "args": {}, "id": "call_2"},
        ],
    )
    patch_chat_client(make_fake_chat_client([response]))

    state = {"messages": [HumanMessage(content="sharpen and crop")], "candidates": []}

    result = agent_node(state, settings=sample_settings)

    assert len(result["messages"][0].tool_calls) == 1
    assert result["messages"][0].tool_calls[0]["name"] == "gimp-sharpen"


def test_full_graph_reproduces_sharpen_then_crop_scenario(
    sample_settings, fake_embeddings, patch_chat_client, make_fake_chat_client
):
    ingest.build_index(sample_settings)

    responses = [
        AIMessage(content="", tool_calls=[{"name": "gimp-image-flatten", "args": {}, "id": "call_1"}]),
        AIMessage(content="", tool_calls=[{"name": "gimp-image-resize", "args": {}, "id": "call_2"}]),
        AIMessage(content="All done — flattened and resized."),
    ]
    patch_chat_client(make_fake_chat_client(responses))

    graph = build_graph(sample_settings)
    config = {"configurable": {"thread_id": "thread-a"}}

    step1 = graph.invoke({"messages": [HumanMessage(content="flatten then resize")]}, config=config)
    assert step1["messages"][-1].tool_calls[0]["name"] == "gimp-image-flatten"

    step2 = graph.invoke(
        {"messages": [ToolMessage(content="flatten ok", tool_call_id="call_1")]}, config=config
    )
    assert step2["messages"][-1].tool_calls[0]["name"] == "gimp-image-resize"

    step3 = graph.invoke(
        {"messages": [ToolMessage(content="resize ok", tool_call_id="call_2")]}, config=config
    )
    assert not step3["messages"][-1].tool_calls
    assert step3["messages"][-1].content == "All done — flattened and resized."

    # Full history persisted across all three turns on this thread.
    assert len(step3["messages"]) == 6


def test_separate_thread_ids_do_not_share_history(
    sample_settings, fake_embeddings, patch_chat_client, make_fake_chat_client
):
    ingest.build_index(sample_settings)
    patch_chat_client(make_fake_chat_client([AIMessage(content="hi from thread a"), AIMessage(content="hi from thread b")]))

    graph = build_graph(sample_settings)

    graph.invoke({"messages": [HumanMessage(content="hello")]}, config={"configurable": {"thread_id": "a"}})
    result_b = graph.invoke(
        {"messages": [HumanMessage(content="hello")]}, config={"configurable": {"thread_id": "b"}}
    )

    assert len(result_b["messages"]) == 2  # only thread b's own human + AI message


def _scored(name: str, blurb: str) -> ScoredProcedure:
    return ScoredProcedure(procedure=PDBProcedure(name=name, proc_type="PLUGIN", blurb=blurb), distance=0.1)
