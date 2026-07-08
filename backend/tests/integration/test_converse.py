from __future__ import annotations

from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient

from backend.main import app
from backend.rag import ingest


def test_converse_sharpen_then_crop_scenario(sample_settings, fake_embeddings, patch_chat_client, make_fake_chat_client, monkeypatch):
    ingest.build_index(sample_settings)
    monkeypatch.setattr("backend.main.get_settings", lambda: sample_settings)

    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "gimp-image-flatten", "args": {"image": 1}, "id": "call_1"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "gimp-image-resize", "args": {"image": 1}, "id": "call_2"},
            ],
        ),
        AIMessage(content="I have flattened and resized the image as requested."),
    ]
    patch_chat_client(make_fake_chat_client(responses))

    with TestClient(app) as client:
        thread_id = client.post("/refresh-conversation").json()["thread_id"]

        first = client.post(
            "/converse",
            json={
                "thread_id": thread_id,
                "message": "flatten this image and then resize it",
                "context": {"image_id": 1},
            },
        )
        assert first.status_code == 200
        first_body = first.json()
        assert first_body["done"] is False
        assert first_body["tool_call"] == {"procedure": "gimp-image-flatten", "args": {"image": 1}}

        second = client.post(
            "/converse",
            json={
                "thread_id": thread_id,
                "tool_result": {"procedure": "gimp-image-flatten", "ok": True, "result": [1]},
                "context": {"image_id": 1},
            },
        )
        assert second.status_code == 200
        second_body = second.json()
        assert second_body["done"] is False
        assert second_body["tool_call"] == {"procedure": "gimp-image-resize", "args": {"image": 1}}

        third = client.post(
            "/converse",
            json={
                "thread_id": thread_id,
                "tool_result": {"procedure": "gimp-image-resize", "ok": True, "result": []},
                "context": {"image_id": 1},
            },
        )
        assert third.status_code == 200
        third_body = third.json()
        assert third_body["done"] is True
        assert third_body["tool_call"] is None
        assert third_body["message"] == "I have flattened and resized the image as requested."


def test_converse_requires_message_or_tool_result(sample_settings, fake_embeddings, patch_chat_client, make_fake_chat_client, monkeypatch):
    monkeypatch.setattr("backend.main.get_settings", lambda: sample_settings)
    patch_chat_client(make_fake_chat_client([]))

    with TestClient(app) as client:
        thread_id = client.post("/refresh-conversation").json()["thread_id"]
        response = client.post("/converse", json={"thread_id": thread_id})

    assert response.status_code == 400


def test_converse_message_without_context_and_failed_tool_result(
    sample_settings, fake_embeddings, patch_chat_client, make_fake_chat_client, monkeypatch
):
    monkeypatch.setattr("backend.main.get_settings", lambda: sample_settings)
    ingest.build_index(sample_settings)
    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": "gimp-image-flatten", "args": {"image": 1}, "id": "call_1"}],
        ),
        AIMessage(content="Sorry, that didn't work — could you check the image is open?"),
    ]
    patch_chat_client(make_fake_chat_client(responses))

    with TestClient(app) as client:
        thread_id = client.post("/refresh-conversation").json()["thread_id"]

        # No `context` this time.
        first = client.post(
            "/converse", json={"thread_id": thread_id, "message": "flatten this image"}
        )
        assert first.status_code == 200

        # Tool failed.
        second = client.post(
            "/converse",
            json={
                "thread_id": thread_id,
                "tool_result": {
                    "procedure": "gimp-image-flatten",
                    "ok": False,
                    "error": "no active image",
                },
            },
        )
        assert second.status_code == 200
        assert second.json()["done"] is True


def test_converse_flattens_gemini_list_style_content(
    sample_settings, fake_embeddings, patch_chat_client, make_fake_chat_client, monkeypatch
):
    # Gemini sometimes returns `content` as a list of blocks (text + a
    # signature block) rather than a plain string — regression test for that.
    monkeypatch.setattr("backend.main.get_settings", lambda: sample_settings)
    patch_chat_client(
        make_fake_chat_client(
            [
                AIMessage(
                    content=[
                        {"type": "text", "text": "All done."},
                        {"type": "signature", "signature": "abc123"},
                    ]
                )
            ]
        )
    )

    with TestClient(app) as client:
        thread_id = client.post("/refresh-conversation").json()["thread_id"]
        response = client.post(
            "/converse", json={"thread_id": thread_id, "message": "anything"}
        )

    assert response.status_code == 200
    assert response.json()["message"] == "All done."


def test_converse_tool_result_without_pending_tool_call_is_rejected(
    sample_settings, fake_embeddings, patch_chat_client, make_fake_chat_client, monkeypatch
):
    monkeypatch.setattr("backend.main.get_settings", lambda: sample_settings)
    patch_chat_client(make_fake_chat_client([]))

    with TestClient(app) as client:
        thread_id = client.post("/refresh-conversation").json()["thread_id"]
        response = client.post(
            "/converse",
            json={
                "thread_id": thread_id,
                "tool_result": {"procedure": "gimp-noop", "ok": True, "result": []},
            },
        )

    assert response.status_code == 400
