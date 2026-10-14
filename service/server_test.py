"""Tests for the peer-agent HTTP service using the stub backend."""

import json
import pathlib

from fastapi import testclient

from service import event_log
from service import server
from service import store
from service.backends import stub_backend

from typing import Any


def _build_client(tmp_path: pathlib.Path) -> testclient.TestClient:
    app = server.build_app(
        stub_backend.StubBackend(),
        store.ConversationStore(tmp_path / "conversations"),
        event_log.EventLog(tmp_path / "logs"),
    )
    return testclient.TestClient(app)


def test_health(tmp_path: pathlib.Path) -> None:
    """GET /health reports ok."""
    assert _build_client(tmp_path).get("/health").json() == {"status": "ok"}


def test_chat_new_conversation(tmp_path: pathlib.Path) -> None:
    """POST /chat without an id starts a conversation and persists history."""
    client = _build_client(tmp_path)
    body: dict[str, Any] = client.post("/chat", json={"prompt": "hello"}).json()
    assert body["text"] == "stub-response to: hello"
    assert body["conversation_id"]
    saved = store.ConversationStore(tmp_path / "conversations").load(
        body["conversation_id"]
    )
    assert [message["role"] for message in saved] == ["user", "assistant"]
    assert (tmp_path / "logs" / f"{body['conversation_id']}.jsonl").exists()


def test_chat_continuation(tmp_path: pathlib.Path) -> None:
    """POST /chat with an id continues the stored thread."""
    client = _build_client(tmp_path)
    first: dict[str, Any] = client.post("/chat", json={"prompt": "one"}).json()
    second: dict[str, Any] = client.post(
        "/chat",
        json={"prompt": "two", "conversation_id": first["conversation_id"]},
    ).json()
    assert second["conversation_id"] == first["conversation_id"]
    assert second["text"] == "stub-response to: two"


def test_chat_unknown_conversation_id_is_404(tmp_path: pathlib.Path) -> None:
    """POST /chat with a bogus id is rejected so callers start fresh."""
    response = _build_client(tmp_path).post(
        "/chat", json={"prompt": "hi", "conversation_id": "nope"}
    )
    assert response.status_code == 404


def test_chat_stream(tmp_path: pathlib.Path) -> None:
    """POST /chat/stream yields token events plus a done event with the id."""
    client = _build_client(tmp_path)
    response = client.post("/chat/stream", json={"prompt": "stream me"})
    assert response.status_code == 200
    texts: list[str] = []
    done_id = ""
    for line in response.text.splitlines():
        if not line.startswith("data: "):
            continue
        event: dict[str, Any] = json.loads(line[len("data: ") :])
        if event.get("type") == "token":
            texts.append(str(event.get("text", "")))
        elif event.get("type") == "done":
            done_id = str(event.get("conversation_id", ""))
    assert done_id
    assert "stub-response" in "".join(texts)
    follow: dict[str, Any] = client.post(
        "/chat", json={"prompt": "again", "conversation_id": done_id}
    ).json()
    assert follow["conversation_id"] == done_id
