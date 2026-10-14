"""Tests for the peer-agent HTTP service using the stub backend."""

import json
import pathlib

import pydantic
from fastapi import testclient

from service import event_log
from service import log_events
from service import server
from service import store
from service.backends import stub_backend

from typing import Any

_TEST_SYSTEM_PROMPT = "test system prompt"


def _build_client(
    tmp_path: pathlib.Path, system_prompt: str = _TEST_SYSTEM_PROMPT
) -> testclient.TestClient:
    app = server.build_app(
        stub_backend.StubBackend(),
        store.ConversationStore(tmp_path / "conversations"),
        event_log.EventLog(tmp_path / "logs"),
        system_prompt,
    )
    return testclient.TestClient(app)


def _parse_log(tmp_path: pathlib.Path, conversation_id: str) -> list[Any]:
    adapter = pydantic.TypeAdapter(log_events.LogEvent)
    lines = (tmp_path / "logs" / f"{conversation_id}.jsonl").read_text().splitlines()
    return [adapter.validate_json(line) for line in lines]


def test_health(tmp_path: pathlib.Path) -> None:
    """GET /health reports ok."""
    assert _build_client(tmp_path).get("/health").json() == {"status": "ok"}


def test_chat_new_conversation(tmp_path: pathlib.Path) -> None:
    """POST /chat without an id starts a conversation and persists history."""
    client = _build_client(tmp_path)
    body: dict[str, Any] = client.post(
        "/chat", json={"prompt": "hello", "cwd": str(tmp_path)}
    ).json()
    assert body["text"] == "stub-response to: hello"
    assert body["conversation_id"]
    saved = store.ConversationStore(tmp_path / "conversations").load(
        body["conversation_id"]
    )
    assert [message["role"] for message in saved] == ["user", "assistant"]
    assert (tmp_path / "logs" / f"{body['conversation_id']}.jsonl").exists()


def test_log_first_line_is_system_prompt(tmp_path: pathlib.Path) -> None:
    """A new conversation log opens with a conversation_start system event."""
    client = _build_client(tmp_path)
    body: dict[str, Any] = client.post(
        "/chat", json={"prompt": "hello", "cwd": str(tmp_path)}
    ).json()
    parsed = _parse_log(tmp_path, body["conversation_id"])
    first = parsed[0]
    assert isinstance(first, log_events.SystemPromptEvent)
    assert first.reason == "conversation_start"
    assert first.system_prompt == _TEST_SYSTEM_PROMPT


def test_log_response_carries_model_identity(tmp_path: pathlib.Path) -> None:
    """Response events record the producing model and backend."""
    client = _build_client(tmp_path)
    body: dict[str, Any] = client.post(
        "/chat", json={"prompt": "hello", "cwd": str(tmp_path)}
    ).json()
    parsed = _parse_log(tmp_path, body["conversation_id"])
    responses = [
        event for event in parsed if isinstance(event, log_events.ResponseEvent)
    ]
    assert len(responses) == 1
    assert responses[0].model == "stub"
    assert responses[0].backend == "stub"


def test_changed_system_prompt_emits_changed_event(tmp_path: pathlib.Path) -> None:
    """A rebuilt app with a new prompt logs a changed event on next turn."""
    first_client = _build_client(tmp_path)
    first: dict[str, Any] = first_client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    second_client = _build_client(tmp_path, system_prompt="new system prompt")
    second: dict[str, Any] = second_client.post(
        "/chat",
        json={
            "prompt": "two",
            "conversation_id": first["conversation_id"],
            "cwd": str(tmp_path),
        },
    ).json()
    assert second["conversation_id"] == first["conversation_id"]
    parsed = _parse_log(tmp_path, first["conversation_id"])
    system_events = [
        event for event in parsed if isinstance(event, log_events.SystemPromptEvent)
    ]
    assert [event.reason for event in system_events] == [
        "conversation_start",
        "changed",
    ]
    assert system_events[1].system_prompt == "new system prompt"


def test_chat_continuation(tmp_path: pathlib.Path) -> None:
    """POST /chat with an id continues the stored thread."""
    client = _build_client(tmp_path)
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    second: dict[str, Any] = client.post(
        "/chat",
        json={
            "prompt": "two",
            "conversation_id": first["conversation_id"],
            "cwd": str(tmp_path),
        },
    ).json()
    assert second["conversation_id"] == first["conversation_id"]
    assert second["text"] == "stub-response to: two"


def test_chat_unknown_conversation_id_is_404(tmp_path: pathlib.Path) -> None:
    """POST /chat with a bogus id is rejected so callers start fresh."""
    response = _build_client(tmp_path).post(
        "/chat",
        json={"prompt": "hi", "conversation_id": "nope", "cwd": str(tmp_path)},
    )
    assert response.status_code == 404


def test_chat_missing_cwd_is_422(tmp_path: pathlib.Path) -> None:
    """POST /chat without the required cwd field is rejected."""
    response = _build_client(tmp_path).post("/chat", json={"prompt": "hi"})
    assert response.status_code == 422


def test_chat_null_cwd_disables_filesystem(tmp_path: pathlib.Path) -> None:
    """POST /chat with null cwd succeeds; the turn simply has no workdir."""
    client = _build_client(tmp_path)
    body: dict[str, Any] = client.post(
        "/chat", json={"prompt": "hi", "cwd": None}
    ).json()
    assert body["text"] == "stub-response to: hi"


def test_chat_bad_cwd_is_400(tmp_path: pathlib.Path) -> None:
    """POST /chat with a relative or missing cwd is rejected."""
    client = _build_client(tmp_path)
    assert (
        client.post("/chat", json={"prompt": "hi", "cwd": "relative/path"}).status_code
        == 400
    )
    assert (
        client.post(
            "/chat", json={"prompt": "hi", "cwd": str(tmp_path / "missing")}
        ).status_code
        == 400
    )


def test_chat_stream(tmp_path: pathlib.Path) -> None:
    """POST /chat/stream yields token events plus a done event with the id."""
    client = _build_client(tmp_path)
    response = client.post(
        "/chat/stream", json={"prompt": "stream me", "cwd": str(tmp_path)}
    )
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
        "/chat",
        json={"prompt": "again", "conversation_id": done_id, "cwd": str(tmp_path)},
    ).json()
    assert follow["conversation_id"] == done_id
