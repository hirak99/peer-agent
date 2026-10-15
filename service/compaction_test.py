"""Tests for margin-triggered history compaction helpers."""

import pathlib

from fastapi import testclient

from service import chat_turn
from service import compaction
from service import event_log
from service import log_events
from service import server
from service import store
from service.backends import stub_backend

from typing import Any


def test_split_keeps_tool_pairs_intact() -> None:
    """A split never leaves a tool_result without its tool_use."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}],
        },
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    old, recent = compaction.split_for_compaction(messages, 3)
    assert recent[0] == {"role": "user", "content": "u1"}
    assert old[-1] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "out"}],
    }


def test_needs_compaction_margin_math() -> None:
    """Est + reserve + margin over the window triggers."""
    assert compaction.needs_compaction(800, 100, 100, 1000) is False
    assert compaction.needs_compaction(801, 100, 100, 1000) is True
    assert compaction.needs_compaction(10**9, 0, 0, None) is False


def test_effective_margin_floors_at_five_percent_of_window() -> None:
    """A fixed margin stays thin on large windows; the floor scales."""
    assert (
        compaction.resolve_effective_margin(
            compaction.CompactionConfig(margin_tokens=20000), 1_000_000
        )
        == 50000
    )
    assert (
        compaction.resolve_effective_margin(
            compaction.CompactionConfig(margin_tokens=200000), 1_000_000
        )
        == 200000
    )
    assert (
        compaction.resolve_effective_margin(
            compaction.CompactionConfig(margin_tokens=0), 10
        )
        == 0
    )
    assert (
        compaction.resolve_effective_margin(
            compaction.CompactionConfig(margin_tokens=20000), None
        )
        == 20000
    )


def test_proactive_compaction_rewrites_history(tmp_path: pathlib.Path) -> None:
    """A tiny window override forces summarization on the next turn."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    backend = stub_backend.StubBackend()
    config = compaction.CompactionConfig(
        margin_tokens=0, keep_messages=2, context_window_override=10
    )
    app = server.build_app(
        backend, conversations, events, "sys", compaction_config=config
    )
    client = testclient.TestClient(app)
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    # Second turn exceeds the 10-token window, so old turns are summarized.
    second: dict[str, Any] = client.post(
        "/chat",
        json={
            "prompt": "two",
            "conversation_id": first["conversation_id"],
            "cwd": str(tmp_path),
        },
    ).json()
    assert second["conversation_id"] == first["conversation_id"]
    saved = conversations.load(first["conversation_id"])
    assert saved[0]["role"] == "user"
    assert "summary" in str(saved[0]["content"])


def test_compact_flag_inhibits(tmp_path: pathlib.Path) -> None:
    """compact=false skips even a blown margin."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    backend = stub_backend.StubBackend()
    config = compaction.CompactionConfig(
        margin_tokens=0, keep_messages=2, context_window_override=10
    )
    app = server.build_app(
        backend, conversations, events, "sys", compaction_config=config
    )
    client = testclient.TestClient(app)
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    second: dict[str, Any] = client.post(
        "/chat",
        json={
            "prompt": "two",
            "conversation_id": first["conversation_id"],
            "cwd": str(tmp_path),
            "compact": False,
        },
    ).json()
    assert second["conversation_id"] == first["conversation_id"]
    saved = conversations.load(first["conversation_id"])
    assert "summary" not in str(saved[0].get("content", ""))


def test_manual_compact_endpoint(tmp_path: pathlib.Path) -> None:
    """POST /compact summarizes stored history on demand."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    backend = stub_backend.StubBackend()
    app = server.build_app(
        backend,
        conversations,
        events,
        "sys",
        compaction_config=compaction.CompactionConfig(keep_messages=2),
    )
    client = testclient.TestClient(app)
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    cid = first["conversation_id"]
    client.post(
        "/chat", json={"prompt": "two", "conversation_id": cid, "cwd": str(tmp_path)}
    )
    client.post(
        "/chat", json={"prompt": "three", "conversation_id": cid, "cwd": str(tmp_path)}
    )
    body: dict[str, Any] = client.post("/compact", json={"conversation_id": cid}).json()
    assert body["compacted"] is True
    assert body["post_messages"] < body["pre_messages"]
    assert client.post("/compact", json={"conversation_id": "nope"}).status_code == 404
    assert chat_turn.compact_conversation is not None
    assert log_events.CompactionEvent is not None
