"""Tests for warm-cache in-context compaction."""

# pyright: reportPrivateUsage=false

import pathlib
from collections.abc import Callable

from fastapi import testclient

from service import compaction
from service import event_log
from service import in_context_compaction
from service import server
from service import store
from service.backends import base
from service.backends import stub_backend

from typing import Any, override


def _history() -> list[dict[str, Any]]:
    """Four plain turns: splittable with keep=2."""
    return [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "uno"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "dos"},
    ]


def test_attempt_summary_leaves_history_untouched() -> None:
    """The transient exchange is popped; the summary comes back."""
    history = _history()
    summary = in_context_compaction.attempt_summary(
        stub_backend.StubBackend(), history, keep=2
    )
    assert summary is not None and "stub-response" in summary
    assert history == _history()


def test_attempt_summary_declines_without_split_point() -> None:
    """History shorter than keep yields None (caller falls back)."""
    history = _history()
    assert (
        in_context_compaction.attempt_summary(
            stub_backend.StubBackend(), history, keep=10
        )
        is None
    )
    assert history == _history()


class _FailingCompleteBackend(stub_backend.StubBackend):
    """complete() blows up only for the summary turn; normal turns work."""

    @override
    def complete(
        self,
        messages: list[dict[str, Any]],
        context: base.ToolContext,
        host_max_tool_turns: int | None = None,
        on_tool_event: Callable[[base.ToolEvent], None] | None = None,
    ) -> base.Completion:
        if messages and messages[-1].get("content") == in_context_compaction.SUMMARY_REQUEST:
            raise RuntimeError("boom")
        return super().complete(messages, context, host_max_tool_turns)


class _EmptyReplyBackend(stub_backend.StubBackend):
    """complete() succeeds with blank text."""

    @override
    def complete(
        self,
        messages: list[dict[str, Any]],
        context: base.ToolContext,
        host_max_tool_turns: int | None = None,
        on_tool_event: Callable[[base.ToolEvent], None] | None = None,
    ) -> base.Completion:
        messages.append({"role": "assistant", "content": "   "})
        return base.Completion(text="   ", usage=None, tool_events=[])


def test_attempt_summary_restores_history_on_failure() -> None:
    """A failed turn returns None and pops the transient request."""
    history = _history()
    assert (
        in_context_compaction.attempt_summary(
            _FailingCompleteBackend(), history, keep=2
        )
        is None
    )
    assert history == _history()


def test_attempt_summary_rejects_empty_reply() -> None:
    """Blank replies fall back instead of splicing an empty summary."""
    history = _history()
    assert (
        in_context_compaction.attempt_summary(_EmptyReplyBackend(), history, keep=2)
        is None
    )
    assert history == _history()


def _app(
    tmp_path: pathlib.Path, backend: base.Backend
) -> testclient.TestClient:
    """Two-turn app with in-context compaction forced by a tiny window."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    config = compaction.CompactionConfig(
        margin_tokens=0,
        keep_messages=2,
        context_window_override=10,
        in_context=True,
    )
    app = server.build_app(backend, conversations, events, "sys", compaction_config=config)
    return testclient.TestClient(app)


def test_proactive_prefers_in_context(tmp_path: pathlib.Path) -> None:
    """With the flag on, the preamble comes from the in-context turn."""
    client = _app(tmp_path, stub_backend.StubBackend())
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
    conversations = store.ConversationStore(tmp_path / "conversations")
    saved = conversations.load(first["conversation_id"])
    assert "stub-response" in str(saved[0]["content"])


def test_proactive_falls_back_to_summarize(tmp_path: pathlib.Path) -> None:
    """A failed in-context turn still compacts via summarize()."""
    client = _app(tmp_path, _FailingCompleteBackend())
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    cid = first["conversation_id"]
    client.post("/chat", json={"prompt": "two", "conversation_id": cid, "cwd": str(tmp_path)})
    conversations = store.ConversationStore(tmp_path / "conversations")
    saved = conversations.load(cid)
    assert "stub-summary" in str(saved[0]["content"])


def test_manual_compact_uses_in_context(tmp_path: pathlib.Path) -> None:
    """POST /compact honors the flag as well."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    backend = stub_backend.StubBackend()
    app = server.build_app(
        backend,
        conversations,
        events,
        "sys",
        compaction_config=compaction.CompactionConfig(keep_messages=2, in_context=True),
    )
    client = testclient.TestClient(app)
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    cid = first["conversation_id"]
    client.post("/chat", json={"prompt": "two", "conversation_id": cid, "cwd": str(tmp_path)})
    client.post("/chat", json={"prompt": "three", "conversation_id": cid, "cwd": str(tmp_path)})
    body: dict[str, Any] = client.post("/compact", json={"conversation_id": cid}).json()
    assert body["compacted"] is True
    assert "stub-response" in str(conversations.load(cid)[0]["content"])
