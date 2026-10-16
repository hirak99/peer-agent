"""Tests for live per-call tool event logging."""

# pyright: reportPrivateUsage=false

import json
import pathlib
from collections.abc import Callable

import pytest

from service import chat_turn
from service import event_log
from service import store
from service.backends import base
from service.backends import stub_backend

from typing import Any, override


class _LiveBackend(stub_backend.StubBackend):
    """Reports one tool event through the sink, then replies like the stub."""

    @override
    def complete(
        self,
        messages: list[dict[str, Any]],
        context: base.ToolContext,
        host_max_tool_turns: int | None = None,
        on_tool_event: Callable[[base.ToolEvent], None] | None = None,
    ) -> base.Completion:
        event = base.ToolEvent(name="t", tool_input={"c": "x"}, output="ok")
        if on_tool_event is not None:
            on_tool_event(event)
        completion = super().complete(messages, context, host_max_tool_turns)
        completion.tool_events.append(event)
        return completion


class _DyingBackend(stub_backend.StubBackend):
    """Reports one tool event through the sink, then blows up mid-turn."""

    @override
    def complete(
        self,
        messages: list[dict[str, Any]],
        context: base.ToolContext,
        host_max_tool_turns: int | None = None,
        on_tool_event: Callable[[base.ToolEvent], None] | None = None,
    ) -> base.Completion:
        if on_tool_event is not None:
            on_tool_event(
                base.ToolEvent(name="t", tool_input={"c": "x"}, output="partial")
            )
        raise RuntimeError("boom")


def _log_lines(tmp_path: pathlib.Path, cid: str) -> list[dict[str, Any]]:
    """Reads back the recorded JSONL lines for one conversation."""
    path = tmp_path / "logs" / f"{cid}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _state(cid: str) -> chat_turn.TurnState:
    """Minimal live turn state (history already holds the user prompt)."""
    return chat_turn.TurnState(
        conversation_id=cid,
        history=[{"role": "user", "content": "hi"}],
        context=base.ToolContext(workdir=None),
        cap=None,
    )


def test_tool_event_recorded_once(tmp_path: pathlib.Path) -> None:
    """Live sink plus log_completion must not double-record the call."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    cid = conversations.new_id()
    chat_turn.finish_turn(conversations, events, _LiveBackend(), _state(cid))
    lines = _log_lines(tmp_path, cid)
    assert [line["type"] for line in lines].count("tool_call") == 1
    assert [line["type"] for line in lines].count("tool_result") == 1
    assert lines[-1]["type"] == "response"


def test_mid_turn_failure_keeps_executed_calls(tmp_path: pathlib.Path) -> None:
    """Calls settled before a mid-turn blowup stay in the log."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    cid = conversations.new_id()
    with pytest.raises(RuntimeError, match="boom"):
        chat_turn.finish_turn(conversations, events, _DyingBackend(), _state(cid))
    lines = _log_lines(tmp_path, cid)
    assert [line["type"] for line in lines].count("tool_call") == 1
    assert [line["type"] for line in lines].count("tool_result") == 1
    assert all(line["type"] != "response" for line in lines)
