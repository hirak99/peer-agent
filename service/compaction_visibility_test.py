"""Tests for host-visible compaction (idea 001)."""

# pyright: reportPrivateUsage=false

import json
import pathlib
import tempfile

from fastapi import testclient

from service import chat_turn
from service import event_log
from service import models
from service import server
from service import store
from service.backends import base
from service.backends import stub_backend

from typing import Any


def _client(tmp: pathlib.Path) -> testclient.TestClient:
    app = server.build_app(
        stub_backend.StubBackend(),
        store.ConversationStore(tmp / "conversations"),
        event_log.EventLog(tmp / "logs"),
        "test prompt",
    )
    return testclient.TestClient(app)


def _long_conversation(
    client: testclient.TestClient, tmp: pathlib.Path
) -> str:
    """Builds >20 messages of history so forced compaction has content."""
    cid: str | None = None
    for i in range(12):
        body: dict[str, Any] = client.post(
            "/chat",
            json={
                "prompt": f"msg {i}",
                "cwd": str(tmp),
                **({"conversation_id": cid} if cid else {}),
            },
        ).json()
        cid = str(body["conversation_id"])
    assert cid is not None
    return cid


def test_chat_omits_compaction_by_default(tmp_path: pathlib.Path) -> None:
    """A normal turn carries no compaction key; hint/usage still present."""
    body: dict[str, Any] = (
        _client(tmp_path)
        .post("/chat", json={"prompt": "hello", "cwd": str(tmp_path)})
        .json()
    )
    assert "compaction" not in body
    assert "hint" in body
    assert "usage" in body


def test_chat_emits_forced_reason(tmp_path: pathlib.Path) -> None:
    """A forced compaction surfaces as reason forced."""
    client = _client(tmp_path)
    cid = _long_conversation(client, tmp_path)
    body: dict[str, Any] = client.post(
        "/chat",
        json={
            "prompt": "force me",
            "conversation_id": cid,
            "cwd": str(tmp_path),
            "compact": True,
        },
    ).json()
    assert body["compaction"] == {"reason": "forced"}


def test_stream_done_omits_by_default(tmp_path: pathlib.Path) -> None:
    """Stream done event omits compaction when nothing happened."""
    resp = _client(tmp_path).post(
        "/chat/stream", json={"prompt": "hi", "cwd": str(tmp_path)}
    )
    done: dict[str, Any] = {}
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("type") == "done":
                done = event
    assert done
    assert "compaction" not in done


def test_stream_done_emits_forced(tmp_path: pathlib.Path) -> None:
    """Stream done event carries the forced reason."""
    client = _client(tmp_path)
    cid = _long_conversation(client, tmp_path)
    resp = client.post(
        "/chat/stream",
        json={
            "prompt": "force stream",
            "conversation_id": cid,
            "cwd": str(tmp_path),
            "compact": True,
        },
    )
    done: dict[str, Any] = {}
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            event = json.loads(line[6:])
            if event.get("type") == "done":
                done = event
    assert done.get("compaction") == {"reason": "forced"}


def test_recovery_wins_and_renames_to_overflow() -> None:
    """Later recovery record wins; error_recovery surfaces as overflow."""
    proactive = base.CompactionRecord(10, 5, 100, 50, "margin")
    recovery = base.CompactionRecord(10, 5, 100, 50, "error_recovery")
    assert chat_turn._host_compaction_info(proactive, recovery) == (
        models.CompactionInfo(reason="overflow")
    )
    assert chat_turn._host_compaction_info(proactive, None) == (
        models.CompactionInfo(reason="auto")
    )
    assert chat_turn._host_compaction_info(None, None) is None


def test_manual_reason_never_surfaces() -> None:
    """A manual record (own endpoint) yields no /chat signal."""
    manual = base.CompactionRecord(10, 5, 100, 50, "manual")
    assert chat_turn._host_compaction_info(manual, None) is None


def test_base_stream_copies_compaction() -> None:
    """Default stream_text propagates completion.compaction into meta."""

    class _RecoveryStub(stub_backend.StubBackend):
        def complete(  # type: ignore[override]
            self,
            messages: list[dict[str, Any]],
            context: base.ToolContext,
            host_max_tool_turns: int | None = None,
            on_tool_event: Any = None,
        ) -> base.Completion:
            completion = super().complete(
                messages, context, host_max_tool_turns, on_tool_event
            )
            completion.compaction = base.CompactionRecord(
                10, 5, 100, 50, "error_recovery"
            )
            return completion

    meta: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as _:
        list(
            _RecoveryStub().stream_text(
                [], [], base.ToolContext(workdir=None), None, meta
            )
        )
    record = meta.get("compaction")
    assert isinstance(record, base.CompactionRecord)
    assert record.reason == "error_recovery"
