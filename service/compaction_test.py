"""Tests for margin-triggered history compaction helpers."""

# pyright: reportPrivateUsage=false

import json
import pathlib

from fastapi import testclient

from service import chat_turn
from service import compaction
from service import event_log
from service import log_events
from service import server
from service import store
from service.backends import stub_backend

from typing import Any, override


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


def test_split_tool_only_tail_still_splits() -> None:
    """A long tool loop with no plain-text tail still yields a split."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": "do the task"}]
    for seq in range(36):
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": f"t{seq}", "name": "bash", "input": {}}],
            }
        )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": f"t{seq}", "content": "out"}],
            }
        )
    old, recent = compaction.split_for_compaction(messages, 20)
    assert old, "expected a split, not a decline"
    assert len(recent) <= 21
    # No severed pair across the boundary.
    assert not (
        compaction._has_tool_use(old[-1]) or compaction._has_tool_result(recent[0])
    )


def test_split_tail_ending_mid_pair_scans_back() -> None:
    """Forward scan running off the end falls back to a backward split."""
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
        },
    ]
    old, recent = compaction.split_for_compaction(messages, 1)
    assert old == [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
    ]
    assert recent == [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {}}],
        },
    ]


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


class _TruncatingBackend(stub_backend.StubBackend):
    """summarize() always hits its output budget."""

    @override
    def summarize(self, messages: list[dict[str, Any]]) -> str:
        raise compaction.SummaryTruncatedError(
            "Summary truncated at the test budget; compaction refused."
        )


class _EmptySummaryBackend(stub_backend.StubBackend):
    """summarize() returns blank text."""

    @override
    def summarize(self, messages: list[dict[str, Any]]) -> str:
        return "   "


def _tiny_window_client(
    tmp_path: pathlib.Path, backend: Any
) -> tuple[testclient.TestClient, store.ConversationStore]:
    """App whose next turn exceeds its input window, forcing compaction."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    config = compaction.CompactionConfig(
        margin_tokens=0, keep_messages=2, context_window_override=10
    )
    app = server.build_app(
        backend, conversations, events, "sys", compaction_config=config
    )
    return testclient.TestClient(app), conversations


def test_proactive_refuses_truncated_summary(tmp_path: pathlib.Path) -> None:
    """A budget-hit summary leaves history untouched; the turn still succeeds."""
    client, conversations = _tiny_window_client(tmp_path, _TruncatingBackend())
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
    saved = conversations.load(first["conversation_id"])
    assert "[Conversation summary" not in json.dumps(saved)


def test_proactive_refuses_empty_summary(tmp_path: pathlib.Path) -> None:
    """A blank summary is never spliced into history."""
    client, conversations = _tiny_window_client(tmp_path, _EmptySummaryBackend())
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    client.post(
        "/chat",
        json={
            "prompt": "two",
            "conversation_id": first["conversation_id"],
            "cwd": str(tmp_path),
        },
    )
    saved = conversations.load(first["conversation_id"])
    assert "[Conversation summary" not in json.dumps(saved)


def test_manual_compact_fails_on_truncated_summary(tmp_path: pathlib.Path) -> None:
    """POST /compact surfaces truncation as a 502 and saves nothing."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    backend = _TruncatingBackend()
    app = server.build_app(
        backend,
        conversations,
        events,
        "sys",
        compaction_config=compaction.CompactionConfig(keep_messages=2),
    )
    client = testclient.TestClient(app, raise_server_exceptions=False)
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    cid = first["conversation_id"]
    client.post(
        "/chat", json={"prompt": "two", "conversation_id": cid, "cwd": str(tmp_path)}
    )
    before = conversations.load(cid)
    response = client.post("/compact", json={"conversation_id": cid})
    assert response.status_code == 502
    assert conversations.load(cid) == before


class _BudgetedStubBackend(stub_backend.StubBackend):
    """Stub carrying a summary budget like the real backends."""

    @override
    def summary_budget(self) -> int:
        return 32


def _compaction_events(tmp_path: pathlib.Path, cid: str) -> list[dict[str, Any]]:
    """Reads compaction events for a conversation from its JSONL log."""
    lines = (tmp_path / "logs" / f"{cid}.jsonl").read_text().splitlines()
    return [
        event
        for line in lines
        if (event := json.loads(line)).get("type") == "compaction"
    ]


def test_manual_compact_snapshots_history(tmp_path: pathlib.Path) -> None:
    """POST /compact backs up the exact pre-compaction history first."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    app = server.build_app(
        _BudgetedStubBackend(),
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
    before = conversations.load(cid)
    compacted_body: dict[str, Any] = client.post(
        "/compact", json={"conversation_id": cid}
    ).json()
    assert compacted_body["compacted"] is True
    snapshots = sorted((tmp_path / "conversations" / "backups" / cid).glob("*.json"))
    assert len(snapshots) == 1
    assert json.loads(snapshots[0].read_text()) == before
    (event,) = _compaction_events(tmp_path, cid)
    assert event["backup_path"] == str(snapshots[0])
    assert event["summary_budget"] == 32


def test_proactive_compaction_snapshots_history(tmp_path: pathlib.Path) -> None:
    """A margin-triggered compaction also leaves a restorable snapshot."""
    client = _tiny_window_client(tmp_path, stub_backend.StubBackend())[0]
    first: dict[str, Any] = client.post(
        "/chat", json={"prompt": "one", "cwd": str(tmp_path)}
    ).json()
    cid = first["conversation_id"]
    client.post(
        "/chat",
        json={"prompt": "two", "conversation_id": cid, "cwd": str(tmp_path)},
    )
    snapshots = sorted((tmp_path / "conversations" / "backups" / cid).glob("*.json"))
    assert len(snapshots) == 1
    (event,) = _compaction_events(tmp_path, cid)
    assert event["backup_path"] == str(snapshots[0])


class _UsageStubBackend(stub_backend.StubBackend):
    """Stub reporting a fixed billed input size per turn."""

    def __init__(
        self,
        prompt_tokens: int,
        cached_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        total_input_tokens: int | None = None,
    ) -> None:
        self._prompt_tokens = prompt_tokens
        self._cached_tokens = cached_tokens
        self._cache_creation_tokens = cache_creation_tokens
        self._total_input_tokens = total_input_tokens

    def complete(  # type: ignore[override]
        self,
        messages: list[dict[str, Any]],
        context: Any,
        host_max_tool_turns: Any = None,
        on_tool_event: Any = None,
    ) -> Any:
        completion = super().complete(
            messages, context, host_max_tool_turns, on_tool_event
        )
        completion.usage = log_events.TokenUsage(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=10,
            total_tokens=self._prompt_tokens + 10,
            cached_prompt_tokens=self._cached_tokens,
            cache_creation_input_tokens=self._cache_creation_tokens,
            total_input_tokens=self._total_input_tokens,
        )
        return completion


def _trigger_client(
    tmp_path: pathlib.Path, backend: Any, trigger_tokens: int
) -> tuple[testclient.TestClient, store.ConversationStore]:
    """App whose compaction trigger is an exact billed-input threshold."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    events = event_log.EventLog(tmp_path / "logs")
    config = compaction.CompactionConfig(trigger_tokens=trigger_tokens, keep_messages=2)
    app = server.build_app(
        backend, conversations, events, "sys", compaction_config=config
    )
    return testclient.TestClient(app), conversations


def _seed_turns(
    client: testclient.TestClient, tmp_path: pathlib.Path, *prompts: str
) -> str:
    """Runs turns on one thread; returns the conversation id."""
    cid = ""
    for index, prompt in enumerate(prompts):
        body: dict[str, Any] = client.post(
            "/chat",
            json={
                "prompt": prompt,
                "cwd": str(tmp_path),
                **({"conversation_id": cid} if cid else {}),
            },
        ).json()
        cid = str(body["conversation_id"])
        assert index < 10
    return cid


def test_trigger_compacts_when_last_billed_reaches_threshold(
    tmp_path: pathlib.Path,
) -> None:
    """Last prompt_tokens >= trigger compacts the past before the new turn."""
    client, conversations = _trigger_client(tmp_path, _UsageStubBackend(50), 100)
    cid = _seed_turns(client, tmp_path, "one", "two", "three")
    # Simulate a heavy billed turn; the backend itself reports small usage.
    conversations.save_last_prompt_tokens(cid, 150)
    body: dict[str, Any] = client.post(
        "/chat",
        json={"prompt": "four", "conversation_id": cid, "cwd": str(tmp_path)},
    ).json()
    assert body["compaction"] == {"reason": "auto"}
    saved = conversations.load(cid)
    assert "[Conversation summary" in json.dumps(saved)
    # The new prompt survives compaction and gets its reply.
    assert saved[-2] == {"role": "user", "content": "four"}
    assert body["conversation_id"] == cid


def test_trigger_below_threshold_skips_compaction(tmp_path: pathlib.Path) -> None:
    """Last prompt_tokens under the trigger leaves history untouched."""
    client, conversations = _trigger_client(tmp_path, _UsageStubBackend(50), 100)
    cid = _seed_turns(client, tmp_path, "one", "two", "three")
    conversations.save_last_prompt_tokens(cid, 60)
    body: dict[str, Any] = client.post(
        "/chat",
        json={"prompt": "four", "conversation_id": cid, "cwd": str(tmp_path)},
    ).json()
    assert "compaction" not in body
    saved = conversations.load(cid)
    assert "[Conversation summary" not in json.dumps(saved)


def test_trigger_without_usage_warns_and_falls_back(
    tmp_path: pathlib.Path, caplog: Any
) -> None:
    """Trigger set but nothing billed: warn every turn, keep estimating."""
    client, _ = _trigger_client(tmp_path, stub_backend.StubBackend(), 100)
    with caplog.at_level("WARNING", logger="service.chat_turn"):
        _seed_turns(client, tmp_path, "one", "two")
    warnings = [
        record
        for record in caplog.records
        if "falling back to chars/4 estimate" in record.getMessage()
    ]
    assert len(warnings) == 2


def test_trigger_with_usage_never_warns(tmp_path: pathlib.Path, caplog: Any) -> None:
    """Exact counts available: only the first turn still falls back."""
    client, _ = _trigger_client(tmp_path, _UsageStubBackend(50), 10**9)
    with caplog.at_level("WARNING", logger="service.chat_turn"):
        _seed_turns(client, tmp_path, "one", "two")
    warnings = [
        record
        for record in caplog.records
        if "falling back to chars/4 estimate" in record.getMessage()
    ]
    # Turn one has no billed usage yet; turn two decides from turn one's.
    assert len(warnings) == 1


def test_last_prompt_tokens_persisted_per_turn(tmp_path: pathlib.Path) -> None:
    """Each finished turn records its billed input for the next decision."""
    client, conversations = _trigger_client(tmp_path, _UsageStubBackend(123), 10**9)
    cid = _seed_turns(client, tmp_path, "one", "two")
    assert conversations.load_last_prompt_tokens(cid) == 123


def test_missing_usage_leaves_last_tokens_unset(tmp_path: pathlib.Path) -> None:
    """Backends without usage keep the trigger falling back, not crashing."""
    client, conversations = _trigger_client(
        tmp_path, stub_backend.StubBackend(), 10**9
    )
    cid = _seed_turns(client, tmp_path, "one")
    assert conversations.load_last_prompt_tokens(cid) is None


def test_remembered_usage_is_total_input(tmp_path: pathlib.Path) -> None:
    """Fresh-only prompt_tokens must not hide cache-read/write volume."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    # Shape of a cache-hot Anthropic turn: 2 fresh + 671k read + 339k write.
    chat_turn._remember_last_usage(
        conversations,
        "cid",
        log_events.TokenUsage(
            prompt_tokens=2,
            completion_tokens=2875,
            total_tokens=2883,
            cached_prompt_tokens=671736,
            cache_creation_input_tokens=339307,
        ),
    )
    assert conversations.load_last_prompt_tokens("cid") == 671736 + 339307
    # OpenAI-style inclusive counts: cached is a subset, never double-count.
    chat_turn._remember_last_usage(
        conversations,
        "cid2",
        log_events.TokenUsage(prompt_tokens=617, cached_prompt_tokens=512),
    )
    assert conversations.load_last_prompt_tokens("cid2") == 617


def test_remembered_usage_prefers_total_input(tmp_path: pathlib.Path) -> None:
    """Backends reporting total input bypass the max() fallback."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    chat_turn._remember_last_usage(
        conversations,
        "cid",
        log_events.TokenUsage(
            prompt_tokens=4,
            cached_prompt_tokens=0,
            cache_creation_input_tokens=342226,
            total_input_tokens=342230,
        ),
    )
    assert conversations.load_last_prompt_tokens("cid") == 342230


def test_cache_hot_turn_trips_trigger_next_turn(tmp_path: pathlib.Path) -> None:
    """A 1M-context turn compacts the turn after, not never."""
    backend = _UsageStubBackend(
        prompt_tokens=5, cached_tokens=900, cache_creation_tokens=100
    )
    client, conversations = _trigger_client(tmp_path, backend, 1000)
    cid = _seed_turns(client, tmp_path, "one", "two", "three")
    assert conversations.load_last_prompt_tokens(cid) == 1000
    body: dict[str, Any] = client.post(
        "/chat",
        json={"prompt": "four", "conversation_id": cid, "cwd": str(tmp_path)},
    ).json()
    assert body["compaction"] == {"reason": "auto"}
