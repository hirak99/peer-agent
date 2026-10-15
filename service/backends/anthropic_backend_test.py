"""Tests for failure-aware tool handling in the Anthropic backend."""

import json
import pathlib
import time
import types
from collections import deque

from service.backends import anthropic_backend
from service.backends import base
from service.tools import registry

from typing import Any

# pyright: reportPrivateUsage=false


def _backend(handler: Any) -> tuple[anthropic_backend.AnthropicBackend, list[Any]]:
    """Builds a backend with one fake tool, bypassing API construction."""
    backend = anthropic_backend.AnthropicBackend.__new__(
        anthropic_backend.AnthropicBackend
    )
    seen_inputs: list[Any] = []

    def _handle(
        tool_input: dict[str, Any], context: base.ToolContext
    ) -> registry.ToolResult:
        seen_inputs.append(dict(tool_input))
        return handler(tool_input)

    tools = registry.ToolRegistry()
    tools.register(
        registry.Tool(name="t", description="fake", input_schema={}, handler=_handle)
    )
    backend._tools = tools
    return backend, seen_inputs


def _call(name: str, tool_input: dict[str, Any], call_id: str) -> Any:
    return types.SimpleNamespace(name=name, input=tool_input, id=call_id)


def test_hold_after_five_identical_failures() -> None:
    """The sixth identical failure is held; its retry runs again."""
    backend, seen_inputs = _backend(
        lambda _: registry.ToolResult(output="boom", failure_reason="exit code 1")
    )
    context = base.ToolContext(workdir=None)
    seen: list[tuple[str, bool]] = []
    results = [
        backend._run_calls([_call("t", {"c": "x"}, f"id{n}")], [], context, seen)[0]
        for n in range(7)
    ]
    assert len(seen_inputs) == 6
    assert all(r["is_error"] for r in results)
    assert "Held" in results[5]["content"]
    assert "Held" not in results[6]["content"]


def test_successes_are_never_held() -> None:
    """Identical succeeding calls pass through no matter the volume."""
    backend, seen_inputs = _backend(
        lambda _: registry.ToolResult(output="ok", failure_reason=None)
    )
    context = base.ToolContext(workdir=None)
    seen: list[tuple[str, bool]] = []
    for n in range(8):
        results = backend._run_calls(
            [_call("t", {"c": "x"}, f"id{n}")], [], context, seen
        )
        assert results[0]["is_error"] is False
    assert len(seen_inputs) == 8


def test_different_call_resets_the_streak() -> None:
    """An interleaved different call breaks the identical-failure run."""
    backend, seen_inputs = _backend(
        lambda _: registry.ToolResult(output="boom", failure_reason="exit code 1")
    )
    context = base.ToolContext(workdir=None)
    seen: list[tuple[str, bool]] = []
    for n in range(3):
        backend._run_calls([_call("t", {"c": "x"}, f"a{n}")], [], context, seen)
    backend._run_calls([_call("t", {"c": "other"}, "b")], [], context, seen)
    for n in range(3):
        results = backend._run_calls(
            [_call("t", {"c": "x"}, f"c{n}")], [], context, seen
        )
        assert "Held" not in results[0]["content"]
    assert len(seen_inputs) == 7


def test_unknown_tool_is_an_error() -> None:
    """Calls to unregistered tools come back as errors, not text."""
    backend, _ = _backend(lambda _: registry.ToolResult(output="ok"))
    context = base.ToolContext(workdir=None)
    results = backend._run_calls([_call("missing", {}, "id0")], [], context, [])
    assert results[0]["is_error"] is True


def _wired_backend() -> anthropic_backend.AnthropicBackend:
    """Builds a backend with identity fields set, bypassing API construction."""
    backend, _ = _backend(lambda _: registry.ToolResult(output="ok"))
    backend._model = "test-model"
    backend._system_prompt = "test system"
    backend._max_tokens = 64
    return backend


def test_system_param_carries_cache_breakpoint() -> None:
    """The system prompt is sent as a cached text block."""
    assert _wired_backend()._system_param() == [
        {
            "type": "text",
            "text": "test system",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_extra_params_breakpoint_marks_last_tool_only(
    tmp_path: pathlib.Path,
) -> None:
    """The last tool definition is cached; the registry copy is untouched."""
    backend = _wired_backend()
    params = backend._extra_params(base.ToolContext(workdir=tmp_path))
    assert params["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in backend._tools.anthropic_schemas()[-1]


def test_message_params_marks_tail_without_mutating_history() -> None:
    """Breakpoints go on per-request copies; stored history stays clean."""
    backend = _wired_backend()
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "second"}],
        },
        {"role": "user", "content": "third"},
    ]
    params = backend._message_params(history)
    assert history[-1] == {"role": "user", "content": "third"}
    assert params[-1]["content"] == [
        {
            "type": "text",
            "text": "third",
            "cache_control": {"type": "ephemeral"},
        }
    ]

    tool_history: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "input": {}}],
        },
    ]
    tool_params = backend._message_params(tool_history)
    assert tool_params[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tool_history[0]["content"][-1]
    assert backend._message_params([]) == []


def test_complete_leaves_stored_history_cache_free() -> None:
    """The full request is cache-marked but persistence sees plain history."""
    backend = _wired_backend()
    _seed_times(backend, [30, 20, 10])
    calls: list[dict[str, Any]] = []

    def _create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="done")],
            usage=types.SimpleNamespace(input_tokens=50, output_tokens=5),
        )

    backend._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=_create)
    )
    history: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    completion = backend.complete(history, base.ToolContext(workdir=None), None)
    assert completion.text == "done"
    assert "cache_control" not in json.dumps(history)
    sent = calls[0]
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def _seed_times(
    backend: anthropic_backend.AnthropicBackend, offsets: list[float]
) -> None:
    """Seeds recency history at offsets (seconds ago)."""
    now = time.monotonic()
    backend._request_times = deque(
        [now - offset for offset in offsets],
        maxlen=anthropic_backend._CACHE_RECENCY_COUNT,
    )


def test_cache_gating_starts_cold() -> None:
    """Fresh backends skip breakpoints until brisk cadence is proven."""
    backend = _wired_backend()
    assert backend._should_cache() is False
    assert backend._should_cache() is False
    assert backend._should_cache() is False
    assert backend._should_cache() is True


def test_cache_gating_rejects_stale_cadence() -> None:
    """One old request in the window disables breakpoints for this call."""
    backend = _wired_backend()
    _seed_times(backend, [400, 10, 0])
    assert backend._should_cache() is False


def test_cache_gating_reenables_after_brisk_resume() -> None:
    """Post-idle calls pay base price until three brisk requests re-arm."""
    backend = _wired_backend()
    _seed_times(backend, [1000, 900, 800])
    assert backend._should_cache() is False
    assert backend._should_cache() is False
    assert backend._should_cache() is False
    assert backend._should_cache() is True


def test_helpers_omit_breakpoints_when_gated_off() -> None:
    """use_cache=False leaves system and history unmarked."""
    backend = _wired_backend()
    assert backend._system_param(use_cache=False) == [
        {"type": "text", "text": "test system"}
    ]
    params = backend._message_params([{"role": "user", "content": "hi"}], False)
    assert params == [{"role": "user", "content": "hi"}]


def test_complete_skips_breakpoints_on_cold_start() -> None:
    """The first calls of a session carry no cache markers."""
    backend = _wired_backend()
    calls: list[dict[str, Any]] = []

    def _create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="done")],
            usage=types.SimpleNamespace(input_tokens=50, output_tokens=5),
        )

    backend._client = types.SimpleNamespace(
        messages=types.SimpleNamespace(create=_create)
    )
    history: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    backend.complete(history, base.ToolContext(workdir=None), None)
    assert "cache_control" not in json.dumps(calls[0])
