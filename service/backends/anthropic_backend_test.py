"""Tests for failure-aware tool handling in the Anthropic backend."""

import json
import pathlib
import time
import types

from service.backends import anthropic_backend
from service.backends import base
from service.tools import registry

from typing import Any, cast

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
    marked = cast(Any, tool_params[0]["content"])[-1]
    assert marked["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in tool_history[0]["content"][-1]
    assert backend._message_params([]) == []


def test_complete_leaves_stored_history_cache_free() -> None:
    """The full request is cache-marked but persistence sees plain history."""
    backend = _wired_backend()
    calls: list[dict[str, Any]] = []

    def _create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="done")],
            usage=types.SimpleNamespace(input_tokens=50, output_tokens=5),
        )

    backend._client = cast(
        Any, types.SimpleNamespace(messages=types.SimpleNamespace(create=_create))
    )
    history: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    completion = backend.complete(history, base.ToolContext(workdir=None), None)
    assert completion.text == "done"
    assert "cache_control" not in json.dumps(history)
    sent = calls[0]
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}


def _suspend_gate(backend: anthropic_backend.AnthropicBackend) -> None:
    """Drives the backend's gate into the suspended state deterministically."""
    backend._should_cache()
    gate = backend._cache_gate
    start = time.monotonic() - 2000.0
    for offset in (0.0, 400.0, 800.0, 1200.0):
        gate.should_cache(now=start + offset)


def test_backend_delegates_caching_to_gate() -> None:
    """_should_cache passes through the shared gate decision."""
    backend = _wired_backend()
    assert backend._should_cache() is True
    _suspend_gate(backend)
    assert backend._should_cache() is False


def test_helpers_omit_breakpoints_when_gated_off() -> None:
    """use_cache=False leaves system and history unmarked."""
    backend = _wired_backend()
    assert backend._system_param(use_cache=False) == [
        {"type": "text", "text": "test system"}
    ]
    params = backend._message_params([{"role": "user", "content": "hi"}], False)
    assert params == [{"role": "user", "content": "hi"}]


def test_complete_skips_breakpoints_when_sustained_idle() -> None:
    """Only sustained idleness strips markers; cold starts still attach."""
    backend = _wired_backend()
    calls: list[dict[str, Any]] = []

    def _create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text="done")],
            usage=types.SimpleNamespace(input_tokens=50, output_tokens=5),
        )

    backend._client = cast(
        Any, types.SimpleNamespace(messages=types.SimpleNamespace(create=_create))
    )
    history: list[dict[str, Any]] = [{"role": "user", "content": "hi"}]
    backend.complete(history, base.ToolContext(workdir=None), None)
    assert "cache_control" in json.dumps(calls[0])
    _suspend_gate(backend)
    backend.complete(history, base.ToolContext(workdir=None), None)
    assert "cache_control" not in json.dumps(calls[1])
