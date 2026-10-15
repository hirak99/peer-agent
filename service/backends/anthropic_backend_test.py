"""Tests for failure-aware tool handling in the Anthropic backend."""

import types

from service.backends import anthropic_backend
from service.backends import base
from service.tools import registry

from typing import Any


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
