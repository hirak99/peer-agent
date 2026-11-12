"""Tests for peer-facing tool-output notices shared by all backends."""

import pytest

from service import log_events
from service.backends import base
from service.backends import tool_loop
from service.tools import registry


@pytest.mark.parametrize("name", ["read", "bash", "web_fetch"])
@pytest.mark.parametrize("size", [0, 10239, 10240, 10241, 32768])
def test_large_output_notice_is_peer_facing(name: str, size: int) -> None:
    output = "x" * size
    tools = registry.ToolRegistry()
    tools.register(
        registry.Tool(
            name=name,
            description="fake",
            input_schema={},
            handler=lambda _input, _context: registry.ToolResult(output=output),
        )
    )
    events: list[base.ToolEvent] = []
    live_events: list[base.ToolEvent] = []
    executed = tool_loop.execute_calls(
        tools,
        [("id", name, {})],
        events,
        base.ToolContext(None),
        [],
        live_events.append,
    )

    expected = output
    if size > 10240:
        expected += f"\n[note: this result was ~{size / 1024:.0f} KB; prefer narrower reads/searches]"
    assert executed == [("id", name, expected, False)]
    assert events[0].output == output
    assert live_events[0].output == output


def test_large_truncated_unicode_output_gets_one_notice() -> None:
    output = "<truncated, showing first 32768 bytes>\n" + "界" * 4000
    tools = registry.ToolRegistry()
    tools.register(
        registry.Tool(
            name="read",
            description="fake",
            input_schema={},
            handler=lambda _input, _context: registry.ToolResult(output=output),
        )
    )
    executed = tool_loop.execute_calls(
        tools, [("id", "read", {})], [], base.ToolContext(None), []
    )

    assert len(output) < 10240 < len(output.encode("utf-8"))
    assert executed[0][2] == (
        output + "\n[note: this result was ~12 KB; prefer narrower reads/searches]"
    )
    assert executed[0][2].count("[note:") == 1
    assert executed[0][2].count("<truncated,") == 1


def _usage(
    prompt: int | None,
    cached: int | None = None,
    created: int | None = None,
    total_input: int | None = None,
) -> log_events.TokenUsage:
    return log_events.TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=1,
        total_tokens=(prompt or 0) + 1,
        cached_prompt_tokens=cached,
        cache_creation_input_tokens=created,
        total_input_tokens=total_input,
    )


def test_accumulate_keeps_latest_total_input() -> None:
    """Input size is per-call, so the latest call wins like prompt_tokens."""
    first = _usage(600, cached=0, created=600, total_input=1200)
    second = _usage(2, cached=600, created=10, total_input=612)
    merged = tool_loop.accumulate_usage(first, second)
    assert merged is not None
    assert merged.prompt_tokens == 2
    assert merged.total_input_tokens == 612
    # Cache counters still sum for cost accounting.
    assert merged.cached_prompt_tokens == 600
    assert merged.cache_creation_input_tokens == 610


def test_accumulate_tolerates_missing_total_input() -> None:
    """Hand-built usages without the field degrade to None, not a crash."""
    merged = tool_loop.accumulate_usage(_usage(10), _usage(20))
    assert merged is not None
    assert merged.total_input_tokens is None
    assert merged.prompt_tokens == 20
