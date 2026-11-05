"""Tests for peer-facing tool-output notices shared by all backends."""

import pytest

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
