"""Tests for the OpenAI-compatible backend (offline, faked transport)."""

import types

from service.backends import base
from service.backends import openai_compat_backend
from service.tools import registry

from typing import Any

# pyright: reportPrivateUsage=false


def _backend() -> openai_compat_backend.OpenAICompatBackend:
    """Builds a backend with one fake tool, bypassing endpoint validation."""
    backend = openai_compat_backend.OpenAICompatBackend.__new__(
        openai_compat_backend.OpenAICompatBackend
    )
    backend._model = "fake"
    backend._system_prompt = "sys"
    backend._base_url = "http://localhost:1/v1"
    backend._max_tokens = 32
    backend._context_window = None
    backend._compact_model = "fake"
    backend._compact_summary_max_tokens = 32
    backend._compact_keep = 20
    tools = registry.ToolRegistry()
    tools.register(
        registry.Tool(
            name="bash",
            description="Run a shell command",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
            handler=lambda tool_input, context: registry.ToolResult(
                output=f"ran {tool_input.get('command', '')}"
            ),
        )
    )
    backend._tools = tools
    return backend


def _response(
    content: Any = "", tool_calls: Any | None = None, usage: Any | None = None
) -> Any:
    message = types.SimpleNamespace(
        content=content, tool_calls=tool_calls, reasoning_content="thinking..."
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)


def _tool_call(call_id: str, name: str, arguments: str) -> Any:
    return types.SimpleNamespace(
        id=call_id,
        type="function",
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


def test_history_translation_round_trip() -> None:
    """Stored Anthropic-flavored blocks become OpenAI messages and back."""
    backend = _backend()
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "c1", "name": "bash", "input": {"command": "x"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "c1",
                    "content": "out",
                    "is_error": False,
                }
            ],
        },
    ]
    params = backend._chat_messages(messages)
    assert params[0] == {"role": "system", "content": "sys"}
    assert params[1] == {"role": "user", "content": "hi"}
    assert params[2]["role"] == "assistant"
    assert params[2]["tool_calls"][0]["id"] == "c1"
    assert params[2]["tool_calls"][0]["function"]["name"] == "bash"
    assert params[3] == {"role": "tool", "tool_call_id": "c1", "content": "out"}


def test_parse_response_drops_reasoning() -> None:
    """Reasoning content never becomes reply text or tool input."""
    text, calls, reasoning = openai_compat_backend.OpenAICompatBackend._parse_response(
        _response(content="four", tool_calls=[_tool_call("c1", "bash", '{"command": "x"}')])
    )
    assert text == "four"
    assert calls == [("c1", "bash", {"command": "x"})]
    assert reasoning == "thinking..."


def test_parse_response_bad_arguments() -> None:
    """Unparseable tool arguments degrade to an empty input dict."""
    _, calls, _ = openai_compat_backend.OpenAICompatBackend._parse_response(
        _response(content="", tool_calls=[_tool_call("c1", "bash", "not-json")])
    )
    assert calls == [("c1", "bash", {})]


def test_complete_captures_reasoning() -> None:
    """Reasoning text is accumulated but never leaks into reply text."""
    backend = _backend()
    first = _response(
        content="first text",
        tool_calls=[_tool_call("c1", "bash", '{"command": "echo hi"}')],
    )
    final = _response(content="done")
    calls = iter([first, final])
    backend._request_turn = lambda messages, context: next(calls)  # type: ignore[method-assign]
    messages: list[dict[str, Any]] = [{"role": "user", "content": "run it"}]
    completion = backend.complete(messages, base.ToolContext(workdir=None))
    assert completion.text == "first text\n\ndone"
    assert completion.reasoning == "thinking...\n\nthinking..."
    assert "thinking..." not in completion.text


def test_complete_runs_tool_loop() -> None:
    """A tool_calls finish executes the tool and returns final text."""
    backend = _backend()
    turn = _response(
        content="",
        tool_calls=[_tool_call("c1", "bash", '{"command": "echo hi"}')],
        usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    final = _response(content="ran echo hi")
    calls = iter([turn, final])
    backend._request_turn = lambda messages, context: next(calls)  # type: ignore[method-assign]
    messages: list[dict[str, Any]] = [{"role": "user", "content": "run it"}]
    completion = backend.complete(messages, base.ToolContext(workdir=None))
    assert completion.text == "ran echo hi"
    assert completion.tool_turns == 1
    assert len(completion.tool_events) == 1
    assert completion.tool_events[0].name == "bash"
    assert completion.usage is not None
    assert completion.usage.prompt_tokens == 10


def test_tool_schemas_use_function_shape() -> None:
    """Registry renders OpenAI function tools for the wire."""
    tools = registry.ToolRegistry()
    tools.register(
        registry.Tool(
            name="bash", description="d", input_schema={"type": "object"}, handler=lambda i, c: registry.ToolResult(output="x")
        )
    )
    schemas = tools.openai_schemas()
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "bash"


def test_overflow_markers() -> None:
    """Context-window rejections are detected; other 400s are not."""

    class _Err(Exception):
        def __init__(self, text: str) -> None:
            super().__init__(text)
            self.status_code = 400
            self.response = None

    assert (
        openai_compat_backend.OpenAICompatBackend.is_context_overflow(
            _Err("requested tokens exceed context window")
        )
        is True
    )
    assert (
        openai_compat_backend.OpenAICompatBackend.is_context_overflow(
            _Err("invalid request: bad json")
        )
        is False
    )
