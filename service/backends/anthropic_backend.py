"""Anthropic backend with a server-side tool loop."""

import os
from collections.abc import Iterator

import anthropic

from service.backends import base
from service.tools import registry

from typing import Any, cast

# Environment variable holding the Anthropic API key (never logged).
_ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
# Safety bound on consecutive tool turns per reply.
_MAX_TOOL_TURNS = 10
# Per-turn output budget.
_MAX_TOKENS = 4096


class AnthropicBackend(base.Backend):
    """Runs replies via the Anthropic messages API, executing tools inline."""

    def __init__(
        self, model: str, system_prompt: str, tools: registry.ToolRegistry
    ) -> None:
        api_key = os.environ.get(_ANTHROPIC_KEY_ENV)
        if not api_key:
            raise RuntimeError(
                f"Set {_ANTHROPIC_KEY_ENV} to use the Anthropic backend."
            )
        self._model = model
        self._system_prompt = system_prompt
        self._tools = tools
        self._client = anthropic.Anthropic(api_key=api_key)

    def _message_params(
        self, messages: list[dict[str, Any]]
    ) -> list[anthropic.types.MessageParam]:
        return cast(list[anthropic.types.MessageParam], messages)

    def _extra_params(self) -> dict[str, Any]:
        schemas = self._tools.anthropic_schemas()
        if not schemas:
            return {}
        return {"tools": cast(list[anthropic.types.ToolUnionParam], schemas)}

    def _run_calls(
        self, calls: list[Any], tool_events: list[base.ToolEvent]
    ) -> list[dict[str, Any]]:
        """Executes tool_use blocks into Anthropic tool_result blocks."""
        results: list[dict[str, Any]] = []
        for call in calls:
            tool = self._tools.get(call.name)
            tool_input = dict(call.input) if isinstance(call.input, dict) else {}
            if tool is None:
                output = f"<unknown tool: {call.name}>"
            else:
                output = tool.handler(tool_input)
            tool_events.append(
                base.ToolEvent(name=call.name, tool_input=tool_input, output=output)
            )
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                }
            )
        return results

    def complete(self, messages: list[dict[str, Any]]) -> base.Completion:
        """Runs model turns until final text, executing tools between turns."""
        tool_events: list[base.ToolEvent] = []
        usage: dict[str, Any] | None = None
        for _ in range(_MAX_TOOL_TURNS + 1):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=self._system_prompt,
                messages=self._message_params(messages),
                **self._extra_params(),
            )
            if response.usage is not None:
                usage = response.usage.model_dump()
            calls = [block for block in response.content if block.type == "tool_use"]
            if not calls:
                text = "".join(
                    block.text for block in response.content if block.type == "text"
                )
                messages.append({"role": "assistant", "content": text})
                return base.Completion(text=text, usage=usage, tool_events=tool_events)
            messages.append(
                {
                    "role": "assistant",
                    "content": [block.model_dump() for block in response.content],
                }
            )
            messages.append(
                {"role": "user", "content": self._run_calls(calls, tool_events)}
            )
        raise RuntimeError(f"Exceeded {_MAX_TOOL_TURNS} tool turns.")

    def stream_text(
        self, messages: list[dict[str, Any]], tool_events: list[base.ToolEvent]
    ) -> Iterator[str]:
        """Streams reply tokens live, running tool turns between streams."""
        for _ in range(_MAX_TOOL_TURNS + 1):
            with self._client.messages.stream(
                model=self._model,
                max_tokens=_MAX_TOKENS,
                system=self._system_prompt,
                messages=self._message_params(messages),
                **self._extra_params(),
            ) as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
            calls = [block for block in final.content if block.type == "tool_use"]
            if not calls:
                text = "".join(
                    block.text for block in final.content if block.type == "text"
                )
                messages.append({"role": "assistant", "content": text})
                return
            messages.append(
                {
                    "role": "assistant",
                    "content": [block.model_dump() for block in final.content],
                }
            )
            messages.append(
                {"role": "user", "content": self._run_calls(calls, tool_events)}
            )
        raise RuntimeError(f"Exceeded {_MAX_TOOL_TURNS} tool turns.")
