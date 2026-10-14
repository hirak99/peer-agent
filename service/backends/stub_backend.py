"""Deterministic stub backend for tests and offline bring-up."""

from collections.abc import Iterator

from service.backends import base

from typing import Any


class StubBackend(base.Backend):
    """Echoes the last user message without calling any provider."""

    def complete(self, messages: list[dict[str, Any]]) -> base.Completion:
        """Returns a canned reply quoting the latest user turn."""
        last: Any = messages[-1]["content"] if messages else ""
        text = f"stub-response to: {last}" if isinstance(last, str) else "stub-response"
        messages.append({"role": "assistant", "content": text})
        return base.Completion(text=text, usage=None, tool_events=[])

    def stream_text(
        self, messages: list[dict[str, Any]], tool_events: list[base.ToolEvent]
    ) -> Iterator[str]:
        """Yields the canned reply word by word."""
        completion = self.complete(messages)
        for word in completion.text.split(" "):
            yield word + " "
