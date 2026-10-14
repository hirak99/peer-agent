"""Interface between the HTTP service and a model provider."""

import abc
import dataclasses
import pathlib
from collections.abc import Iterator

from service import log_events

from typing import Any


@dataclasses.dataclass
class ToolContext:
    """Per-turn execution context handed to every tool handler."""

    # Host working directory, or None when this turn has no filesystem.
    workdir: pathlib.Path | None


@dataclasses.dataclass
class ToolEvent:
    """One tool invocation observed while producing a reply."""

    # Tool name from the model tool_use block.
    name: str
    # Decoded JSON input passed to the handler.
    tool_input: dict[str, Any]
    # Truncated handler output fed back to the model.
    output: str


@dataclasses.dataclass
class Completion:
    """Final reply text plus observability side-channels."""

    # The peer agent's final reply text.
    text: str
    # Provider-reported token counts, or None when unreported.
    usage: log_events.TokenUsage | None
    # Tool invocations in order, for logging.
    tool_events: list[ToolEvent]


class Backend(abc.ABC):
    """Produces replies from stored message histories.

    Both entry points append their assistant turns onto `messages`, so the
    caller can persist the list verbatim for the next turn.
    """

    @abc.abstractmethod
    def describe(self) -> dict[str, str]:
        """Returns backend/model identity for logging."""

    @abc.abstractmethod
    def complete(
        self, messages: list[dict[str, Any]], context: ToolContext
    ) -> Completion:
        """Runs the model (including tool turns) to a final reply."""

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        tool_events: list[ToolEvent],
        context: ToolContext,
    ) -> Iterator[str]:
        """Yields reply text pieces; appends observed tool events."""
        completion = self.complete(messages, context)
        tool_events.extend(completion.tool_events)
        for index in range(0, len(completion.text), 400):
            yield completion.text[index : index + 400]
