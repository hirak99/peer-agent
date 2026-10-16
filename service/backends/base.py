"""Interface between the HTTP service and a model provider."""

import abc
import dataclasses
import pathlib
from collections.abc import Callable, Iterator

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
    # Why the call failed; None means success.
    failure_reason: str | None = None


@dataclasses.dataclass
class CompactionRecord:
    """Describes one history compaction for logging."""

    # Messages before compaction.
    pre_messages: int
    # Messages after compaction.
    post_messages: int
    # Estimated input tokens before compaction.
    pre_tokens_est: int
    # Estimated input tokens after compaction.
    post_tokens_est: int
    # Why compaction ran.
    reason: str


@dataclasses.dataclass
class Completion:
    """Final reply text plus observability side-channels."""

    # The peer agent's final reply text.
    text: str
    # Provider-reported token counts, or None when unreported.
    usage: log_events.TokenUsage | None
    # Tool invocations in order, for logging.
    tool_events: list[ToolEvent]
    # Yield gate outcome: done, or paused for host to send "continue".
    status: str = "done"
    # Human-readable resume instruction when status != done.
    hint: str | None = None
    # Tool turns executed to produce this reply.
    tool_turns: int = 0
    # Compaction performed during error recovery, if any.
    compaction: CompactionRecord | None = None
    # Accumulated model thinking text across all turns (logs only).
    reasoning: str = ""


class Backend(abc.ABC):
    """Produces replies from stored message histories.

    Both entry points append their assistant turns onto `messages`, so the
    caller can persist the list verbatim for the next turn.
    """

    @abc.abstractmethod
    def describe(self) -> dict[str, str]:
        """Returns backend/model identity for logging."""

    def get_context_window(self) -> int | None:
        """Returns the model input budget, or None when unknown."""
        return None

    def get_max_output_tokens(self) -> int | None:
        """Returns the per-turn output reserve, or None when unknown."""
        return None

    def summarize(self, messages: list[dict[str, Any]]) -> str:
        """Distills messages into a handoff summary for compaction."""
        raise NotImplementedError("summarize is not supported by this backend.")

    @abc.abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        context: ToolContext,
        host_max_tool_turns: int | None = None,
        on_tool_event: Callable[[ToolEvent], None] | None = None,
    ) -> Completion:
        """Runs the model (including tool turns) to a final reply."""

    def stream_text(
        self,
        messages: list[dict[str, Any]],
        tool_events: list[ToolEvent],
        context: ToolContext,
        host_max_tool_turns: int | None = None,
        meta: dict[str, Any] | None = None,
        on_tool_event: Callable[[ToolEvent], None] | None = None,
    ) -> Iterator[str]:
        """Yields reply text pieces; appends observed tool events."""
        completion = self.complete(messages, context, host_max_tool_turns, on_tool_event)
        tool_events.extend(completion.tool_events)
        if meta is not None:
            meta["usage"] = completion.usage
            meta["status"] = completion.status
            meta["hint"] = completion.hint
            meta["tool_turns"] = completion.tool_turns
            meta["reasoning"] = completion.reasoning
        for index in range(0, len(completion.text), 400):
            yield completion.text[index : index + 400]
