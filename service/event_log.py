"""Append-only per-conversation JSONL event log."""

import pathlib

from service import log_events


class EventLog:
    """Records prompts, replies, and tool traffic for later inspection."""

    def __init__(self, directory: pathlib.Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def record(self, conversation_id: str, event: log_events.LogEvent) -> None:
        """Appends one typed event to the conversation's log file."""
        path = self._directory / f"{conversation_id}.jsonl"
        with open(path, "a") as handle:
            handle.write(event.model_dump_json() + "\n")
