"""Append-only per-conversation JSONL event log."""

import pathlib
import sys

from service import log_events


class EventLog:
    """Records prompts, replies, and tool traffic for later inspection."""

    def __init__(self, directory: pathlib.Path, verbose: bool = False) -> None:
        self._directory = directory
        self._verbose = verbose
        self._directory.mkdir(parents=True, exist_ok=True)

    def record(self, conversation_id: str, event: log_events.LogEvent) -> None:
        """Appends one typed event to the conversation's log file."""
        line = event.model_dump_json()
        path = self._directory / f"{conversation_id}.jsonl"
        with open(path, "a") as handle:
            handle.write(line + "\n")
        if self._verbose:
            print(line, file=sys.stderr, flush=True)
