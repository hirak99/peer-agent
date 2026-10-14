"""Append-only per-conversation JSONL event log."""

import json
import pathlib
import time

from typing import Any


class EventLog:
    """Records prompts, replies, and tool traffic for later inspection."""

    def __init__(self, directory: pathlib.Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def record(self, conversation_id: str, event: dict[str, Any]) -> None:
        """Appends one timestamped event to the conversation's log file."""
        line = {"ts": time.time(), **event}
        path = self._directory / f"{conversation_id}.jsonl"
        with open(path, "a") as handle:
            handle.write(json.dumps(line) + "\n")
