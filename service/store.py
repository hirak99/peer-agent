"""File-backed store for conversation histories."""

import json
import pathlib
import uuid

from typing import Any


class ConversationStore:
    """Maps conversation ids to persisted message lists."""

    def __init__(self, directory: pathlib.Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path_for(self, conversation_id: str) -> pathlib.Path:
        return self._directory / f"{conversation_id}.json"

    def exists(self, conversation_id: str) -> bool:
        """Reports whether a conversation id has stored history."""
        return self._path_for(conversation_id).exists()

    def new_id(self) -> str:
        """Returns an unused conversation id."""
        return uuid.uuid4().hex

    def load(self, conversation_id: str) -> list[dict[str, Any]]:
        """Loads stored messages, oldest first; empty when unknown."""
        path = self._path_for(conversation_id)
        if not path.exists():
            return []
        raw: Any = json.loads(path.read_text())
        return raw if isinstance(raw, list) else []

    def save(self, conversation_id: str, messages: list[dict[str, Any]]) -> None:
        """Overwrites the stored history for a conversation."""
        self._path_for(conversation_id).write_text(json.dumps(messages))
