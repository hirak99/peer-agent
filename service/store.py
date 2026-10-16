"""File-backed store for conversation histories."""

import datetime
import json
import os
import pathlib
import uuid

from typing import Any

# Pre-compaction snapshots retained per conversation; older ones are pruned.
BACKUP_KEEP_COUNT = 5


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
        """Overwrites the stored history for a conversation, atomically."""
        path = self._path_for(conversation_id)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(messages))
        os.replace(tmp, path)

    def _backup_dir_for(self, conversation_id: str) -> pathlib.Path:
        return self._directory / "backups" / conversation_id

    def backup_current(self, conversation_id: str) -> str | None:
        """Snapshots the stored history before a compaction overwrites it.

        Returns the snapshot path, or None when nothing is stored yet.
        Only the newest BACKUP_KEEP_COUNT snapshots are retained.
        """
        path = self._path_for(conversation_id)
        if not path.exists():
            return None
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S_%fZ")
        backup_dir = self._backup_dir_for(conversation_id)
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"{stamp}.json"
        backup_path.write_bytes(path.read_bytes())
        for stale in sorted(backup_dir.glob("*.json"))[:-BACKUP_KEEP_COUNT]:
            stale.unlink()
        return str(backup_path)

    def load_system_prompt(self, conversation_id: str) -> str | None:
        """Loads the tracked system prompt, or None for new conversations."""
        path = self._directory / f"{conversation_id}.meta.json"
        if not path.exists():
            return None
        raw: Any = json.loads(path.read_text())
        if isinstance(raw, dict) and isinstance(raw.get("system_prompt"), str):
            return raw["system_prompt"]
        return None

    def save_system_prompt(self, conversation_id: str, system_prompt: str) -> None:
        """Records the active system prompt for change detection."""
        path = self._directory / f"{conversation_id}.meta.json"
        path.write_text(json.dumps({"system_prompt": system_prompt}))
