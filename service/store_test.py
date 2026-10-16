"""Tests for conversation snapshots and atomic history writes."""

import json
import pathlib

from service import store


def _messages(n: int) -> list[dict[str, str]]:
    """Builds n tiny user messages."""
    return [{"role": "user", "content": f"m{i}"} for i in range(n)]


def test_backup_returns_snapshot_of_stored_history(tmp_path: pathlib.Path) -> None:
    """backup_current copies the file verbatim and reports its path."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    conversations.save("cid", _messages(3))
    backup = conversations.backup_current("cid")
    assert backup is not None
    assert json.loads(pathlib.Path(backup).read_text()) == _messages(3)
    assert "cid" in backup


def test_backup_missing_conversation_returns_none(tmp_path: pathlib.Path) -> None:
    """No stored history means nothing to snapshot."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    assert conversations.backup_current("nope") is None


def test_backup_prunes_to_newest_five(tmp_path: pathlib.Path) -> None:
    """Only BACKUP_KEEP_COUNT snapshots survive repeated compactions."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    conversations.save("cid", _messages(1))
    for _ in range(store.BACKUP_KEEP_COUNT + 2):
        assert conversations.backup_current("cid") is not None
    kept = sorted((tmp_path / "conversations" / "backups" / "cid").glob("*.json"))
    assert len(kept) == store.BACKUP_KEEP_COUNT


def test_save_leaves_no_temp_file(tmp_path: pathlib.Path) -> None:
    """Atomic save replaces the file without litter."""
    conversations = store.ConversationStore(tmp_path / "conversations")
    conversations.save("cid", _messages(2))
    assert conversations.load("cid") == _messages(2)
    assert list((tmp_path / "conversations").glob("*.tmp-*")) == []
