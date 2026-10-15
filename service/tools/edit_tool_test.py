"""Tests for the edit tool: uniqueness guards and failure reasons."""

import pathlib

from service.backends import base
from service.tools import edit_tool


def _context(tmp_path: pathlib.Path) -> base.ToolContext:
    return base.ToolContext(workdir=tmp_path)


def test_unique_replace_edits_file(tmp_path: pathlib.Path) -> None:
    """A unique match is replaced and reported with its line number."""
    (tmp_path / "a.txt").write_text("hello world\nsecond line\n")
    result = edit_tool.build_edit_tool().handler(
        {"path": "a.txt", "old_string": "world", "new_string": "there"},
        _context(tmp_path),
    )
    assert result.failure_reason is None
    assert (tmp_path / "a.txt").read_text() == "hello there\nsecond line\n"
    assert "line 1" in result.output


def test_no_match_is_an_error(tmp_path: pathlib.Path) -> None:
    """Unknown old_string fails without touching the file."""
    (tmp_path / "a.txt").write_text("hello\n")
    result = edit_tool.build_edit_tool().handler(
        {"path": "a.txt", "old_string": "missing", "new_string": "x"},
        _context(tmp_path),
    )
    assert result.failure_reason == "no match"
    assert (tmp_path / "a.txt").read_text() == "hello\n"


def test_ambiguous_match_fails_without_replace_all(tmp_path: pathlib.Path) -> None:
    """Repeated text fails so the caller must disambiguate or opt into all."""
    (tmp_path / "a.txt").write_text("aa\nbb\naa\n")
    result = edit_tool.build_edit_tool().handler(
        {"path": "a.txt", "old_string": "aa", "new_string": "zz"},
        _context(tmp_path),
    )
    assert result.failure_reason == "ambiguous: 2 matches"
    assert (tmp_path / "a.txt").read_text() == "aa\nbb\naa\n"


def test_replace_all_replaces_every_match(tmp_path: pathlib.Path) -> None:
    """replace_all=true replaces all occurrences and reports the count."""
    (tmp_path / "a.txt").write_text("aa\nbb\naa\n")
    result = edit_tool.build_edit_tool().handler(
        {
            "path": "a.txt",
            "old_string": "aa",
            "new_string": "zz",
            "replace_all": True,
        },
        _context(tmp_path),
    )
    assert result.failure_reason is None
    assert (tmp_path / "a.txt").read_text() == "zz\nbb\nzz\n"
    assert "2 occurrences" in result.output


def test_escape_is_rejected(tmp_path: pathlib.Path) -> None:
    """Paths leaving the workdir are refused."""
    result = edit_tool.build_edit_tool().handler(
        {
            "path": "sub/../../outside.txt",
            "old_string": "a",
            "new_string": "b",
        },
        _context(tmp_path),
    )
    assert result.failure_reason == "outside workdir"


def test_missing_workdir_is_rejected(tmp_path: pathlib.Path) -> None:
    """Edit without a workdir fails without running anything."""
    (tmp_path / "a.txt").write_text("hello\n")
    result = edit_tool.build_edit_tool().handler(
        {"path": "a.txt", "old_string": "hello", "new_string": "hi"},
        base.ToolContext(workdir=None),
    )
    assert result.failure_reason == "no workdir"


def test_binary_is_rejected(tmp_path: pathlib.Path) -> None:
    """Non-text files fail instead of being rewritten."""
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02binary")
    result = edit_tool.build_edit_tool().handler(
        {"path": "b.bin", "old_string": "binary", "new_string": "x"},
        _context(tmp_path),
    )
    assert result.failure_reason == "binary file"
