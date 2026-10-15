"""Tests for the read tool: slices, numbers, and failure reasons."""

import pathlib

from service.backends import base
from service.tools import read_tool


def _context(tmp_path: pathlib.Path) -> base.ToolContext:
    return base.ToolContext(workdir=tmp_path)


def _write(tmp_path: pathlib.Path, name: str, lines: int) -> None:
    (tmp_path / name).write_text("".join(f"line {n}\n" for n in range(1, lines + 1)))


def test_numbered_slice_with_header(tmp_path: pathlib.Path) -> None:
    """Reads show line numbers plus a shown/total header."""
    _write(tmp_path, "a.txt", 10)
    tool = read_tool.build_read_tool()
    result = tool.handler(
        {"path": "a.txt", "offset": 3, "limit": 2}, _context(tmp_path)
    )
    assert result.failure_reason is None
    assert result.output.splitlines()[0] == "--- a.txt (lines 3-4 of 10) ---"
    assert "3: line 3" in result.output
    assert "4: line 4" in result.output
    assert "line 5" not in result.output


def test_defaults_to_first_page(tmp_path: pathlib.Path) -> None:
    """Omitting offset/limit starts at line 1 with the default page."""
    _write(tmp_path, "a.txt", 5)
    result = read_tool.build_read_tool().handler({"path": "a.txt"}, _context(tmp_path))
    assert result.failure_reason is None
    assert result.output.startswith("--- a.txt (lines 1-5 of 5) ---")


def test_missing_file_is_an_error(tmp_path: pathlib.Path) -> None:
    """Unknown paths fail with a reason instead of empty output."""
    result = read_tool.build_read_tool().handler(
        {"path": "nope.txt"}, _context(tmp_path)
    )
    assert result.failure_reason == "no such file"


def test_escape_is_rejected(tmp_path: pathlib.Path) -> None:
    """Paths leaving the workdir are refused, including via subdirs."""
    (tmp_path / "sub").mkdir()
    result = read_tool.build_read_tool().handler(
        {"path": "sub/../../outside.txt"}, _context(tmp_path)
    )
    assert result.failure_reason == "outside workdir"


def test_absolute_escape_is_rejected(tmp_path: pathlib.Path) -> None:
    """Absolute paths outside the workdir are refused."""
    result = read_tool.build_read_tool().handler(
        {"path": "/etc/hostname"}, _context(tmp_path)
    )
    assert result.failure_reason == "outside workdir"


def test_binary_is_rejected(tmp_path: pathlib.Path) -> None:
    """Non-text files fail instead of returning garbage."""
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02binary")
    result = read_tool.build_read_tool().handler({"path": "b.bin"}, _context(tmp_path))
    assert result.failure_reason == "binary file"


def test_directory_is_rejected(tmp_path: pathlib.Path) -> None:
    """Directories fail instead of listing or dumping."""
    (tmp_path / "sub").mkdir()
    result = read_tool.build_read_tool().handler({"path": "sub"}, _context(tmp_path))
    assert result.failure_reason == "not a file"


def test_truncation_is_marked(tmp_path: pathlib.Path) -> None:
    """Oversized slices are cut with a marker, not silently."""
    _write(tmp_path, "big.txt", 50)
    tool = read_tool.build_read_tool(max_bytes=60)
    result = tool.handler({"path": "big.txt"}, _context(tmp_path))
    assert result.failure_reason is None
    assert "<truncated," in result.output
