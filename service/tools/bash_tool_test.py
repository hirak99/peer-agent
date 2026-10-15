"""Tests for the bash tool result shape and failure reasons."""

import pathlib

from service.backends import base
from service.tools import bash_tool


def _context() -> base.ToolContext:
    return base.ToolContext(workdir=pathlib.Path("/tmp"))


def test_success_has_no_failure_reason() -> None:
    """A clean command reports success with its output."""
    result = bash_tool.build_bash_tool().handler({"command": "echo hi"}, _context())
    assert result.failure_reason is None
    assert "hi" in result.output


def test_nonzero_exit_reports_failure_reason() -> None:
    """A failing command keeps its output and names the exit code."""
    result = bash_tool.build_bash_tool().handler({"command": "exit 3"}, _context())
    assert result.failure_reason == "exit code 3"
    assert "failed with exit 3" in result.output


def test_missing_workdir_reports_failure_reason() -> None:
    """Bash without a workdir fails without running anything."""
    result = bash_tool.build_bash_tool().handler(
        {"command": "echo hi"}, base.ToolContext(workdir=None)
    )
    assert result.failure_reason == "no workdir"


def test_timeout_reports_failure_reason() -> None:
    """An overrunning command fails with a timeout reason."""
    tool = bash_tool.build_bash_tool(timeout_s=1)
    result = tool.handler({"command": "sleep 5"}, _context())
    assert result.failure_reason is not None
    assert "timed out" in result.failure_reason


def test_large_output_is_marked_truncated() -> None:
    """Truncated output says so instead of silently showing a tail."""
    tool = bash_tool.build_bash_tool(max_bytes=10)
    result = tool.handler({"command": "echo hello-world-long-output"}, _context())
    assert result.failure_reason is None
    assert result.output.startswith("<truncated,")
