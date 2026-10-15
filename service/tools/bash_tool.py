"""Bash tool: run a shell command, return truncated output."""

import pathlib
import subprocess

from service.backends import base
from service.tools import registry

from typing import Any

# Machine name referenced by model tool_use blocks.
BASH_TOOL_NAME = "bash"
# Default cap on command runtime; override with --bash-timeout.
DEFAULT_BASH_TIMEOUT_S = 120
# Default cap on captured output kept in history and logs.
DEFAULT_BASH_MAX_BYTES = 32768


def _tail(full: str, max_bytes: int) -> str:
    """Truncates output to its tail, marking the cut when one happened."""
    if (
        len(full) <= max_bytes
        and len(full.encode("utf-8", errors="ignore")) <= max_bytes
    ):
        return full
    return f"<truncated, showing last {max_bytes} of ~{len(full)} chars>\n{full[-max_bytes:]}"


def _run_command(
    command: str, workdir: pathlib.Path, timeout_s: int, max_bytes: int
) -> registry.ToolResult:
    """Runs one shell command in workdir, merging stdout/stderr truncated."""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return registry.ToolResult(
            output=f"<timed out after {timeout_s}s: {command}>",
            failure_reason=f"timed out after {timeout_s}s",
        )
    full = completed.stdout + completed.stderr
    if completed.returncode != 0:
        body = _tail(full, max_bytes) if full else "<no output>"
        return registry.ToolResult(
            output=f"{body}\n<failed with exit {completed.returncode}>",
            failure_reason=f"exit code {completed.returncode}",
        )
    if not full:
        return registry.ToolResult(output="<ok, no output>")
    return registry.ToolResult(output=_tail(full, max_bytes))


def build_bash_tool(
    timeout_s: int = DEFAULT_BASH_TIMEOUT_S,
    max_bytes: int = DEFAULT_BASH_MAX_BYTES,
) -> registry.Tool:
    """Builds the registry entry for the bash tool."""

    def _handle(
        tool_input: dict[str, Any], context: base.ToolContext
    ) -> registry.ToolResult:
        if context.workdir is None:
            return registry.ToolResult(
                output="<error: bash offered without a workdir; no command run>",
                failure_reason="no workdir",
            )
        return _run_command(
            str(tool_input.get("command", "")), context.workdir, timeout_s, max_bytes
        )

    return registry.Tool(
        name=BASH_TOOL_NAME,
        description="Run a bash command and return its combined output.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                }
            },
            "required": ["command"],
        },
        handler=_handle,
    )
