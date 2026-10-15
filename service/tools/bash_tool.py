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


def _run_command(
    command: str, workdir: pathlib.Path, timeout_s: int, max_bytes: int
) -> str:
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
        return f"<timed out after {timeout_s}s: {command}>"
    full = completed.stdout + completed.stderr
    if len(full.encode("utf-8", errors="ignore")) > max_bytes or len(full) > max_bytes:
        tail = full[-max_bytes:]
        return (
            f"<truncated, showing last {max_bytes} of ~{len(full)} chars>\n{tail}"
            or f"<exit {completed.returncode}, no output>"
        )
    return full or f"<exit {completed.returncode}, no output>"


def build_bash_tool(
    timeout_s: int = DEFAULT_BASH_TIMEOUT_S,
    max_bytes: int = DEFAULT_BASH_MAX_BYTES,
) -> registry.Tool:
    """Builds the registry entry for the bash tool."""

    def _handle(tool_input: dict[str, Any], context: base.ToolContext) -> str:
        if context.workdir is None:
            return "<error: bash offered without a workdir; no command run>"
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
