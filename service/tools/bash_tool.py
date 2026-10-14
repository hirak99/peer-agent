"""Bash tool: run a shell command, return truncated output."""

import pathlib
import subprocess

from service.backends import base
from service.tools import registry

from typing import Any

# Machine name referenced by model tool_use blocks.
BASH_TOOL_NAME = "bash"
# Hard cap on command runtime; no permission system yet by design.
_BASH_TIMEOUT_S = 120
# Hard cap on captured output kept in history and logs.
_BASH_MAX_BYTES = 32768


def _run_command(command: str, workdir: pathlib.Path) -> str:
    """Runs one shell command in workdir, merging stdout/stderr truncated."""
    try:
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_BASH_TIMEOUT_S,
            cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return f"<timed out after {_BASH_TIMEOUT_S}s: {command}>"
    output = (completed.stdout + completed.stderr)[-_BASH_MAX_BYTES:]
    return output or f"<exit {completed.returncode}, no output>"


def build_bash_tool() -> registry.Tool:
    """Builds the registry entry for the bash tool."""

    def _handle(tool_input: dict[str, Any], context: base.ToolContext) -> str:
        assert context.workdir is not None, "bash offered without a workdir"
        return _run_command(str(tool_input.get("command", "")), context.workdir)

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
