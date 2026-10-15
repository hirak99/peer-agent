"""Read tool: return a line-numbered slice of a file under the workdir."""

import pathlib

from service.backends import base
from service.tools import registry

from typing import Any

# Machine name referenced by model tool_use blocks.
READ_TOOL_NAME = "read"
# Default number of lines per read; override per call with limit.
DEFAULT_READ_MAX_LINES = 200
# Hard cap on rendered output kept in history and logs.
DEFAULT_READ_MAX_BYTES = 32768


def _resolve(target: str, workdir: pathlib.Path) -> pathlib.Path | None:
    """Resolves a tool-supplied path, confining it inside the workdir."""
    candidate = (workdir / target).resolve()
    try:
        candidate.relative_to(workdir.resolve())
    except ValueError:
        return None
    return candidate


def _read_lines(path: pathlib.Path) -> tuple[list[str] | None, str | None]:
    """Reads a file as UTF-8 lines, or returns a failure reason."""
    if not path.exists():
        return None, "no such file"
    if not path.is_file():
        return None, "not a file"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"unreadable: {exc.strerror or exc}"
    if b"\x00" in raw[:8192]:
        return None, "binary file"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "not UTF-8 text"
    return text.splitlines(), None


def build_read_tool(
    max_lines: int = DEFAULT_READ_MAX_LINES,
    max_bytes: int = DEFAULT_READ_MAX_BYTES,
) -> registry.Tool:
    """Builds the registry entry for the read tool."""

    def _handle(
        tool_input: dict[str, Any], context: base.ToolContext
    ) -> registry.ToolResult:
        if context.workdir is None:
            return registry.ToolResult(
                output="<error: read offered without a workdir; no file read>",
                failure_reason="no workdir",
            )
        target = str(tool_input.get("path", ""))
        if not target:
            return registry.ToolResult(
                output="<error: missing required 'path'>",
                failure_reason="missing path",
            )
        path = _resolve(target, context.workdir)
        if path is None:
            return registry.ToolResult(
                output=f"<error: path escapes the workdir: {target}>",
                failure_reason="outside workdir",
            )
        lines, reason = _read_lines(path)
        if lines is None:
            return registry.ToolResult(
                output=f"<error: cannot read {target}: {reason}>",
                failure_reason=reason,
            )
        try:
            offset = int(tool_input.get("offset", 1))
        except (TypeError, ValueError):
            offset = 1
        try:
            limit = int(tool_input.get("limit", max_lines))
        except (TypeError, ValueError):
            limit = max_lines
        offset = max(offset, 1)
        limit = max(min(limit, max_lines * 10), 1)
        total = len(lines)
        start = min(offset, total + 1)
        shown = lines[start - 1 : start - 1 + limit]
        rel = path.relative_to(context.workdir.resolve())
        end = start + len(shown) - 1 if shown else start - 1
        header = f"--- {rel} (lines {start}-{end} of {total}) ---"
        body = "\n".join(
            f"{n}: {line}" for n, line in zip(range(start, end + 1), shown, strict=True)
        )
        output = f"{header}\n{body}" if body else header
        if len(output) > max_bytes:
            cut = output[:max_bytes]
            output = (
                f"{header}\n<truncated, showing first {max_bytes} of "
                f"~{len(output)} chars>\n" + cut[len(header) + 1 :]
            )
        return registry.ToolResult(output=output)

    return registry.Tool(
        name=READ_TOOL_NAME,
        description=(
            "Read a file under the workdir and return a line-numbered slice. "
            "Paths are relative to the workdir and cannot escape it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workdir.",
                },
                "offset": {
                    "type": "integer",
                    "description": "First line to show (1-based, default 1).",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Max lines to show (default {DEFAULT_READ_MAX_LINES}).",
                },
            },
            "required": ["path"],
        },
        handler=_handle,
    )
