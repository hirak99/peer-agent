"""Edit tool: exact search-and-replace of a file under the workdir."""

import pathlib

from service.backends import base
from service.tools import registry

from typing import Any

# Machine name referenced by model tool_use blocks.
EDIT_TOOL_NAME = "edit"


def _resolve(target: str, workdir: pathlib.Path) -> pathlib.Path | None:
    """Resolves a tool-supplied path, confining it inside the workdir."""
    candidate = (workdir / target).resolve()
    try:
        candidate.relative_to(workdir.resolve())
    except ValueError:
        return None
    return candidate


def _load_text(path: pathlib.Path) -> tuple[str | None, str | None]:
    """Reads a file as UTF-8 text, or returns a failure reason."""
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
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return None, "not UTF-8 text"


def build_edit_tool() -> registry.Tool:
    """Builds the registry entry for the edit tool."""

    def _handle(
        tool_input: dict[str, Any], context: base.ToolContext
    ) -> registry.ToolResult:
        if context.workdir is None:
            return registry.ToolResult(
                output="<error: edit offered without a workdir; nothing changed>",
                failure_reason="no workdir",
            )
        target = str(tool_input.get("path", ""))
        if not target:
            return registry.ToolResult(
                output="<error: missing required 'path'>",
                failure_reason="missing path",
            )
        old_string = tool_input.get("old_string", "")
        if not isinstance(old_string, str) or not old_string:
            return registry.ToolResult(
                output="<error: missing required 'old_string'>",
                failure_reason="missing old_string",
            )
        new_string = tool_input.get("new_string", "")
        if not isinstance(new_string, str):
            return registry.ToolResult(
                output="<error: 'new_string' must be a string>",
                failure_reason="bad new_string",
            )
        if new_string == old_string:
            return registry.ToolResult(
                output="<error: old_string and new_string are identical; nothing changed>",
                failure_reason="no change",
            )
        replace_all = bool(tool_input.get("replace_all", False))
        path = _resolve(target, context.workdir)
        if path is None:
            return registry.ToolResult(
                output=f"<error: path escapes the workdir: {target}>",
                failure_reason="outside workdir",
            )
        text, reason = _load_text(path)
        if text is None:
            return registry.ToolResult(
                output=f"<error: cannot edit {target}: {reason}>",
                failure_reason=reason,
            )
        matches = text.count(old_string)
        if matches == 0:
            return registry.ToolResult(
                output=f"<error: no match for old_string in {target}>",
                failure_reason="no match",
            )
        if matches > 1 and not replace_all:
            return registry.ToolResult(
                output=(
                    f"<error: old_string matches {matches} times in {target}; "
                    "refine it to be unique or set replace_all=true>"
                ),
                failure_reason=f"ambiguous: {matches} matches",
            )
        first_lineno = text.count("\n", 0, text.index(old_string)) + 1
        updated = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        try:
            path.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return registry.ToolResult(
                output=f"<error: cannot write {target}: {exc.strerror or exc}>",
                failure_reason="unwritable",
            )
        rel = path.relative_to(context.workdir.resolve())
        done = matches if replace_all else 1
        noun = "occurrence" if done == 1 else "occurrences"
        return registry.ToolResult(
            output=f"Edited {rel}: replaced {done} {noun} (first at line {first_lineno})."
        )

    return registry.Tool(
        name=EDIT_TOOL_NAME,
        description=(
            "Replace exact text in a file under the workdir. "
            "old_string must match exactly once unless replace_all is true; "
            "paths are relative to the workdir and cannot escape it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the workdir.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find (must be unique).",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence (default false).",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=_handle,
    )
