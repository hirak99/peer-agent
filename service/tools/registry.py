"""Registry of tools the peer agent may invoke."""

import dataclasses
from collections.abc import Callable

from service.backends import base

from typing import Any


@dataclasses.dataclass
class ToolResult:
    """One tool handler outcome: result text plus optional failure reason."""

    # Truncated handler output fed back to the model.
    output: str
    # Why the call failed; None means success. "unknown" when the handler
    # knows it failed but not why. Presence (not content) is the signal.
    failure_reason: str | None = None


@dataclasses.dataclass
class Tool:
    """One named capability offered to the model backend."""

    # Machine name referenced by model tool_use blocks.
    name: str
    # Description sent to the model alongside the input schema.
    description: str
    # JSON Schema object describing the tool input.
    input_schema: dict[str, Any]
    # Handler mapping decoded input and turn context to a result.
    handler: Callable[[dict[str, Any], base.ToolContext], ToolResult]
    # Whether the tool needs a workdir; False means it stays offered
    # on null-cwd turns where filesystem tools are withheld.
    requires_workdir: bool = True


class ToolRegistry:
    """Ordered set of tools, open for extension with new surfaces."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Adds a tool, replacing any previous tool with the same name."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Looks up a tool by name, or None when unregistered."""
        return self._tools.get(name)

    def available(self, include_workdir_tools: bool = True) -> list[Tool]:
        """Lists tools filtered for the turn context.

        Null-cwd turns pass False, withholding filesystem tools while
        keeping workdir-independent ones (e.g. web_fetch).
        """
        if include_workdir_tools:
            return list(self._tools.values())
        return [tool for tool in self._tools.values() if not tool.requires_workdir]

    def anthropic_schemas(
        self, include_workdir_tools: bool = True
    ) -> list[dict[str, Any]]:
        """Renders the registry as Anthropic tool parameter dicts."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self.available(include_workdir_tools)
        ]

    def openai_schemas(
        self, include_workdir_tools: bool = True
    ) -> list[dict[str, Any]]:
        """Renders the registry as OpenAI function-tool parameter dicts."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.available(include_workdir_tools)
        ]
