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

    def anthropic_schemas(self) -> list[dict[str, Any]]:
        """Renders the registry as Anthropic tool parameter dicts."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def openai_schemas(self) -> list[dict[str, Any]]:
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
            for tool in self._tools.values()
        ]
