"""Registry of tools the peer agent may invoke."""

import dataclasses
from collections.abc import Callable

from typing import Any


@dataclasses.dataclass
class Tool:
    """One named capability offered to the model backend."""

    # Machine name referenced by model tool_use blocks.
    name: str
    # Description sent to the model alongside the input schema.
    description: str
    # JSON Schema object describing the tool input.
    input_schema: dict[str, Any]
    # Handler mapping decoded input to result text.
    handler: Callable[[dict[str, Any]], str]


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
