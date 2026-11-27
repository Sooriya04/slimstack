"""Tool registry and execution engine for SlimStack."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from slimstack.tools.decorators import Tool, ToolError, tool
from slimstack.types.messages import Message, ToolCall


def _format_output(output: Any) -> str:
    """Safely format tool execution output to a string for model consumption."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, default=str)
    except Exception:
        return str(output)


class ToolResult(BaseModel):
    """The result of executing a ToolCall."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    name: str
    content: str
    success: bool
    error: str | None = None
    raw_output: Any = None

    def to_message(self) -> Message:
        """Convert this execution result into a SlimStack tool message."""
        return Message(
            role="tool",
            content=self.content,
            tool_call_id=self.tool_call_id,
        )


class ToolRegistry:
    """Registry and execution engine for callable Tool objects."""

    def __init__(self, tools: Sequence[Tool | Callable[..., Any]] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(
        self,
        tool_or_func: Tool | Callable[..., Any],
        *,
        override: bool = False,
    ) -> Tool:
        """Register a Tool or ordinary function.

        Raises ToolError if a tool with the same name exists and override is False.
        """
        t = tool_or_func if isinstance(tool_or_func, Tool) else tool(tool_or_func)
        if t.name in self._tools and not override:
            raise ToolError(
                f"Tool '{t.name}' is already registered. Set override=True to replace it."
            )
        self._tools[t.name] = t
        return t

    def get(self, name: str) -> Tool | None:
        """Look up a registered tool by name, returning None if not found."""
        return self._tools.get(name)

    def __getitem__(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def list_tools(self) -> list[Tool]:
        """Return a list of all registered Tool instances."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible JSON function schemas for all registered tools."""
        return [t.schema for t in self._tools.values()]

    def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a ToolCall by resolving the tool name and validating arguments."""
        t = self.get(tool_call.name)
        if t is None:
            err_msg = f"Error: Tool '{tool_call.name}' is not registered."
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=err_msg,
                success=False,
                error=err_msg,
                raw_output=None,
            )

        try:
            validated_kwargs = t.validate_arguments(tool_call.arguments)
        except Exception as exc:
            err_msg = f"Error: Invalid arguments for tool '{tool_call.name}': {exc}"
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=err_msg,
                success=False,
                error=str(exc),
                raw_output=None,
            )

        try:
            output = t(**validated_kwargs)
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=_format_output(output),
                success=True,
                error=None,
                raw_output=output,
            )
        except Exception as exc:
            err_msg = f"Error executing tool '{tool_call.name}': {exc}"
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=err_msg,
                success=False,
                error=str(exc),
                raw_output=None,
            )

    def execute_to_message(self, tool_call: ToolCall) -> Message:
        """Execute a ToolCall and directly return a SlimStack Message."""
        return self.execute(tool_call).to_message()
