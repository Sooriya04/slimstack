"""Tools package for SlimStack."""

from slimstack.tools.decorators import Tool, ToolError, tool
from slimstack.tools.registry import ToolRegistry, ToolResult

__all__ = ["Tool", "ToolError", "ToolRegistry", "ToolResult", "tool"]
