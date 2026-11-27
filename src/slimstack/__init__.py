"""
slimstack: The unbloated AI stack.
Minimal, zero-dependency inference and agent primitives.
"""

from slimstack.models import Model, ModelError
from slimstack.tools import Tool, ToolError, ToolRegistry, ToolResult, tool

__version__ = "0.0.1"
__author__ = "Sooriya"

__all__ = [
    "Model",
    "ModelError",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "tool",
    "__version__",
]
