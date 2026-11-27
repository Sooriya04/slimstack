# Tool Reflection & Execution (`slimstack.tools`)

The `slimstack.tools` package provides function schema reflection via `@tool`, along with a robust `ToolRegistry` for registering, inspecting, and safely executing `ToolCall` objects.

---

## 1. Quickstart & Command-Line Usage

```bash
uv run python -c '
from slimstack import ToolRegistry, tool
from slimstack.types import ToolCall

registry = ToolRegistry()

@registry.register
def calculate_area(width: float, height: float) -> float:
    """Calculate the area of a rectangle.

    Args:
        width: Rectangle width.
        height: Rectangle height.
    """
    return width * height

# 1. Inspect OpenAI-compatible schema
print("Schemas:\n", registry.get_schemas())

# 2. Execute a ToolCall
call = ToolCall(id="call_1", name="calculate_area", arguments={"width": 4.5, "height": 2.0})
result = registry.execute(call)
print("Execution success:", result.success)
print("Result content:", result.content)

# 3. Convert directly to a tool Message
msg = result.to_message()
print("Tool Message:", msg.model_dump())
'
```

---

## 2. API Reference

### `ToolRegistry(tools=None)`
A registry and execution coordinator for callable `Tool` objects.

| Method | Return Type | Description |
|---|---|---|
| `register(tool_or_func, *, override=False)` | `Tool` | Registers a tool. Raises `ToolError` if duplicate unless `override=True`. |
| `get(name)` | `Tool \| None` | Looks up a tool by name, returning `None` if not registered. |
| `__getitem__(name)` | `Tool` | Looks up a tool by name, raising `KeyError` if not registered. |
| `get_schemas()` | `list[dict]` | Returns OpenAI-compatible schemas for all registered tools. |
| `execute(tool_call)` | `ToolResult` | Validates arguments and executes the tool, returning `ToolResult`. |
| `execute_to_message(tool_call)` | `Message` | Executes and directly returns a `Message(role="tool", ...)`. |

### `ToolResult`
A frozen Pydantic model representing execution output:

| Field | Type | Description |
|---|---|---|
| `tool_call_id` | `str` | Matching `id` of the executed `ToolCall`. |
| `name` | `str` | Name of the tool. |
| `content` | `str` | Formatted output or error string for LLM consumption. |
| `success` | `bool` | `True` if execution succeeded, `False` on failure or error. |
| `error` | `str \| None` | Exception message or validation failure details. |
| `raw_output` | `Any` | Original unformatted Python return value. |

---

## 3. Error Handling & Safety
1. **Pre-execution Argument Validation:** Arguments are validated against the function's Pydantic model before execution. Missing parameters or type mismatches yield a `ToolResult(success=False, error=...)` without crashing the app.
2. **Exception Containment:** If a tool function raises an exception during execution (e.g. division by zero, network failure), `execute()` captures the error, stores the original exception message, and returns `success=False`.
3. **Duplicate Protection:** Registering a tool with an existing name raises `ToolError` unless explicitly allowed via `override=True`.
