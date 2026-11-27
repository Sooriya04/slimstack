# Tool Reflection (`slimstack.tools`)

The `@tool` decorator converts standard Python functions into callable tools that automatically generate OpenAI-compatible JSON function schemas via Pydantic.

---

## 1. Quickstart & Command-Line Usage

Test tool reflection directly in your terminal:

```bash
uv run python -c '
from typing import Annotated
from slimstack import tool

@tool
def get_weather(city: str, unit: Annotated[str, "Temperature unit: c or f"] = "c") -> str:
    """Fetch the current weather forecast for a city.

    Args:
        city: The name of the city (e.g. San Francisco).
        unit: Metric or imperial units.
    """
    return f"Weather in {city}: 22 degrees {unit}"

# Call like a regular function:
print("Function call result:", get_weather("Paris"))

# Inspect the generated OpenAI function schema:
import json
print("JSON Schema:\n", json.dumps(get_weather.schema, indent=2))
'
```

---

## 2. API Reference

### `@tool(func=None, *, name=None, description=None)`
Decorator producing a `Tool` instance wrapping the function.

- **`name`**: `str | None` — Custom tool name. If omitted, defaults to `func.__name__`.
- **`description`**: `str | None` — Custom tool description. If omitted, defaults to the cleaned docstring (`inspect.cleandoc(func.__doc__)`).

### `Tool` Attributes & Properties
- **`tool.name`**: Name of the tool.
- **`tool.description`**: Cleaned docstring or custom description.
- **`tool.func`**: Reference to the original unwrapped function.
- **`tool.schema`**: The OpenAI-compatible function schema dictionary:
  ```json
  {
    "type": "function",
    "function": {
      "name": "...",
      "description": "...",
      "parameters": {
        "type": "object",
        "properties": { ... },
        "required": [ ... ]
      }
    }
  }
  ```
- **`tool(*args, **kwargs)`**: Directly calls the underlying function with supplied arguments.

---

## 3. Features & Supported Type Annotations

1. **Standard Primitives:** `str`, `int`, `float`, `bool`.
2. **Collections & Nested Types:** `list[str]`, `dict[str, Any]`, `list[dict[str, int]]`.
3. **Optional Parameters:** `str | None = None` or `Optional[int] = None`. Parameters with defaults are automatically excluded from the `"required"` list.
4. **Parameter Descriptions:**
   - Extracted from `Annotated[T, "description"]` or `Annotated[T, Field(description="...")]`.
   - Extracted from Google or Sphinx docstrings (`Args:`, `Parameters:`, `:param name: ...`).
5. **Pydantic Nested Models:** If a parameter is annotated with a Pydantic `BaseModel`, its schema is reflected via standard `$defs` and `$ref`.
6. **Zero Side-Effects:** Schema inspection uses `inspect` and `pydantic.create_model` without executing the underlying function.
