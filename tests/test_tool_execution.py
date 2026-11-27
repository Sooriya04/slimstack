"""Tests for ToolRegistry and tool execution."""

import json
from typing import Any

from pydantic import BaseModel
import pytest

from slimstack.tools import Tool, ToolError, ToolRegistry, ToolResult, tool
from slimstack.types import ToolCall


def test_registration_lookup_and_membership() -> None:
    registry = ToolRegistry()

    @tool
    def add(a: int, b: int) -> int:
        return a + b

    def greet(name: str) -> str:
        return f"Hello, {name}"

    registry.register(add)
    registry.register(greet)  # auto-wrapped

    assert len(registry) == 2
    assert "add" in registry
    assert "greet" in registry
    assert "missing" not in registry

    assert registry.get("add") is not None
    assert registry.get("add").name == "add"  # type: ignore[union-attr]
    assert registry["greet"].name == "greet"
    assert registry.get("missing") is None

    with pytest.raises(KeyError):
        _ = registry["missing"]


def test_duplicate_registration_behavior() -> None:
    registry = ToolRegistry()

    @tool
    def fetch() -> str:
        return "first"

    registry.register(fetch)

    # Re-registering with same name raises ToolError by default
    @tool(name="fetch")
    def fetch_replacement() -> str:
        return "second"

    with pytest.raises(ToolError) as exc_info:
        registry.register(fetch_replacement)
    assert "already registered" in str(exc_info.value)

    # Re-registering with override=True succeeds
    registry.register(fetch_replacement, override=True)
    assert len(registry) == 1
    assert registry["fetch"]() == "second"


def test_get_schemas() -> None:
    @tool
    def search(query: str, limit: int = 5) -> list[str]:
        """Search items."""
        return []

    registry = ToolRegistry([search])
    schemas = registry.get_schemas()

    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "search"
    assert "query" in schemas[0]["function"]["parameters"]["properties"]


def test_successful_execution_and_message_conversion() -> None:
    @tool
    def multiply(x: int, y: int = 2) -> int:
        """Multiply numbers."""
        return x * y

    registry = ToolRegistry([multiply])
    call = ToolCall(id="call_mult_1", name="multiply", arguments={"x": 6, "y": 7})

    result = registry.execute(call)
    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.error is None
    assert result.tool_call_id == "call_mult_1"
    assert result.name == "multiply"
    assert result.content == "42"
    assert result.raw_output == 42

    # Conversion to SlimStack Message
    msg = result.to_message()
    assert msg.role == "tool"
    assert msg.content == "42"
    assert msg.tool_call_id == "call_mult_1"

    # execute_to_message convenience method
    msg2 = registry.execute_to_message(call)
    assert msg2.content == "42"
    assert msg2.tool_call_id == "call_mult_1"


def test_argument_validation_missing_required_parameter() -> None:
    @tool
    def divide(a: float, b: float) -> float:
        return a / b

    registry = ToolRegistry([divide])
    call = ToolCall(id="call_div_1", name="divide", arguments={"a": 10.0})  # missing b

    result = registry.execute(call)
    assert result.success is False
    assert result.error is not None
    assert "b" in result.error
    assert result.raw_output is None
    assert "Error: Invalid arguments" in result.content

    # Calling workflow must not crash, and message can be returned to model
    msg = result.to_message()
    assert msg.role == "tool"
    assert "Error" in (msg.content or "")


def test_argument_validation_invalid_type() -> None:
    @tool
    def parse_age(age: int) -> str:
        return f"Age: {age}"

    registry = ToolRegistry([parse_age])
    call = ToolCall(id="call_age_1", name="parse_age", arguments={"age": "not-an-integer"})

    result = registry.execute(call)
    assert result.success is False
    assert result.error is not None
    assert "Input should be a valid integer" in result.error


def test_unknown_tool_execution() -> None:
    registry = ToolRegistry()
    call = ToolCall(id="call_missing_1", name="unregistered_tool", arguments={})

    result = registry.execute(call)
    assert result.success is False
    assert result.error == "Error: Tool 'unregistered_tool' is not registered."
    assert "is not registered" in result.content


def test_function_exception_handling_preserves_debug_info() -> None:
    @tool
    def unstable_tool(n: int) -> float:
        """Divide 100 by n."""
        return 100 / n

    registry = ToolRegistry([unstable_tool])
    call = ToolCall(id="call_zero_1", name="unstable_tool", arguments={"n": 0})

    result = registry.execute(call)
    assert result.success is False
    assert "division by zero" in str(result.error)
    assert "Error executing tool 'unstable_tool': division by zero" in result.content
    assert result.raw_output is None


def test_output_formatting_types() -> None:
    @tool
    def return_dict() -> dict[str, Any]:
        return {"status": "ok", "count": 3}

    @tool
    def return_none() -> None:
        return None

    registry = ToolRegistry([return_dict, return_none])

    r_dict = registry.execute(ToolCall(id="c1", name="return_dict", arguments={}))
    assert r_dict.success is True
    assert json.loads(r_dict.content) == {"status": "ok", "count": 3}
    assert r_dict.raw_output == {"status": "ok", "count": 3}

    r_none = registry.execute(ToolCall(id="c2", name="return_none", arguments={}))
    assert r_none.success is True
    assert r_none.content == ""
    assert r_none.raw_output is None


@pytest.mark.asyncio
async def test_async_tool_execution_safety_and_aexecute() -> None:
    registry = ToolRegistry()

    @registry.register
    async def async_greet(name: str) -> str:
        return f"async hello {name}"

    call = ToolCall(id="call_async", name="async_greet", arguments={"name": "Alice"})

    # 1. Synchronous execute() must fail cleanly without unawaited warning
    sync_result = registry.execute(call)
    assert sync_result.success is False
    assert "asynchronous and cannot be executed synchronously" in sync_result.content

    # 2. Asynchronous aexecute() succeeds
    async_result = await registry.aexecute(call)
    assert async_result.success is True
    assert async_result.content == "async hello Alice"

    # 3. aexecute_to_message() produces Message
    msg = await registry.aexecute_to_message(call)
    assert msg.role == "tool"
    assert msg.content == "async hello Alice"
    assert msg.tool_call_id == "call_async"


def test_positional_only_tool_parameters() -> None:
    registry = ToolRegistry()

    @registry.register
    def add_pos(a: int, /, b: int = 10) -> int:
        return a + b

    call = ToolCall(id="call_pos", name="add_pos", arguments={"a": 5, "b": 15})
    res = registry.execute(call)
    assert res.success is True
    assert res.content == "20"


def test_pydantic_model_output_formatting() -> None:
    registry = ToolRegistry()

    class UserProfile(BaseModel):
        user_id: int
        username: str

    @registry.register
    def get_user(uid: int) -> UserProfile:
        return UserProfile(user_id=uid, username="lois")

    call = ToolCall(id="call_model", name="get_user", arguments={"uid": 42})
    res = registry.execute(call)
    assert res.success is True
    assert json.loads(res.content) == {"user_id": 42, "username": "lois"}
