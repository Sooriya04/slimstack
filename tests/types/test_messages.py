"""Unit tests for Message and ToolCall contracts in slimstack.types."""

from typing import get_args

import pytest
from pydantic import ValidationError

from slimstack.types import Message, Role, ToolCall


def test_public_exports() -> None:
    """Verify that public types are cleanly exported and Role literal matches expected set."""
    expected_roles = {"system", "user", "assistant", "tool"}
    assert set(get_args(Role)) == expected_roles


def test_standard_system_message() -> None:
    """Verify standard system message creation and defaults."""
    msg = Message(role="system", content="You are a helpful AI assistant.")
    assert msg.role == "system"
    assert msg.content == "You are a helpful AI assistant."
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_standard_user_message() -> None:
    """Verify standard user message creation and defaults."""
    msg = Message(role="user", content="Calculate 2 + 2")
    assert msg.role == "user"
    assert msg.content == "Calculate 2 + 2"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_standard_assistant_message() -> None:
    """Verify standard assistant text response."""
    msg = Message(role="assistant", content="The answer is 4.")
    assert msg.role == "assistant"
    assert msg.content == "The answer is 4."
    assert msg.tool_calls is None
    assert msg.tool_call_id is None


def test_assistant_multiple_tool_calls() -> None:
    """Verify assistant message containing multiple tool calls."""
    tc1 = ToolCall(id="call_weather_1", name="get_weather", arguments={"city": "Tokyo", "unit": "celsius"})
    tc2 = ToolCall(id="call_traffic_2", name="get_traffic", arguments={"city": "Tokyo", "mode": "driving"})

    msg = Message(role="assistant", tool_calls=[tc1, tc2])
    assert msg.role == "assistant"
    assert msg.content is None
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 2
    assert msg.tool_calls[0] == tc1
    assert msg.tool_calls[1] == tc2
    assert msg.tool_calls[0].arguments == {"city": "Tokyo", "unit": "celsius"}
    assert msg.tool_calls[1].arguments == {"city": "Tokyo", "mode": "driving"}
    # Sequence should be coerced into an immutable tuple
    assert isinstance(msg.tool_calls, tuple)


def test_tool_result_message() -> None:
    """Verify tool result message referencing a tool call id."""
    msg = Message(
        role="tool",
        content='{"temperature": 18, "condition": "Sunny"}',
        tool_call_id="call_weather_1",
    )
    assert msg.role == "tool"
    assert msg.content == '{"temperature": 18, "condition": "Sunny"}'
    assert msg.tool_call_id == "call_weather_1"
    assert msg.tool_calls is None


def test_optional_content_behavior() -> None:
    """Verify content is optional and can be omitted, None, or string."""
    # Omitted content defaults to None
    msg_omitted = Message(role="assistant")
    assert msg_omitted.content is None

    # Explicit None content for tool call responses
    tc = ToolCall(id="call_1", name="echo", arguments={"msg": "hi"})
    msg_none = Message(role="assistant", content=None, tool_calls=[tc])
    assert msg_none.content is None
    assert msg_none.tool_calls == (tc,)

    # CoT / commentary accompanying tool call
    msg_with_thought = Message(
        role="assistant",
        content="I will check the tool output first.",
        tool_calls=[tc],
    )
    assert msg_with_thought.content == "I will check the tool output first."


@pytest.mark.parametrize("invalid_role", ["admin", "human", "ai", "developer", "function", "", "USER"])
def test_invalid_roles(invalid_role: str) -> None:
    """Verify that roles outside the provider-independent Role type fail validation."""
    with pytest.raises(ValidationError):
        Message(role=invalid_role, content="Test")  # type: ignore[arg-type]


def test_invalid_field_types_on_message() -> None:
    """Verify validation errors when incorrect types are supplied to Message fields."""
    # Non-string role
    with pytest.raises(ValidationError):
        Message(role=123, content="hello")  # type: ignore[arg-type]

    # Non-string / non-None content
    with pytest.raises(ValidationError):
        Message(role="user", content=123)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Message(role="user", content=["hello"])  # type: ignore[arg-type]

    # Non-sequence or non-ToolCall tool_calls
    with pytest.raises(ValidationError):
        Message(role="assistant", tool_calls="not-a-sequence")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Message(role="assistant", tool_calls=[{"id": "call_1"}])  # missing name and arguments

    # Non-string tool_call_id
    with pytest.raises(ValidationError):
        Message(role="tool", content="ok", tool_call_id=999)  # type: ignore[arg-type]

    # Missing required role field
    with pytest.raises(ValidationError):
        Message()  # type: ignore[call-arg]


def test_invalid_field_types_on_tool_call() -> None:
    """Verify validation errors when incorrect types are supplied to ToolCall fields."""
    # Non-string id
    with pytest.raises(ValidationError):
        ToolCall(id=123, name="calc", arguments={})  # type: ignore[arg-type]

    # Non-string name
    with pytest.raises(ValidationError):
        ToolCall(id="call_1", name=123, arguments={})  # type: ignore[arg-type]

    # Non-dict arguments
    with pytest.raises(ValidationError):
        ToolCall(id="call_1", name="calc", arguments="not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ToolCall(id="call_1", name="calc", arguments=123)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ToolCall(id="call_1", name="calc", arguments=["a", "b"])  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ToolCall(id="call_1", name="calc", arguments=None)  # type: ignore[arg-type]

    # Missing required fields
    with pytest.raises(ValidationError):
        ToolCall(id="call_1", name="calc")  # missing arguments # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        ToolCall(name="calc", arguments={})  # missing id # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        ToolCall(id="call_1", arguments={})  # missing name # type: ignore[call-arg]


def test_frozen_immutability_field_reassignment() -> None:
    """Verify that frozen models forbid reassigning their fields."""
    tc = ToolCall(id="call_1", name="search", arguments={"q": "slimstack"})
    msg = Message(role="assistant", content="Initial", tool_calls=[tc], tool_call_id=None)

    # Reassigning Message fields must fail
    with pytest.raises(ValidationError):
        msg.role = "user"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        msg.content = "Mutated"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        msg.tool_calls = ()  # type: ignore[misc]

    with pytest.raises(ValidationError):
        msg.tool_call_id = "call_2"  # type: ignore[misc]

    # Reassigning ToolCall fields must fail
    with pytest.raises(ValidationError):
        tc.id = "call_2"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        tc.name = "new_search"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        tc.arguments = {"q": "mutated"}  # type: ignore[misc]


def test_nested_immutability_evaluation() -> None:
    """Verify and evaluate nested mutable values behavior.

    1. tool_calls is typed as a tuple, preventing in-place sequence mutations (.append or index assignment).
    2. arguments is a standard JSON-compatible Python dict: while the field reference is frozen on ToolCall,
       the underlying dict keys can be mutated in-place unless defensively copied.
    """
    tc = ToolCall(id="call_1", name="search", arguments={"q": "slimstack"})
    msg = Message(role="assistant", tool_calls=[tc])

    # Sequence cannot be mutated in place
    assert isinstance(msg.tool_calls, tuple)
    with pytest.raises(TypeError):
        msg.tool_calls[0] = tc  # type: ignore[index]

    with pytest.raises(AttributeError):
        msg.tool_calls.append(tc)  # type: ignore[attr-defined]

    # ToolCall arguments field cannot be reassigned to another object
    with pytest.raises(ValidationError):
        tc.arguments = {}  # type: ignore[misc]

    # However, Python dict references remain mutable internally to preserve standard JSON compatibility
    tc.arguments["q"] = "updated_query"
    assert tc.arguments["q"] == "updated_query"
