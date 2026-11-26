"""Unit tests for engine I/O contracts in slimstack.types."""

import pytest
from pydantic import ValidationError

from slimstack.types import Message, ModelResponse, StreamChunk, ToolCall, Usage


def test_valid_usage_construction() -> None:
    """Verify standard Usage metrics construction."""
    usage = Usage(prompt_tokens=15, completion_tokens=35, total_tokens=50)
    assert usage.prompt_tokens == 15
    assert usage.completion_tokens == 35
    assert usage.total_tokens == 50


def test_usage_zero_tokens_allowed() -> None:
    """Verify non-negative zero tokens is valid."""
    usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


@pytest.mark.parametrize(
    "prompt,completion,total",
    [
        (-1, 10, 10),
        (10, -1, 10),
        (10, 10, -1),
        (-5, -5, -10),
    ],
)
def test_usage_negative_tokens_rejected(prompt: int, completion: int, total: int) -> None:
    """Verify validation error when token counts are negative."""
    with pytest.raises(ValidationError):
        Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def test_usage_invalid_field_types() -> None:
    """Verify validation error when token counts are non-integers or missing."""
    with pytest.raises(ValidationError):
        Usage(prompt_tokens="not-int", completion_tokens=10, total_tokens=10)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Usage(prompt_tokens=10, completion_tokens=None, total_tokens=10)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Usage(prompt_tokens=10)  # type: ignore[call-arg]


def test_model_response_with_usage() -> None:
    """Verify ModelResponse with message and usage populated."""
    msg = Message(role="assistant", content="Engine response text")
    usage = Usage(prompt_tokens=12, completion_tokens=4, total_tokens=16)

    resp = ModelResponse(message=msg, usage=usage)
    assert resp.message == msg
    assert resp.usage == usage
    assert resp.message.content == "Engine response text"


def test_model_response_without_usage() -> None:
    """Verify ModelResponse usage defaults to None when provider does not report it."""
    msg = Message(role="assistant", content="Zero usage response")
    resp = ModelResponse(message=msg)
    assert resp.message == msg
    assert resp.usage is None


def test_model_response_with_tool_calls_message() -> None:
    """Verify ModelResponse containing an assistant message with tool calls."""
    tc = ToolCall(id="call_engine_01", name="search_docs", arguments={"term": "pydantic"})
    msg = Message(role="assistant", tool_calls=[tc])
    resp = ModelResponse(message=msg)

    assert resp.message.role == "assistant"
    assert resp.message.content is None
    assert resp.message.tool_calls is not None
    assert resp.message.tool_calls[0].name == "search_docs"


def test_model_response_invalid_fields() -> None:
    """Verify validation error when ModelResponse receives invalid types or missing message."""
    with pytest.raises(ValidationError):
        ModelResponse()  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        ModelResponse(message="not-a-message")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ModelResponse(message=Message(role="assistant", content="hi"), usage="not-usage")  # type: ignore[arg-type]


def test_stream_chunk_standard() -> None:
    """Verify standard StreamChunk with delta token."""
    chunk = StreamChunk(delta="Hello")
    assert chunk.delta == "Hello"
    assert chunk.finish_reason is None
    assert chunk.usage is None


def test_stream_chunk_empty_delta() -> None:
    """Verify empty string delta is allowed for metadata/final chunks."""
    chunk = StreamChunk(delta="", finish_reason="stop")
    assert chunk.delta == ""
    assert chunk.finish_reason == "stop"
    assert chunk.usage is None


def test_stream_chunk_final_with_usage_and_custom_finish_reason() -> None:
    """Verify final StreamChunk carrying unrestricted finish_reason and usage."""
    usage = Usage(prompt_tokens=8, completion_tokens=16, total_tokens=24)
    chunk = StreamChunk(delta="", finish_reason="tool_calls", usage=usage)
    assert chunk.delta == ""
    assert chunk.finish_reason == "tool_calls"
    assert chunk.usage == usage


def test_stream_chunk_invalid_fields() -> None:
    """Verify validation error when StreamChunk receives invalid types or missing delta."""
    with pytest.raises(ValidationError):
        StreamChunk()  # missing delta # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        StreamChunk(delta=123)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        StreamChunk(delta="ok", finish_reason=123)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        StreamChunk(delta="ok", usage={"prompt_tokens": -1, "completion_tokens": 0, "total_tokens": 0})  # invalid usage


def test_frozen_immutability_attribute_reassignment() -> None:
    """Verify frozen models forbid reassigning their fields."""
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    msg = Message(role="assistant", content="Test")
    resp = ModelResponse(message=msg, usage=usage)
    chunk = StreamChunk(delta="token", finish_reason="stop", usage=usage)

    with pytest.raises(ValidationError):
        usage.prompt_tokens = 15  # type: ignore[misc]

    with pytest.raises(ValidationError):
        usage.completion_tokens = 25  # type: ignore[misc]

    with pytest.raises(ValidationError):
        usage.total_tokens = 40  # type: ignore[misc]

    with pytest.raises(ValidationError):
        resp.message = Message(role="user", content="Other")  # type: ignore[misc]

    with pytest.raises(ValidationError):
        resp.usage = None  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.delta = "other"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.finish_reason = "length"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.usage = None  # type: ignore[misc]
