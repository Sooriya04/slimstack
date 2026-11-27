"""Tests for the unified Model engine using httpx.MockTransport."""

import json
from typing import Any

import httpx
import pytest

from slimstack.models import Model, ModelError
from slimstack.types import Message, ToolCall


def _make_mock_client(handler: Any) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


@pytest.mark.asyncio
async def test_successful_generation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.read())
        assert body["model"] == "gpt-4o"
        assert body["messages"] == [{"role": "user", "content": "Hello"}]

        resp_body = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi there!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        return httpx.Response(200, json=resp_body)

    client = _make_mock_client(handler)
    model = Model(model_name="gpt-4o", api_key="test-key", client=client)

    response = await model.agenerate([Message(role="user", content="Hello")])
    assert response.message.role == "assistant"
    assert response.message.content == "Hi there!"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 5
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_message_serialization_and_tool_call_conversion() -> None:
    recorded_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(json.loads(request.read()))
        resp_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": json.dumps({"query": "python"}),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": None,
        }
        return httpx.Response(200, json=resp_body)

    client = _make_mock_client(handler)
    model = Model(model_name="gpt-4o", client=client)

    messages = [
        Message(role="system", content="System prompt"),
        Message(role="user", content="Search query"),
        Message(
            role="assistant",
            content=None,
            tool_calls=(ToolCall(id="c1", name="search", arguments={"q": "slimstack"}),),
        ),
        Message(role="tool", content='{"results": 1}', tool_call_id="c1"),
    ]

    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    response = await model.agenerate(messages, tools=tools)

    req = recorded_requests[0]
    assert len(req["messages"]) == 4
    assert req["messages"][0] == {"role": "system", "content": "System prompt"}
    assert req["messages"][1] == {"role": "user", "content": "Search query"}
    assert req["messages"][2]["tool_calls"][0]["id"] == "c1"
    assert req["messages"][2]["tool_calls"][0]["function"]["name"] == "search"
    assert json.loads(req["messages"][2]["tool_calls"][0]["function"]["arguments"]) == {"q": "slimstack"}
    assert req["messages"][3] == {"role": "tool", "content": '{"results": 1}', "tool_call_id": "c1"}
    assert req["tools"] == tools

    assert response.message.role == "assistant"
    assert response.message.content is None
    assert response.message.tool_calls is not None
    assert len(response.message.tool_calls) == 1
    assert response.message.tool_calls[0].id == "call_abc"
    assert response.message.tool_calls[0].name == "lookup"
    assert response.message.tool_calls[0].arguments == {"query": "python"}
    assert response.usage is None


@pytest.mark.asyncio
async def test_missing_and_invalid_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    client = _make_mock_client(handler)
    model = Model(model_name="test-model", client=client)

    response = await model.agenerate([Message(role="user", content="Hi")])
    assert response.usage is None


@pytest.mark.asyncio
async def test_http_error_sanitization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text='{"error": "Unauthorized sk-super-secret-12345"}')

    client = _make_mock_client(handler)
    model = Model(model_name="test-model", api_key="sk-super-secret-12345", client=client)

    with pytest.raises(ModelError) as exc_info:
        await model.agenerate([Message(role="user", content="Hi")])

    error_str = str(exc_info.value)
    assert exc_info.value.status_code == 401
    assert "sk-super-secret-12345" not in error_str
    assert "[REDACTED]" in error_str


@pytest.mark.asyncio
async def test_auth_configuration_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. API key passed directly
    m1 = Model(model_name="test", api_key="key-direct")
    assert m1.api_key == "key-direct"
    assert m1._get_headers()["Authorization"] == "Bearer key-direct"

    # 2. API key from environment variable
    monkeypatch.setenv("OPENAI_API_KEY", "key-from-env")
    m2 = Model(model_name="test")
    assert m2.api_key == "key-from-env"
    assert m2._get_headers()["Authorization"] == "Bearer key-from-env"

    # 3. No API key (e.g. local Ollama)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    m3 = Model(model_name="test", base_url="http://localhost:11434/v1/")
    assert m3.api_key is None
    assert "Authorization" not in m3._get_headers()
    assert m3.base_url == "http://localhost:11434/v1"


@pytest.mark.asyncio
async def test_streaming_text_chunks_and_done() -> None:
    sse_body = (
        b": ping\n\n"
        b"data: {\"choices\": [{\"delta\": {\"content\": \"Hello\"}}]}\n\n"
        b"data: malformed-json\n\n"
        b"data: {\"choices\": [{\"delta\": {\"content\": \" world\"}}]}\n\n"
        b"data: {\"choices\": [{\"delta\": {}, \"finish_reason\": \"stop\"}], \"usage\": {\"prompt_tokens\": 3, \"completion_tokens\": 2, \"total_tokens\": 5}}\n\n"
        b"data: [DONE]\n\n"
        b"data: {\"choices\": [{\"delta\": {\"content\": \"after done\"}}]}\n\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse_body)

    client = _make_mock_client(handler)
    model = Model(model_name="gpt-4o", client=client)

    chunks = [chunk async for chunk in model.astream([Message(role="user", content="Hi")])]

    assert len(chunks) == 3
    assert chunks[0].delta == "Hello"
    assert chunks[0].finish_reason is None
    assert chunks[0].usage is None

    assert chunks[1].delta == " world"
    assert chunks[1].finish_reason is None

    assert chunks[2].delta == ""
    assert chunks[2].finish_reason == "stop"
    assert chunks[2].usage is not None
    assert chunks[2].usage.total_tokens == 5


@pytest.mark.asyncio
async def test_streaming_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    client = _make_mock_client(handler)
    model = Model(model_name="gpt-4o", client=client)

    with pytest.raises(ModelError) as exc_info:
        async for _ in model.astream([Message(role="user", content="Hi")]):
            pass

    assert exc_info.value.status_code == 500
    assert "HTTP 500 streaming error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_lifecycle_external_vs_internal() -> None:
    # 1. External client must NOT be closed automatically
    user_client = httpx.AsyncClient()
    model_external = Model(model_name="gpt-4o", client=user_client)
    await model_external.aclose()
    assert not user_client.is_closed
    await user_client.aclose()

    # 2. Internal client MUST be closed on aclose()
    model_internal = Model(model_name="gpt-4o")
    internal_client = model_internal._client
    assert not internal_client.is_closed
    await model_internal.aclose()
    assert internal_client.is_closed

    # 3. Async context manager properly closes internal client
    async with Model(model_name="gpt-4o") as ctx_model:
        ctx_client = ctx_model._client
        assert not ctx_client.is_closed
    assert ctx_client.is_closed


@pytest.mark.asyncio
async def test_scalar_tool_arguments_wrapped_in_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        resp_body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "calc",
                                    "arguments": "42",  # scalar JSON number
                                },
                            }
                        ],
                    }
                }
            ]
        }
        return httpx.Response(200, json=resp_body)

    client = _make_mock_client(handler)
    model = Model(model_name="gpt-4o", client=client)

    response = await model.agenerate([Message(role="user", content="calc")])
    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].arguments == {"raw": 42}


@pytest.mark.asyncio
async def test_empty_tool_calls_omitted_and_tool_null_content_coerced() -> None:
    recorded_requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(json.loads(request.read()))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "done"}}]},
        )

    client = _make_mock_client(handler)
    model = Model(model_name="gpt-4o", client=client)

    messages = [
        Message(role="assistant", content="thought", tool_calls=()),  # empty tuple
        Message(role="tool", content=None, tool_call_id="call_999"),  # null content
    ]

    await model.agenerate(messages)

    req = recorded_requests[0]
    # Empty tool_calls must NOT be serialized as "tool_calls": []
    assert "tool_calls" not in req["messages"][0]
    # Tool message with content=None must be serialized as ""
    assert req["messages"][1]["content"] == ""
    assert req["messages"][1]["tool_call_id"] == "call_999"


@pytest.mark.asyncio
async def test_network_timeout_and_non_json_response() -> None:
    # 1. Connection timeout wrapping
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("Connection timed out to gateway")

    client_timeout = _make_mock_client(timeout_handler)
    model_timeout = Model(model_name="gpt-4o", client=client_timeout)

    with pytest.raises(ModelError) as exc_info:
        await model_timeout.agenerate([Message(role="user", content="Hi")])
    assert "Network error communicating with" in str(exc_info.value)
    assert "Connection timed out" in str(exc_info.value)

    # 2. 200 OK with non-JSON HTML body
    def html_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html><body>502 Bad Gateway</body></html>")

    client_html = _make_mock_client(html_handler)
    model_html = Model(model_name="gpt-4o", client=client_html)

    with pytest.raises(ModelError) as exc_info:
        await model_html.agenerate([Message(role="user", content="Hi")])
    assert "Failed to parse JSON response from" in str(exc_info.value)


@pytest.mark.asyncio
async def test_short_api_key_does_not_over_redact() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Invalid keyword provided in query")

    # Key is only 3 chars long ("key")
    client = _make_mock_client(handler)
    model = Model(model_name="gpt-4o", api_key="key", client=client)

    with pytest.raises(ModelError) as exc_info:
        await model.agenerate([Message(role="user", content="Hi")])

    # Should NOT over-redact "keyword" into "[REDACTED]word"
    assert "Invalid keyword provided in query" in str(exc_info.value)


