"""Unified model engine for OpenAI-compatible chat completion endpoints."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator, Self

import httpx

from slimstack.types.engine import ModelResponse, StreamChunk, Usage
from slimstack.types.messages import Message, ToolCall


class ModelError(Exception):
    """Exception raised when an error occurs during model communication."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class Model:
    """Unified client for OpenAI-compatible /v1/chat/completions endpoints."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float | httpx.Timeout | None = 60.0,
        *,
        model: str | None = None,
    ) -> None:
        resolved_name = model_name or model
        if not resolved_name:
            raise ValueError("A model name must be provided via 'model_name' or 'model'.")
        self.model_name = resolved_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.timeout = timeout
        self._is_external_client = client is not None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        """Close the underlying HTTP client if it was created internally."""
        if not self._is_external_client and self._client is not None and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()

    def _get_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _sanitize(self, text: str) -> str:
        if self.api_key and len(self.api_key.strip()) > 3 and self.api_key in text:
            return text.replace(self.api_key, "[REDACTED]")
        return text

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        for msg in messages:
            item: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.role == "tool" and msg.content is None:
                item["content"] = ""
            if msg.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            if msg.tool_call_id is not None:
                item["tool_call_id"] = msg.tool_call_id
            serialized.append(item)
        return serialized

    def _parse_usage(self, raw_usage: Any) -> Usage | None:
        if not isinstance(raw_usage, dict):
            return None
        p_tokens = raw_usage.get("prompt_tokens")
        c_tokens = raw_usage.get("completion_tokens")
        t_tokens = raw_usage.get("total_tokens")
        if (
            isinstance(p_tokens, int)
            and isinstance(c_tokens, int)
            and isinstance(t_tokens, int)
            and p_tokens >= 0
            and c_tokens >= 0
            and t_tokens >= 0
        ):
            try:
                return Usage(
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                )
            except Exception:
                return None
        return None

    def _parse_response(self, data: dict[str, Any]) -> ModelResponse:
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise ModelError("Provider response contains no choices.")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise ModelError("Invalid choice structure in provider response.")

        choice_msg = choice.get("message", {})
        if not isinstance(choice_msg, dict):
            raise ModelError("Invalid message structure in provider response.")

        role = choice_msg.get("role", "assistant")
        content = choice_msg.get("content")

        tool_calls: tuple[ToolCall, ...] | None = None
        raw_tool_calls = choice_msg.get("tool_calls")
        if raw_tool_calls and isinstance(raw_tool_calls, list):
            parsed_calls: list[ToolCall] = []
            for tc in raw_tool_calls:
                if not isinstance(tc, dict):
                    continue
                tc_id = tc.get("id") or ""
                fn = tc.get("function", {})
                name = fn.get("name") or "" if isinstance(fn, dict) else ""
                raw_args = fn.get("arguments", "{}") if isinstance(fn, dict) else "{}"
                if isinstance(raw_args, str):
                    try:
                        parsed = json.loads(raw_args)
                        arguments = parsed if isinstance(parsed, dict) else {"raw": parsed}
                    except Exception:
                        arguments = {"raw": raw_args}
                elif isinstance(raw_args, dict):
                    arguments = raw_args
                else:
                    arguments = {"raw": raw_args}
                parsed_calls.append(ToolCall(id=tc_id, name=name, arguments=arguments))
            tool_calls = tuple(parsed_calls) if parsed_calls else None

        message = Message(role=role, content=content, tool_calls=tool_calls)
        usage = self._parse_usage(data.get("usage"))
        return ModelResponse(message=message, usage=usage)

    async def agenerate(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Generate a complete chat completion response."""
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._serialize_messages(messages),
            **kwargs,
        }
        if tools:
            payload["tools"] = tools

        try:
            response = await self._client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            msg = self._sanitize(str(exc))
            raise ModelError(f"Network error communicating with {self.base_url}: {msg}") from None

        if response.is_error:
            detail = self._sanitize(response.text)
            raise ModelError(
                f"HTTP {response.status_code} error from {self.base_url}: {detail}",
                status_code=response.status_code,
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ModelError(f"Failed to parse JSON response from {self.base_url}: {exc}") from None

        return self._parse_response(data)

    async def astream(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completion tokens incrementally as server-sent events."""
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._serialize_messages(messages),
            "stream": True,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools

        try:
            async with self._client.stream("POST", url, headers=headers, json=payload) as response:
                if response.is_error:
                    error_body = await response.aread()
                    detail = self._sanitize(error_body.decode(errors="replace"))
                    raise ModelError(
                        f"HTTP {response.status_code} streaming error from {self.base_url}: {detail}",
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                    else:
                        continue

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                    except Exception:
                        continue

                    if not isinstance(chunk_data, dict):
                        continue

                    choices = chunk_data.get("choices", [])
                    delta_content = ""
                    finish_reason = None
                    if choices and isinstance(choices, list):
                        first_choice = choices[0]
                        if isinstance(first_choice, dict):
                            delta = first_choice.get("delta", {})
                            if isinstance(delta, dict):
                                delta_content = delta.get("content") or ""
                            finish_reason = first_choice.get("finish_reason")

                    chunk_usage = self._parse_usage(chunk_data.get("usage"))

                    yield StreamChunk(
                        delta=delta_content,
                        finish_reason=finish_reason,
                        usage=chunk_usage,
                    )
        except httpx.RequestError as exc:
            msg = self._sanitize(str(exc))
            raise ModelError(f"Network error during streaming from {self.base_url}: {msg}") from None
