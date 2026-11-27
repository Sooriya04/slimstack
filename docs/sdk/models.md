# Model Engine (`slimstack.models`)

A unified, lightweight client for interacting with any OpenAI-compatible `/v1/chat/completions` endpoint (OpenAI, Ollama, vLLM, LMStudio, Groq, etc.) using `httpx.AsyncClient`.

---

## 1. Quickstart & Command-Line Usage

### Complete Chat Completion
```python
import asyncio
from slimstack import Model
from slimstack.types import Message

async def main():
    async with Model(model_name="gpt-4o", api_key="sk-...") as model:
        response = await model.agenerate([
            Message(role="user", content="Explain quantum computing in one sentence.")
        ])
        print("Response:", response.message.content)
        if response.usage:
            print("Total tokens:", response.usage.total_tokens)

asyncio.run(main())
```

### Local Endpoint (Ollama / vLLM without API key)
```python
model = Model(
    model_name="llama3.2",
    base_url="http://localhost:11434/v1",
)
```

---

## 2. API Reference

### `Model(model_name, base_url="https://api.openai.com/v1", api_key=None, client=None, timeout=60.0)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | *Required* | Model identifier (e.g. `"gpt-4o"`, `"llama3.2"`). |
| `base_url` | `str` | `"https://api.openai.com/v1"` | Base URL of the OpenAI-compatible API (trailing slashes auto-stripped). |
| `api_key` | `str \| None` | `None` | API key. If omitted, falls back to `OPENAI_API_KEY` environment variable. |
| `client` | `httpx.AsyncClient \| None` | `None` | Optional existing client. If provided, `Model` will never close it automatically. |
| `timeout` | `float \| httpx.Timeout` | `60.0` | Request timeout for internal HTTP requests. |

---

## 3. Methods

### `agenerate(messages, tools=None, **kwargs) -> ModelResponse`
Sends a POST request to `{base_url}/chat/completions` and returns a typed `ModelResponse`.

- **`messages`**: `list[Message]` — List of SlimStack `Message` objects.
- **`tools`**: `list[dict] | None` — Optional OpenAI function tool definitions.
- **`**kwargs`**: Extra payload parameters (e.g. `temperature=0.7`, `max_tokens=500`).

### `astream(messages, tools=None, **kwargs) -> AsyncIterator[StreamChunk]`
Asynchronous generator streaming text tokens and metadata chunks incrementally via Server-Sent Events (SSE).

```python
async for chunk in model.astream([Message(role="user", content="Write a haiku.")]):
    print(chunk.delta, end="", flush=True)
    if chunk.finish_reason:
        print(f"\n[Finished: {chunk.finish_reason}]")
```

---

## 4. Key Features & Design Details

1. **Zero Metaclass / Inheritance Bloat:** A single clean `Model` class without complex inheritance trees or abstract wrappers.
2. **Safe Secrets & Error Handling:** When network or HTTP errors occur, `ModelError` is raised with the HTTP status and provider response message, while automatically redacting API keys and avoiding Authorization header leaks.
3. **Client Lifecycle Control:** When using `async with Model(...)` or `await model.aclose()`, internally managed `httpx.AsyncClient` instances are closed cleanly. User-provided clients are left untouched.
