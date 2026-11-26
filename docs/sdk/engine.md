# Engine I/O Contracts (`slimstack.types`)

Provider-independent data contracts for language model responses, token usage telemetry, and streaming chunks.

---

## 1. Quickstart & Command-Line Usage

Test engine I/O contracts directly in your terminal:

```bash
uv run python -c '
from slimstack.types import Message, ModelResponse, StreamChunk, Usage

usage = Usage(prompt_tokens=10, completion_tokens=25, total_tokens=35)
msg = Message(role="assistant", content="Hello! How can I help?")
response = ModelResponse(message=msg, usage=usage)

chunk = StreamChunk(delta="world", finish_reason=None)

print("Total tokens:", response.usage.total_tokens)
print("Response JSON:", response.model_dump_json(exclude_none=True))
print("Stream chunk delta:", repr(chunk.delta))
'
```

---

## 2. API Reference

### `Usage`
Token consumption metrics for a model interaction. All fields validate non-negative integers (`ge=0`).

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt_tokens` | `int` | Yes | Count of tokens consumed by the prompt input (`>= 0`). |
| `completion_tokens` | `int` | Yes | Count of tokens generated in the response (`>= 0`). |
| `total_tokens` | `int` | Yes | Combined count of tokens for the interaction (`>= 0`). |

### `ModelResponse`
A complete output returned by a language model.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `message` | `Message` | Yes | — | The resulting conversation turn from the model. |
| `usage` | `Usage \| None` | No | `None` | Token consumption metrics (if reported by provider). |

### `StreamChunk`
An individual delta or event emitted during a streaming LLM response.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `delta` | `str` | Yes | — | Incremental text content (can be empty `""` for final/metadata chunks). |
| `finish_reason` | `str \| None` | No | `None` | Unrestricted provider finish reason (`"stop"`, `"length"`, `"tool_calls"`, etc.). |
| `usage` | `Usage \| None` | No | `None` | Final token metrics if emitted at stream termination. |

---

## 3. Practical Use Cases

### A. Non-Streaming Model Response
```python
from slimstack.types import Message, ModelResponse, Usage

resp = ModelResponse(
    message=Message(role="assistant", content="Task completed successfully."),
    usage=Usage(prompt_tokens=120, completion_tokens=8, total_tokens=128),
)
```

### B. Streaming Response Processing
Handling incremental token chunks, followed by a final termination chunk with usage metrics:
```python
from slimstack.types import StreamChunk, Usage

# Incremental text tokens:
c1 = StreamChunk(delta="Calculating")
c2 = StreamChunk(delta=" results...")

# Final termination chunk (empty delta, finish reason, and usage):
final_chunk = StreamChunk(
    delta="",
    finish_reason="stop",
    usage=Usage(prompt_tokens=15, completion_tokens=3, total_tokens=18),
)
```

### C. Concurrency Safety & Immutability
`Usage`, `ModelResponse`, and `StreamChunk` are all configured with `frozen=True`. Attempting to modify attributes in-place will raise `pydantic.ValidationError`:
```python
resp = ModelResponse(message=Message(role="assistant", content="Test"))

try:
    resp.message = Message(role="user", content="Tampered")
except Exception as e:
    # Raises ValidationError: Instance is frozen
    pass
```
