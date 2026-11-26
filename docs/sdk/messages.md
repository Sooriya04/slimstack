# Message & Tool Contracts (`slimstack.types`)

Provider-independent data contracts for conversation histories and function-calling workflows.

---

## 1. Quickstart & Command-Line Usage

You can test these primitives directly in your terminal using Python or `uv`:

### Interactive One-Liner
```bash
uv run python -c '
from slimstack.types import Message, ToolCall

tc = ToolCall(id="call_01", name="search", arguments={"q": "slimstack"})
msg = Message(role="assistant", tool_calls=[tc])

print("Role:", msg.role)
print("Tool Call Name:", msg.tool_calls[0].name)
print("JSON Payload:", msg.model_dump_json(exclude_none=True))
'
```

### Output:
```json
Role: assistant
Tool Call Name: search
JSON Payload: {"role":"assistant","tool_calls":[{"id":"call_01","name":"search","arguments":{"q":"slimstack"}}]}
```

---

## 2. API Reference

### `Role`
A static type literal representing allowed conversation participants:
```python
Role = Literal["system", "user", "assistant", "tool"]
```

### `ToolCall`
A frozen Pydantic model representing a tool invocation emitted by a model.

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `str` | Yes | Unique ID identifying the tool execution. |
| `name` | `str` | Yes | Name of the function/tool to run. |
| `arguments` | `dict[str, Any]` | Yes | JSON-compatible dictionary of arguments. |

### `Message`
A frozen Pydantic model representing a single conversation turn.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `role` | `Role` | Yes | — | One of `"system"`, `"user"`, `"assistant"`, `"tool"`. |
| `content` | `str \| None` | No | `None` | Text content of the message. |
| `tool_calls` | `tuple[ToolCall, ...] \| None` | No | `None` | Sequence of tool invocation requests. |
| `tool_call_id` | `str \| None` | No | `None` | Reference ID for `"tool"` results. |

---

## 3. Practical Use Cases

### A. Standard Chat Turns
```python
from slimstack.types import Message

system = Message(role="system", content="You are a helpful assistant.")
user = Message(role="user", content="What is the capital of France?")
assistant = Message(role="assistant", content="The capital of France is Paris.")
```

### B. Emitting Tool Calls
When an assistant decides to invoke one or more tools:
```python
from slimstack.types import Message, ToolCall

tc1 = ToolCall(id="call_w1", name="get_weather", arguments={"city": "Paris"})
tc2 = ToolCall(id="call_t1", name="get_time", arguments={"timezone": "Europe/Paris"})

# content is optional (None for pure tool calls, or str for reasoning / thought)
action = Message(
    role="assistant",
    content=None,
    tool_calls=[tc1, tc2],
)
```

### C. Returning Tool Results
Feeding the tool execution output back into the conversation:
```python
result = Message(
    role="tool",
    content='{"temperature_c": 19, "condition": "Partly Cloudy"}',
    tool_call_id="call_w1",
)
```

### D. Provider Payload Formatting
Generate clean request payloads for any OpenAI-compatible `/v1/chat/completions` endpoint:
```python
conversation = [system, user, action, result]
payload = [msg.model_dump(exclude_none=True) for msg in conversation]
```

### E. Immutability & Concurrency Safety
All models use Pydantic's `frozen=True` configuration. Field reassignments raise `ValidationError`:
```python
msg = Message(role="user", content="Initial")

try:
    msg.content = "Changed"
except Exception as e:
    # Raises pydantic.ValidationError: Instance is frozen
    pass
```
Additionally, `tool_calls` are automatically stored as `tuple`, preventing in-place sequence mutations (`.append` or index reassignments).
