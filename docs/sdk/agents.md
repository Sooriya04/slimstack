# Agent Execution Loop (`slimstack.agents`)

The `slimstack.agents` module provides the autonomous execution loop connecting `Model` generation, `Message` and `ToolCall` contracts, and `ToolRegistry` execution.

---

## 1. Quickstart & Command-Line Usage

```bash
uv run python -c '
import asyncio
from slimstack import Agent, Model, ToolRegistry, tool

# 1. Define tools
registry = ToolRegistry()

@registry.register
def get_weather(location: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {location}: Sunny, 22°C"

# 2. Instantiate Model and Agent
model = Model(model="gpt-4o-mini", api_key="dummy-key")
agent = Agent(
    model=model,
    tools=registry,
    system="You are a helpful assistant with access to real-time tools.",
    max_iterations=5,
)

# 3. Run the agent (async)
async def main():
    result = await agent.run("What is the weather in Paris?")
    print("Response:", result.content)
    print("Iterations:", result.iterations)
    print("Total Messages:", len(result.messages))

# asyncio.run(main())
'
```

---

## 2. API Reference

### `Agent(model, tools=None, system=None, max_iterations=10)`
The core autonomous agent coordinator.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `Model` | Required | Model instance used for chat completion generation. |
| `tools` | `ToolRegistry \| Sequence[Tool \| Callable] \| None` | `None` | Tool registry or collection of tools. |
| `system` | `str \| None` | `None` | Optional system instruction prepended to conversation history. |
| `max_iterations` | `int` | `10` | Maximum reasoning/tool execution turns allowed. Must be $\ge 1$. |

#### Methods

- **`run(input_or_messages: str | Sequence[Message]) -> AgentResult`**
  Executes the agent loop until the model outputs a final text response without tool calls or reaches `max_iterations`.

### `run_agent(model, input_or_messages, *, tools=None, system=None, max_iterations=10) -> AgentResult`
Convenience helper function that instantiates an `Agent` and executes `run(...)` in a single call.

### `AgentResult`
A frozen Pydantic model representing the completed agent execution:

| Field | Type | Description |
|---|---|---|
| `response` | `Message` | Final assistant response returned by the model. |
| `messages` | `tuple[Message, ...]` | Complete, immutable history of messages exchanged during the run. |
| `iterations` | `int` | Number of model generation turns executed. |
| `max_iterations_reached` | `bool` | `True` if execution halted because `max_iterations` was hit. |
| `content` | `str` | Property shortcut accessing `self.response.content or ""`. |

---

## 3. Loop Execution & Safety Guarantees

1. **Non-Mutating History:** Caller message lists are copied and never mutated in place.
2. **Multi-Tool Support:** Multiple parallel tool calls within a single assistant message are executed in sequence, generating matching `role="tool"` responses referencing each corresponding `tool_call_id`.
3. **Graceful Tool Failure Handling:** Tool execution errors (validation failures or caught exceptions) return structured error messages back to the model rather than crashing the loop, enabling the model to recover or self-correct.
4. **Infinite Loop Protection:** Bounded strictly by `max_iterations`. When the limit is reached without a terminal text response, `max_iterations_reached` is set to `True` and the latest assistant message is returned.
