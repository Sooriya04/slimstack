"""Tests for Agent execution loop using mocked model and tools."""

import json
from typing import Any

import httpx
import pytest

from slimstack.agents import Agent, AgentResult, run_agent
from slimstack.models import Model
from slimstack.tools import ToolRegistry, tool
from slimstack.types import Message


def _make_mock_model(responses: list[dict[str, Any]]) -> Model:
    """Helper to produce a Model backed by a sequence of mock JSON responses."""
    resp_iter = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            body = next(resp_iter)
        except StopIteration:
            pytest.fail("Mock model received more requests than expected.")
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    return Model(model_name="test-model", client=client)


@pytest.mark.asyncio
async def test_agent_ordinary_text_response() -> None:
    mock_responses = [
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "I am here to help!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        }
    ]
    model = _make_mock_model(mock_responses)
    agent = Agent(model=model)

    result = await agent.run("Hello")
    assert isinstance(result, AgentResult)
    assert result.content == "I am here to help!"
    assert result.iterations == 1
    assert result.max_iterations_reached is False
    assert len(result.messages) == 2
    assert result.messages[0].role == "user"
    assert result.messages[0].content == "Hello"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].content == "I am here to help!"


@pytest.mark.asyncio
async def test_agent_with_system_prompt() -> None:
    mock_responses = [
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "System prompt acknowledged."},
                }
            ]
        }
    ]
    model = _make_mock_model(mock_responses)
    agent = Agent(model=model, system_prompt="You are a helpful assistant.")

    result = await agent.run("Hi")
    assert result.iterations == 1
    assert len(result.messages) == 3
    assert result.messages[0].role == "system"
    assert result.messages[0].content == "You are a helpful assistant."
    assert result.messages[1].role == "user"
    assert result.messages[1].content == "Hi"
    assert result.messages[2].content == "System prompt acknowledged."


@pytest.mark.asyncio
async def test_agent_single_tool_call_and_resolution() -> None:
    mock_responses = [
        # Turn 1: Model requests tool call
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_w_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": json.dumps({"city": "Berlin"}),
                                },
                            }
                        ],
                    }
                }
            ]
        },
        # Turn 2: Model returns final response after receiving tool result
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The weather in Berlin is 18°C.",
                    }
                }
            ]
        },
    ]

    @tool
    def get_weather(city: str) -> str:
        """Get weather for city."""
        return f"Weather in {city}: 18°C"

    model = _make_mock_model(mock_responses)
    agent = Agent(model=model, tools=[get_weather])

    result = await agent.run("What is the weather in Berlin?")
    assert result.iterations == 2
    assert result.content == "The weather in Berlin is 18°C."
    assert result.max_iterations_reached is False

    # Check conversation history ordering
    assert len(result.messages) == 4
    assert result.messages[0].role == "user"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].tool_calls is not None
    assert result.messages[1].tool_calls[0].name == "get_weather"

    # Tool result message
    assert result.messages[2].role == "tool"
    assert result.messages[2].tool_call_id == "call_w_1"
    assert "18°C" in (result.messages[2].content or "")

    # Final assistant message
    assert result.messages[3].role == "assistant"
    assert result.messages[3].content == "The weather in Berlin is 18°C."


@pytest.mark.asyncio
async def test_agent_multiple_tool_calls_in_single_turn() -> None:
    mock_responses = [
        # Turn 1: Model requests two tool calls at once
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "fetch_user", "arguments": json.dumps({"user_id": 42})},
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "fetch_role", "arguments": json.dumps({"user_id": 42})},
                            },
                        ],
                    }
                }
            ]
        },
        # Turn 2: Final response
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "User 42 is Alice with role Admin.",
                    }
                }
            ]
        },
    ]

    @tool
    def fetch_user(user_id: int) -> str:
        return "Alice"

    @tool
    def fetch_role(user_id: int) -> str:
        return "Admin"

    model = _make_mock_model(mock_responses)
    agent = Agent(model=model, tools=[fetch_user, fetch_role])

    result = await agent.run("Lookup user 42")
    assert result.iterations == 2
    assert result.content == "User 42 is Alice with role Admin."

    # History: [user, assistant_with_2_calls, tool_1, tool_2, final_assistant]
    assert len(result.messages) == 5
    assert result.messages[2].role == "tool"
    assert result.messages[2].tool_call_id == "call_1"
    assert result.messages[2].content == "Alice"

    assert result.messages[3].role == "tool"
    assert result.messages[3].tool_call_id == "call_2"
    assert result.messages[3].content == "Admin"


@pytest.mark.asyncio
async def test_agent_multiple_consecutive_tool_turns() -> None:
    mock_responses = [
        # Turn 1: Tool A
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "step_one", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        # Turn 2: Tool B
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c2",
                                "type": "function",
                                "function": {"name": "step_two", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
        # Turn 3: Final response
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Both steps completed."},
                }
            ]
        },
    ]

    @tool
    def step_one() -> str:
        return "result_one"

    @tool
    def step_two() -> str:
        return "result_two"

    model = _make_mock_model(mock_responses)
    agent = Agent(model=model, tools=[step_one, step_two])

    result = await agent.run("Execute sequential workflow")
    assert result.iterations == 3
    assert result.content == "Both steps completed."
    # [user, ast_1, tool_1, ast_2, tool_2, final_ast] = 6 messages
    assert len(result.messages) == 6


@pytest.mark.asyncio
async def test_agent_tool_failure_recovery() -> None:
    mock_responses = [
        # Turn 1: Calls tool with bad args / causing crash
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c_fail",
                                "type": "function",
                                "function": {"name": "faulty_tool", "arguments": json.dumps({"n": 0})},
                            }
                        ],
                    }
                }
            ]
        },
        # Turn 2: Model receives tool error and explains it gracefully
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The calculation failed due to division by zero.",
                    }
                }
            ]
        },
    ]

    @tool
    def faulty_tool(n: int) -> float:
        return 100 / n

    model = _make_mock_model(mock_responses)
    agent = Agent(model=model, tools=[faulty_tool])

    result = await agent.run("Calculate 100 / 0")
    assert result.iterations == 2
    assert "division by zero" in (result.content or "")

    # Tool error was captured in message history without crashing the agent
    tool_msg = result.messages[2]
    assert tool_msg.role == "tool"
    assert "division by zero" in (tool_msg.content or "")


@pytest.mark.asyncio
async def test_agent_preserves_caller_message_list() -> None:
    mock_responses = [
        {"choices": [{"message": {"role": "assistant", "content": "Acknowledged."}}]}
    ]
    model = _make_mock_model(mock_responses)
    agent = Agent(model=model)

    user_messages = [Message(role="user", content="Original input")]
    result = await agent.run(user_messages)

    # Caller list is NOT mutated
    assert len(user_messages) == 1
    assert user_messages[0].content == "Original input"

    # AgentResult contains the appended conversation
    assert len(result.messages) == 2


@pytest.mark.asyncio
async def test_agent_max_iterations_limit() -> None:
    # Model generates infinite tool calls
    infinite_tool_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c_loop",
                            "type": "function",
                            "function": {"name": "ping", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }
    model = _make_mock_model([infinite_tool_response] * 5)

    @tool
    def ping() -> str:
        return "pong"

    # Set max_iterations to 3
    agent = Agent(model=model, tools=[ping], max_iterations=3)
    result = await agent.run("Loop forever")

    assert result.iterations == 3
    assert result.max_iterations_reached is True
    # The last assistant message was a tool call
    assert result.message.tool_calls is not None


@pytest.mark.asyncio
async def test_run_agent_convenience_helper() -> None:
    mock_responses = [
        {"choices": [{"message": {"role": "assistant", "content": "Done directly."}}]}
    ]
    model = _make_mock_model(mock_responses)

    result = await run_agent(model, "Direct prompt")
    assert result.content == "Done directly."
    assert result.iterations == 1
