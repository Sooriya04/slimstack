"""Lightweight agent execution loop for SlimStack."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from slimstack.models import Model
from slimstack.tools.decorators import Tool
from slimstack.tools.registry import ToolRegistry
from slimstack.types.engine import ModelResponse
from slimstack.types.messages import Message


class AgentResult(BaseModel):
    """The outcome and conversation history of an agent execution loop."""

    model_config = ConfigDict(frozen=True)

    response: ModelResponse
    messages: tuple[Message, ...]
    iterations: int
    max_iterations_reached: bool = False

    @property
    def message(self) -> Message:
        """The final assistant message."""
        return self.response.message

    @property
    def content(self) -> str | None:
        """The text content of the final assistant message."""
        return self.response.message.content


class Agent:
    """Lightweight autonomous agent execution loop."""

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry | Sequence[Tool | Callable[..., Any]] | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 10,
    ) -> None:
        self.model = model
        if isinstance(tools, ToolRegistry):
            self.tools: ToolRegistry = tools
        elif tools:
            self.tools = ToolRegistry(tools)
        else:
            self.tools = ToolRegistry()

        self.system_prompt = system_prompt
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")
        self.max_iterations = max_iterations

    async def run(
        self,
        input_or_messages: str | Sequence[Message],
        **kwargs: Any,
    ) -> AgentResult:
        """Run the agent loop until the model returns text or max_iterations is reached."""
        if isinstance(input_or_messages, str):
            history: list[Message] = [Message(role="user", content=input_or_messages)]
        else:
            history = list(input_or_messages)

        # Prepend system prompt if provided and not already present
        if self.system_prompt:
            if not history or history[0].role != "system":
                history.insert(0, Message(role="system", content=self.system_prompt))

        if not history:
            raise ValueError("Cannot run agent with empty message history.")

        iteration = 0
        last_response: ModelResponse | None = None

        while iteration < self.max_iterations:
            iteration += 1
            tool_schemas = self.tools.get_schemas() if len(self.tools) > 0 else None

            last_response = await self.model.agenerate(
                history,
                tools=tool_schemas,
                **kwargs,
            )
            history.append(last_response.message)

            # If model returned no tool calls, generation is complete
            if not last_response.message.tool_calls:
                return AgentResult(
                    response=last_response,
                    messages=tuple(history),
                    iterations=iteration,
                    max_iterations_reached=False,
                )

            # Execute all requested tool calls in order
            for tool_call in last_response.message.tool_calls:
                tool_msg = await self.tools.aexecute_to_message(tool_call)
                history.append(tool_msg)

        # Iteration limit reached without terminal text response
        assert last_response is not None
        return AgentResult(
            response=last_response,
            messages=tuple(history),
            iterations=iteration,
            max_iterations_reached=True,
        )


async def run_agent(
    model: Model,
    input_or_messages: str | Sequence[Message],
    tools: ToolRegistry | Sequence[Tool | Callable[..., Any]] | None = None,
    system_prompt: str | None = None,
    max_iterations: int = 10,
    **kwargs: Any,
) -> AgentResult:
    """Convenience helper to run an agent loop in a single asynchronous call."""
    agent = Agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )
    return await agent.run(input_or_messages, **kwargs)
