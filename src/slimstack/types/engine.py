"""Provider-independent engine I/O contracts for SlimStack."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from slimstack.types.messages import Message


class Usage(BaseModel):
    """Token consumption metrics for a model interaction."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    """Complete output produced by a language model."""

    model_config = ConfigDict(frozen=True)

    message: Message
    usage: Usage | None = None


class StreamChunk(BaseModel):
    """A streaming token or metadata delta produced during model generation."""

    model_config = ConfigDict(frozen=True)

    delta: str
    finish_reason: str | None = None
    usage: Usage | None = None
