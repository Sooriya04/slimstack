"""Provider-independent document and chunk contracts for SlimStack."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A raw or preprocessed document."""

    model_config = ConfigDict(frozen=True)

    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A chunked segment of a document, optionally embedded."""

    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
