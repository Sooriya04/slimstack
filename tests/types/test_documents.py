"""Unit tests for Document and Chunk contracts in slimstack.types."""

import pytest
from pydantic import ValidationError

from slimstack.types import Chunk, Document


def test_valid_document_creation() -> None:
    """Verify standard document creation with content and explicit metadata."""
    doc = Document(content="System logs entry", metadata={"source": "var/log/syslog", "line": 42})
    assert doc.content == "System logs entry"
    assert doc.metadata == {"source": "var/log/syslog", "line": 42}


def test_document_default_metadata() -> None:
    """Verify document metadata defaults to an empty dictionary."""
    doc = Document(content="Plain text document")
    assert doc.content == "Plain text document"
    assert doc.metadata == {}


def test_document_independent_metadata_instances() -> None:
    """Verify that default metadata dictionaries are distinct instances across documents."""
    doc1 = Document(content="Doc 1")
    doc2 = Document(content="Doc 2")

    assert doc1.metadata is not doc2.metadata
    doc1.metadata["tag"] = "important"
    assert "tag" not in doc2.metadata


def test_valid_chunk_creation() -> None:
    """Verify standard chunk creation with all fields populated."""
    embedding = [0.12, -0.45, 0.88]
    chunk = Chunk(
        id="chk_001",
        doc_id="doc_abc",
        content="Sub-section of document content",
        metadata={"start_char": 0, "end_char": 31},
        embedding=embedding,
    )
    assert chunk.id == "chk_001"
    assert chunk.doc_id == "doc_abc"
    assert chunk.content == "Sub-section of document content"
    assert chunk.metadata == {"start_char": 0, "end_char": 31}
    assert chunk.embedding == embedding


def test_chunk_default_metadata_and_optional_embedding() -> None:
    """Verify chunk metadata defaults to empty dict and embedding defaults to None."""
    chunk = Chunk(id="chk_002", doc_id="doc_abc", content="Another chunk")
    assert chunk.metadata == {}
    assert chunk.embedding is None


def test_chunk_independent_metadata_instances() -> None:
    """Verify chunk metadata dictionaries are independently instantiated."""
    chunk1 = Chunk(id="c1", doc_id="d1", content="Text 1")
    chunk2 = Chunk(id="c2", doc_id="d1", content="Text 2")

    assert chunk1.metadata is not chunk2.metadata
    chunk1.metadata["chunk_index"] = 0
    assert "chunk_index" not in chunk2.metadata


def test_invalid_field_types_on_document() -> None:
    """Verify validation errors for incorrect Document field types or missing fields."""
    # Missing content
    with pytest.raises(ValidationError):
        Document()  # type: ignore[call-arg]

    # Non-string content
    with pytest.raises(ValidationError):
        Document(content=123)  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Document(content=["hello"])  # type: ignore[arg-type]

    # Non-dict metadata
    with pytest.raises(ValidationError):
        Document(content="text", metadata="not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Document(content="text", metadata=[("key", "val")])  # type: ignore[arg-type]


def test_invalid_field_types_on_chunk() -> None:
    """Verify validation errors for incorrect Chunk field types or missing fields."""
    # Missing required fields
    with pytest.raises(ValidationError):
        Chunk(doc_id="d1", content="text")  # missing id # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        Chunk(id="c1", content="text")  # missing doc_id # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        Chunk(id="c1", doc_id="d1")  # missing content # type: ignore[call-arg]

    # Non-string id/doc_id/content
    with pytest.raises(ValidationError):
        Chunk(id=123, doc_id="d1", content="text")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Chunk(id="c1", doc_id=123, content="text")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Chunk(id="c1", doc_id="d1", content=123)  # type: ignore[arg-type]

    # Non-dict metadata
    with pytest.raises(ValidationError):
        Chunk(id="c1", doc_id="d1", content="text", metadata=123)  # type: ignore[arg-type]

    # Invalid embedding: non-list or invalid element type
    with pytest.raises(ValidationError):
        Chunk(id="c1", doc_id="d1", content="text", embedding="invalid")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        Chunk(id="c1", doc_id="d1", content="text", embedding=[0.1, "not-a-float"])  # type: ignore[list-item]


def test_frozen_immutability_attribute_reassignment() -> None:
    """Verify frozen models block attribute reassignment."""
    doc = Document(content="Original", metadata={"key": "val"})
    chunk = Chunk(id="c1", doc_id="d1", content="Chunk text", embedding=[0.1, 0.2])

    with pytest.raises(ValidationError):
        doc.content = "New"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        doc.metadata = {}  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.id = "c2"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.doc_id = "d2"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.content = "New text"  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.metadata = {}  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.embedding = [0.3, 0.4]  # type: ignore[misc]


def test_nested_mutability_limitation_documentation() -> None:
    """Document and verify that Pydantic frozen=True protects model attributes, but does

    not enforce deep immutability on nested mutable structures (dict and list).
    """
    doc = Document(content="Doc", metadata={"author": "Alice"})
    chunk = Chunk(id="c1", doc_id="d1", content="Text", embedding=[0.1, 0.2])

    # Reassignment is forbidden:
    with pytest.raises(ValidationError):
        doc.metadata = {"author": "Bob"}  # type: ignore[misc]

    with pytest.raises(ValidationError):
        chunk.embedding = [0.3]  # type: ignore[misc]

    # Limitation: Nested dict items can be mutated in-place
    doc.metadata["author"] = "Bob"
    assert doc.metadata["author"] == "Bob"

    # Limitation: Nested list elements can be appended/mutated in-place
    assert chunk.embedding is not None
    chunk.embedding.append(0.3)
    assert chunk.embedding == [0.1, 0.2, 0.3]
