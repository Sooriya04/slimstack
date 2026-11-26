# Document & Chunk Contracts (`slimstack.types`)

Provider-independent data contracts for document ingestion and text chunking in retrieval pipelines.

---

## 1. Quickstart & Command-Line Usage

Test these models directly in your terminal:

```bash
uv run python -c '
from slimstack.types import Document, Chunk

doc = Document(content="Raw report content", metadata={"source": "report.pdf"})
chunk = Chunk(id="chk_01", doc_id="doc_01", content="Chunk snippet", embedding=[0.1, 0.2, 0.3])

print("Doc metadata:", doc.metadata)
print("Chunk embedding dim:", len(chunk.embedding))
print("Chunk JSON:", chunk.model_dump_json(exclude_none=True))
'
```

---

## 2. API Reference

### `Document`
A frozen Pydantic model representing a raw or preprocessed document.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `content` | `str` | Yes | — | The document text content. |
| `metadata` | `dict[str, Any]` | No | `{}` | Key-value metadata (source, title, etc.). |

### `Chunk`
A frozen Pydantic model representing a chunked portion of a document, optionally embedded.

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `id` | `str` | Yes | — | Unique identifier for this chunk. |
| `doc_id` | `str` | Yes | — | Identifier of the parent document. |
| `content` | `str` | Yes | — | Text content of the chunk. |
| `metadata` | `dict[str, Any]` | No | `{}` | Chunk-specific metadata (page number, character offset, etc.). |
| `embedding` | `list[float] \| None` | No | `None` | Optional dense vector embedding. |

---

## 3. Practical Use Cases

### A. Document Representation
```python
from slimstack.types import Document

doc = Document(
    content="Production deployment checklist for Kubernetes cluster...",
    metadata={"filename": "deploy.md", "environment": "production"},
)
```

### B. Chunking & Embedding Storage
```python
from slimstack.types import Chunk

chunk = Chunk(
    id="chunk_42",
    doc_id="doc_deploy_01",
    content="Step 1: Check rolling update status using kubectl rollout status...",
    metadata={"section": "Verification", "step": 1},
    embedding=[0.024, -0.015, 0.892, -0.312],
)
```

### C. Immutability & Architectural Notes
Both `Document` and `Chunk` are configured with `frozen=True`. Reassigning top-level attributes raises `pydantic.ValidationError`:
```python
doc = Document(content="Immutable text")

try:
    doc.content = "New text"
except Exception as e:
    # Raises ValidationError: Instance is frozen
    pass
```

> **Design Limitation Note:** Pydantic's `frozen=True` protects model attributes against reassignment (`doc.metadata = {}` is blocked). However, native Python `dict` and `list` structures remain internally mutable by reference (`doc.metadata["new_key"] = val` or `chunk.embedding.append(0.5)`). Custom deep-freezing layers are omitted to preserve standard JSON compatibility and zero-overhead serialization.
