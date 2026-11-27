# SlimStack SDK Documentation

`slimstack` is a lightweight, zero-bloat Python orchestration library for LLMs and agent systems.

## Modules

- [Message & Tool Contracts (`slimstack.types`)](messages.md) — Immutable, provider-agnostic `Role`, `ToolCall`, and `Message` contracts.
- [Document & Chunk Contracts (`slimstack.types`)](documents.md) — Pure data representations `Document` and `Chunk` for RAG pipelines.
- [Engine I/O Contracts (`slimstack.types`)](engine.md) — Provider-independent execution models `Usage`, `ModelResponse`, and `StreamChunk`.
- [Model Engine (`slimstack.models`)](models.md) — Unified async client for OpenAI-compatible chat completions and streaming.
