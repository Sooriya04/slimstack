# slimstack

**The unbloated AI stack.** Minimal, zero-bloat inference and agent orchestration primitives for Python.

> **Status:** `0.0.1` — Initial package reservation. Core orchestration primitives are under active development.

## Why slimstack?

Most LLM frameworks bury simple composition under deep class hierarchies, heavy transitive dependencies, and opaque abstractions. `slimstack` is designed to give you the useful primitives—composable runnables, prompts, OpenAI-compatible model clients, structured parsers, schema-typed tools, and local trace hooks—in a few hundred lines of readable, strictly typed Python.

## Planned Design

- **Minimal footprint:** Only `httpx` and `pydantic` (no `langchain*` packages, no vendor SDK bloat, no mandatory telemetry servers).
- **Flat public API:** Everything imported straight from `slimstack` (`Runnable`, `Sequence`, `Parallel`, `Lambda`, `Prompt`, `ChatModel`, `StrParser`, `JsonParser`, `Tool`, `tool`, `TraceContext`, `RunEvent`).
- **Sync, async, and streaming:** First-class `invoke` / `ainvoke` / `stream` / `astream` / `batch` across pipelines.
- **OpenAI-compatible by default:** Works out of the box with local servers (Ollama, `llama.cpp`, vLLM) and hosted OpenAI-compatible endpoints.

## Installation

```bash
pip install slimstack
```

## License

MIT — see [LICENSE](LICENSE).
