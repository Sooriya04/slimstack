# AGENTS.md — slimstack Engineering Guide

## Mission
`slimstack` is a standalone, zero-bloat Python library replacing LangChain's core orchestration primitives. Keep the codebase small, explicit, strictly typed, and free of framework magic.

## Hard Constraints
1. **Dependencies (Core Release):** Only `httpx>=0.28.1` and `pydantic>=2.14.0` on Python `>=3.11` (note: `0.0.1` is an empty placeholder release with zero dependencies). Never add `langchain*`, database drivers, telemetry servers, or cloud vendor SDKs to the core package.
2. **Flat Public API:** Export all public primitives directly from `slimstack` (`__init__.py`):
   `Runnable`, `Sequence`, `Parallel`, `Lambda`, `Prompt`, `ChatModel`, `StrParser`, `JsonParser`, `Tool`, `tool`, `TraceContext`, `RunEvent`.
   Do not bury primitives in deep nested subpackages.
3. **Code Budget & Style:** Keep the entire `src/slimstack/` implementation compact (~400–600 LOC total across modules), strictly type-annotated with `py.typed`, and free of metaclass hacks.
4. **Zero Network I/O in Telemetry:** `callbacks.py` emits in-memory `RunEvent` objects via `contextvars` to registered local callbacks only. External trace shippers plug in via `register_callback(fn)` outside core.

## Planned Module Layout
- `core.py` — `Runnable[Input, Output]` ABC (`invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`), `|` and callable `__ror__` flattening into `Sequence`, `Parallel` (`ThreadPoolExecutor` / `asyncio.gather`), `Lambda`, `.with_retry()`, `.with_fallbacks()`.
- `prompts.py` — `Prompt` formatter outputting OpenAI-compatible `list[dict[str, str]]`, clear `KeyError` on missing variables, `.partial(**kwargs)`.
- `models.py` — `ChatModel` for OpenAI-compatible `/v1/chat/completions` (default `http://localhost:11434/v1`), sync/async + SSE streaming, `.bind_tools()`.
- `parsers.py` — `StrParser` and `JsonParser[T]` (strips markdown code fences safely, optional Pydantic validation, raises `OutputParseError` preserving raw text).
- `tools.py` — `@tool` decorator and `Tool` (`Runnable` subclass) generating OpenAI function schemas via Pydantic `TypeAdapter` and validating inputs on `invoke`.
- `callbacks.py` — `contextvars`-based `TraceContext` (`trace_id`, `parent_run_id`, `dotted_order`) and Pydantic `RunEvent` emission to `register_callback(fn)`.

## Testing Rules
- All unit tests live in `tests/` (`test_core.py`, `test_prompts_parsers.py`, `test_models.py`, `test_tools_callbacks.py`).
- Use `httpx.MockTransport` for all HTTP/SSE tests—the test suite must pass offline with no live LLM server.
- Run `pytest` and verify every test passes before finishing any change.
