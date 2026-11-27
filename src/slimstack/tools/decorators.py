"""Lightweight tool decorator and schema reflection for SlimStack."""

from __future__ import annotations

import functools
import inspect
import re
from typing import (
    Annotated,
    Any,
    Callable,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import Field, create_model
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo


class ToolError(ValueError):
    """Exception raised when tool schema generation or inspection fails."""


def _parse_docstring_params(doc: str) -> dict[str, str]:
    """Parse parameter descriptions from Google or Sphinx docstrings."""
    param_docs: dict[str, str] = {}
    if not doc:
        return param_docs

    # Sphinx style: :param name: description
    for m in re.finditer(r":param\s+([a-zA-Z_]\w*)\s*:\s*(.+)", doc):
        param_docs[m.group(1)] = m.group(2).strip()

    # Google / NumPy style: indented lines under Args: or Parameters:
    in_args = False
    for line in doc.splitlines():
        if re.match(r"^\s*(Args|Parameters):\s*$", line):
            in_args = True
            continue
        if in_args:
            if line and not line.startswith(" ") and not line.startswith("\t"):
                in_args = False
                continue
            m = re.match(r"^\s+([a-zA-Z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.+)$", line)
            if m:
                param_docs[m.group(1)] = m.group(2).strip()

    return param_docs


class Tool:
    """A callable tool wrapper that reflects an OpenAI-compatible JSON schema."""

    def __init__(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        if not callable(func):
            raise ToolError(f"Expected a callable, got {type(func).__name__}")

        functools.update_wrapper(self, func)
        self.func = func
        self.__wrapped__ = func
        self.name = name or func.__name__

        raw_doc = func.__doc__ or ""
        cleaned_doc = inspect.cleandoc(raw_doc)
        self.description = description if description is not None else cleaned_doc

        sig = inspect.signature(func)
        self.is_async: bool = inspect.iscoroutinefunction(func)
        self._pos_only_names: tuple[str, ...] = tuple(
            name
            for name, p in sig.parameters.items()
            if p.kind == inspect.Parameter.POSITIONAL_ONLY
        )

        self._param_model: Any = None
        self._schema = self._build_schema()

    @property
    def schema(self) -> dict[str, Any]:
        """Return the OpenAI-compatible function calling schema."""
        return self._schema

    def validate_arguments(self, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and coerce arguments dictionary using the reflected parameter schema."""
        args_dict = arguments or {}
        if self._param_model is None:
            return args_dict
        validated = self._param_model(**args_dict)
        return validated.model_dump()

    def call_with_kwargs(self, kwargs: dict[str, Any]) -> Any:
        """Call the underlying function, correctly binding positional-only arguments."""
        kw = dict(kwargs)
        args = [kw.pop(name) for name in self._pos_only_names if name in kw]
        return self.func(*args, **kw)

    async def acall_with_kwargs(self, kwargs: dict[str, Any]) -> Any:
        """Call the underlying function asynchronously, supporting both sync and async callables."""
        kw = dict(kwargs)
        args = [kw.pop(name) for name in self._pos_only_names if name in kw]
        if self.is_async:
            return await self.func(*args, **kw)
        return self.func(*args, **kw)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)

    def _build_schema(self) -> dict[str, Any]:
        try:
            sig = inspect.signature(self.func)
            try:
                hints = get_type_hints(self.func, include_extras=True)
            except Exception:
                hints = {}

            doc_params = _parse_docstring_params(self.func.__doc__ or "")
            fields: dict[str, Any] = {}

            for param_name, param in sig.parameters.items():
                if param_name in ("self", "cls"):
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                ann = hints.get(param_name, Any)
                if ann is inspect.Parameter.empty:
                    ann = Any

                desc = doc_params.get(param_name)
                field_info = None

                if get_origin(ann) is Annotated:
                    args = get_args(ann)
                    base_type = args[0]
                    for extra in args[1:]:
                        if isinstance(extra, FieldInfo):
                            field_info = extra
                            break
                        elif isinstance(extra, str):
                            desc = extra
                            break
                    ann = base_type

                if field_info is None:
                    if param.default is inspect.Parameter.empty:
                        field_info = Field(description=desc) if desc else ...
                    else:
                        field_info = Field(default=param.default, description=desc) if desc else param.default

                fields[param_name] = (ann, field_info)

            model_name = f"{self.name.capitalize()}Parameters"
            self._param_model = create_model(model_name, **fields)
            parameters = self._param_model.model_json_schema()

            # Clean up top-level title to keep schema standard
            parameters.pop("title", None)

            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": parameters,
                },
            }
        except (PydanticSchemaGenerationError, TypeError, ValueError) as exc:
            raise ToolError(f"Failed to generate schema for tool '{self.name}': {exc}") from exc
        except Exception as exc:
            raise ToolError(f"Unexpected error inspecting tool '{self.name}': {exc}") from exc


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator to mark a function as a Tool with automatic schema reflection."""
    if func is not None:
        return Tool(func, name=name, description=description)

    def decorator(fn: Callable[..., Any]) -> Tool:
        return Tool(fn, name=name, description=description)

    return decorator
