"""Tests for tool decorator and schema reflection."""

import asyncio
import inspect
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field

from slimstack.tools import Tool, ToolError, tool


def test_normal_function_call_and_no_execution_on_decorating() -> None:
    executed = False

    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        nonlocal executed
        executed = True
        return a * b

    # Schema inspection must NOT trigger execution
    assert not executed
    assert multiply.schema["function"]["name"] == "multiply"
    assert not executed

    # Direct calling executes as expected
    result = multiply(3, 4)
    assert executed
    assert result == 12


def test_schema_structure_and_docstring_cleaning() -> None:
    @tool
    def search_records(query: str, limit: int = 10) -> list[str]:
        """Search records in database.

        This query fetches matching items based on user criteria.

        Args:
            query: The text to search for.
            limit: Maximum count of results.
        """
        return []

    schema = search_records.schema

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search_records"

    # Description is cleaned of leading indentation
    expected_desc = (
        "Search records in database.\n\n"
        "This query fetches matching items based on user criteria.\n\n"
        "Args:\n"
        "    query: The text to search for.\n"
        "    limit: Maximum count of results."
    )
    assert search_records.description == expected_desc
    assert schema["function"]["description"] == expected_desc

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "query" in params["properties"]
    assert "limit" in params["properties"]

    assert params["properties"]["query"]["type"] == "string"
    assert params["properties"]["query"]["description"] == "The text to search for."

    assert params["properties"]["limit"]["type"] == "integer"
    assert params["properties"]["limit"]["default"] == 10
    assert params["properties"]["limit"]["description"] == "Maximum count of results."

    # Required parameters: query is required, limit has default
    assert params["required"] == ["query"]


def test_type_mappings_primitives_and_collections() -> None:
    @tool
    def data_types(
        text: str,
        count: int,
        score: float,
        is_active: bool,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> None:
        pass

    props = data_types.schema["function"]["parameters"]["properties"]
    assert props["text"]["type"] == "string"
    assert props["count"]["type"] == "integer"
    assert props["score"]["type"] == "number"
    assert props["is_active"]["type"] == "boolean"
    assert props["tags"]["type"] == "array"
    assert props["tags"]["items"]["type"] == "string"
    assert props["metadata"]["type"] == "object"


def test_optional_parameters_and_defaults() -> None:
    @tool
    def filter_items(
        category: str,
        prefix: str | None = None,
        offset: int = 0,
        sort: bool = False,
    ) -> None:
        pass

    params = filter_items.schema["function"]["parameters"]
    props = params["properties"]

    assert params["required"] == ["category"]
    assert props["offset"]["default"] == 0
    assert props["sort"]["default"] is False
    assert props["prefix"]["default"] is None


def test_annotated_parameter_descriptions() -> None:
    @tool
    def locate_user(
        ip_addr: Annotated[str, Field(description="The IPv4 or IPv6 address.")],
        mode: Annotated[str, "Resolution speed mode"] = "fast",
    ) -> None:
        pass

    props = locate_user.schema["function"]["parameters"]["properties"]
    assert props["ip_addr"]["description"] == "The IPv4 or IPv6 address."
    assert props["mode"]["description"] == "Resolution speed mode"
    assert props["mode"]["default"] == "fast"


def test_nested_pydantic_models() -> None:
    class Coordinate(BaseModel):
        lat: float
        lng: float

    @tool
    def update_location(coord: Coordinate, notes: list[str]) -> str:
        """Update geographic coordinates."""
        return "updated"

    schema = update_location.schema
    params = schema["function"]["parameters"]

    assert "$defs" in params
    assert "Coordinate" in params["$defs"]
    assert params["properties"]["coord"]["$ref"] == "#/$defs/Coordinate"
    assert params["properties"]["notes"]["type"] == "array"


def test_custom_name_and_description_decorator() -> None:
    @tool(name="custom_calculator", description="Performs custom evaluation.")
    def raw_fn(x: int) -> int:
        """Ignored docstring."""
        return x + 1

    assert raw_fn.name == "custom_calculator"
    assert raw_fn.description == "Performs custom evaluation."
    assert raw_fn.schema["function"]["name"] == "custom_calculator"
    assert raw_fn.schema["function"]["description"] == "Performs custom evaluation."
    assert raw_fn(5) == 6


def test_async_function_support() -> None:
    @tool
    async def fetch_async(endpoint: str) -> str:
        """Async data fetcher."""
        await asyncio.sleep(0.01)
        return f"data from {endpoint}"

    assert fetch_async.schema["function"]["name"] == "fetch_async"
    assert inspect.iscoroutinefunction(fetch_async.func)
    result = asyncio.run(fetch_async("https://api.example.com"))
    assert result == "data from https://api.example.com"


def test_invalid_or_unsupported_annotations() -> None:
    class UnsupportedArbitrary:
        pass

    def bad_function(unsupported: UnsupportedArbitrary) -> None:
        pass

    with pytest.raises((ToolError, ValueError)) as exc_info:
        tool(bad_function)

    assert "Failed to generate schema" in str(exc_info.value)


def test_non_callable_target_raises_error() -> None:
    with pytest.raises(ToolError):
        Tool("not_a_callable")  # type: ignore[arg-type]
