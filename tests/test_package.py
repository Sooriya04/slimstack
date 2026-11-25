"""Smoke tests for the slimstack 0.0.1 placeholder package."""

import slimstack


def test_version_and_exports() -> None:
    assert slimstack.__version__ == "0.0.1"
    assert slimstack.__all__ == ["__version__"]
