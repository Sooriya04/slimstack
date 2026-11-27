"""Smoke tests for the slimstack package exports."""

import slimstack


def test_version_and_exports() -> None:
    assert slimstack.__version__ == "0.0.1"
    assert "Model" in slimstack.__all__
    assert "ModelError" in slimstack.__all__
    assert "__version__" in slimstack.__all__
    assert slimstack.Model is not None
    assert issubclass(slimstack.ModelError, Exception)
