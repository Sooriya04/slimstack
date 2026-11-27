"""Smoke tests for the slimstack package exports."""

import slimstack


def test_version_and_exports() -> None:
    assert slimstack.__version__ == "0.0.1"
    assert "Model" in slimstack.__all__
    assert "ModelError" in slimstack.__all__
    assert "Tool" in slimstack.__all__
    assert "ToolError" in slimstack.__all__
    assert "ToolRegistry" in slimstack.__all__
    assert "ToolResult" in slimstack.__all__
    assert "Agent" in slimstack.__all__
    assert "AgentResult" in slimstack.__all__
    assert "run_agent" in slimstack.__all__
    assert "tool" in slimstack.__all__
    assert "__version__" in slimstack.__all__
    assert slimstack.Model is not None
    assert slimstack.Agent is not None
    assert slimstack.run_agent is not None
    assert slimstack.tool is not None
    assert slimstack.ToolRegistry is not None
    assert issubclass(slimstack.ModelError, Exception)
    assert issubclass(slimstack.ToolError, ValueError)
