"""Tests for the LLM-facing tool layer."""

import json

import pytest

pytest.importorskip("gmsh")

from tet10.tools import TOOLS, execute_tool


def test_tool_schemas_are_wellformed():
    names = [t["function"]["name"] for t in TOOLS]
    assert names == ["mesh_box", "run_fea"]
    for tool in TOOLS:
        fn = tool["function"]
        assert tool["type"] == "function"
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        for req in params["required"]:
            assert req in params["properties"]


def test_execute_tool_returns_json_error_for_unknown_tool():
    result = json.loads(execute_tool("teleport", {}))
    assert "error" in result
    assert "mesh_box" in result["error"]  # tells the model what IS available


def test_execute_tool_mesh_and_solve_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # results.vtu lands here
    meshed = json.loads(execute_tool("mesh_box", {
        "length": 0.1, "width": 0.02, "height": 0.02,
        "element_size": 0.01, "output_path": str(tmp_path / "box.msh")}))
    assert "error" not in meshed

    solved = json.loads(execute_tool("run_fea", {
        "mesh_file": meshed["mesh_file"],
        "material": {"E": 210e9, "nu": 0.3},
        "bcs": [{"surface": "xmin", "type": "fixed"}],
        "loads": [{"surface": "xmax", "type": "traction",
                   "vector": [1e6, 0, 0]}]}))
    assert "error" not in solved
    assert solved["max_displacement"] > 0
    assert (tmp_path / "results.vtu").exists()


def test_execute_tool_feeds_solver_errors_back_as_data(tmp_path):
    meshed = json.loads(execute_tool("mesh_box", {
        "length": 0.1, "width": 0.02, "height": 0.02,
        "element_size": 0.01, "output_path": str(tmp_path / "box.msh")}))
    bad = json.loads(execute_tool("run_fea", {
        "mesh_file": meshed["mesh_file"],
        "material": {"E": 210, "nu": 0.3},  # the classic GPa-as-Pa mistake
        "bcs": [{"surface": "xmin", "type": "fixed"}],
        "loads": []}))
    assert "error" in bad
    assert "GPa" in bad["error"]
