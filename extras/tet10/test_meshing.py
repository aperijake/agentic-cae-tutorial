"""Tests for the gmsh box mesher."""

import pytest

import pytest

pytest.importorskip("gmsh")

from tet10 import mesh_box


def test_mesh_box_basic(tmp_path):
    out = tmp_path / "box.msh"
    res = mesh_box(0.1, 0.05, 0.02, element_size=0.01, output_path=str(out))
    assert out.exists()
    assert res["mesh_file"] == str(out)
    assert res["element_type"] == "tet10"
    assert res["num_nodes"] > 0
    assert res["num_elements"] > 0
    assert set(res["surface_groups"]) == {"xmin", "xmax", "ymin", "ymax",
                                          "zmin", "zmax"}


def test_mesh_box_rejects_bad_inputs(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        mesh_box(-1.0, 0.1, 0.1, 0.01, output_path=str(tmp_path / "a.msh"))
    with pytest.raises(ValueError, match="element_size"):
        mesh_box(0.1, 0.1, 0.01, element_size=0.05,
                 output_path=str(tmp_path / "b.msh"))
