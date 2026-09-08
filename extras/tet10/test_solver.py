"""Verification tests for the minifea solver.

These are the tests that make the toy solver trustworthy: exact solutions
where exact solutions exist, classical beam theory where they don't.
"""

import numpy as np
import pytest

import pytest

pytest.importorskip("gmsh")
pytest.importorskip("meshio")

from tet10 import mesh_box, solve


@pytest.fixture(scope="module")
def bar_mesh(tmp_path_factory):
    """Slender bar, coarse mesh: 0.2 x 0.02 x 0.02 m."""
    path = tmp_path_factory.mktemp("meshes") / "bar.msh"
    return mesh_box(0.2, 0.02, 0.02, element_size=0.005,
                    output_path=str(path))


def test_uniaxial_tension_exact(bar_mesh):
    """nu = 0 uniaxial tension is reproduced exactly (patch-test level).

    With zero Poisson ratio the fully-fixed end does not disturb the uniaxial
    field, so displacement, stress, and reactions must all match theory to
    numerical precision.
    """
    E, t = 210e9, 1e6  # Pa
    L, w, h = 0.2, 0.02, 0.02
    res = solve(
        bar_mesh["mesh_file"],
        material={"E": E, "nu": 0.0},
        bcs=[{"surface": "xmin", "type": "fixed"}],
        loads=[{"surface": "xmax", "type": "traction", "vector": [t, 0, 0]}],
    )
    delta_exact = t * L / E
    assert res["max_displacement"] == pytest.approx(delta_exact, rel=1e-6)
    assert res["max_displacement_vector"][0] == pytest.approx(delta_exact,
                                                              rel=1e-6)
    assert res["max_von_mises"] == pytest.approx(t, rel=1e-6)
    # Reaction balances the applied load: R_x = -t * A.
    assert res["reactions"]["xmin"][0] == pytest.approx(-t * w * h, rel=1e-6)


def test_pressure_equals_compressive_traction(bar_mesh):
    """Pressure p on xmax must equal a traction of -p * (outward normal)."""
    mat = {"E": 210e9, "nu": 0.0}
    bcs = [{"surface": "xmin", "type": "fixed"}]
    res_p = solve(bar_mesh["mesh_file"], mat, bcs,
                  [{"surface": "xmax", "type": "pressure", "value": 1e6}])
    res_t = solve(bar_mesh["mesh_file"], mat, bcs,
                  [{"surface": "xmax", "type": "traction",
                    "vector": [-1e6, 0, 0]}])
    assert res_p["max_displacement"] == pytest.approx(
        res_t["max_displacement"], rel=1e-9)


def test_cantilever_tip_deflection_vs_beam_theory(bar_mesh):
    """End-loaded cantilever vs Timoshenko beam theory, within 5%.

    delta = F L^3 / (3 E I) + F L / (k G A), k = 5/6 for a square section.
    """
    E, nu = 210e9, 0.3
    L, w, h = 0.2, 0.02, 0.02
    F = 1000.0  # N, applied as uniform traction on the end face
    res = solve(
        bar_mesh["mesh_file"],
        material={"E": E, "nu": nu},
        bcs=[{"surface": "xmin", "type": "fixed"}],
        loads=[{"surface": "xmax", "type": "traction",
                "vector": [0, 0, -F / (w * h)]}],
        output_vtu=None,
    )
    I = w * h ** 3 / 12.0
    G = E / (2 * (1 + nu))
    delta_beam = F * L ** 3 / (3 * E * I) + F * L / (5.0 / 6.0 * G * w * h)
    assert res["max_displacement"] == pytest.approx(delta_beam, rel=0.05)
    # Total vertical reaction balances the applied shear load.
    assert res["reactions"]["xmin"][2] == pytest.approx(F, rel=1e-6)
    # Bending stress at the root: sigma = M c / I, von Mises should be near it
    # (stress concentration at the fixed corners pushes the max above beam
    # theory; nodal max should at least reach the beam value).
    sigma_beam = F * L * (h / 2) / I
    assert res["max_von_mises"] >= 0.9 * sigma_beam


def test_unknown_surface_lists_available(bar_mesh):
    with pytest.raises(ValueError, match="xmin"):
        solve(bar_mesh["mesh_file"], {"E": 210e9, "nu": 0.3},
              bcs=[{"surface": "left_face", "type": "fixed"}], loads=[])


def test_no_bcs_mentions_rigid_body(bar_mesh):
    with pytest.raises(ValueError, match="rigid body"):
        solve(bar_mesh["mesh_file"], {"E": 210e9, "nu": 0.3}, bcs=[],
              loads=[])


def test_gpa_mistake_is_caught(bar_mesh):
    """E = 210 (GPa entered as Pa) must produce the units hint."""
    with pytest.raises(ValueError, match="GPa"):
        solve(bar_mesh["mesh_file"], {"E": 210, "nu": 0.3},
              bcs=[{"surface": "xmin", "type": "fixed"}], loads=[])


def test_missing_mesh_file_message():
    with pytest.raises(ValueError, match="mesh_box"):
        solve("no_such_mesh.msh", {"E": 210e9, "nu": 0.3},
              bcs=[{"surface": "xmin", "type": "fixed"}], loads=[])
