"""Verification for the tet4 element and the hex-to-tet decomposition.

The external check is the CoFEA reference band for Cook's membrane, which the
compressible case has to land in. That matters more than usual here: the
nearly-incompressible answers depend strongly on how the tets are laid out, so
agreement on the compressible problem is what separates "the element is right
and this mesh locks hard" from "the element is wrong".
"""

from collections import Counter

import numpy as np
import pytest

from minifea.cooks import (COOKS_CORNERS, COOKS_THICKNESS,
                           bilinear_hex_mesh, cooks_membrane, hex_to_tet4)
from minifea.hex8 import probe
from minifea.tet4 import _b_matrix, _element_gradient, _assemble_stiffness, solve_tet4
from minifea.material import elasticity_matrix

FIXED = [{"node_set": "clamped", "components": (0, 1, 2)},
         {"node_set": "zmin", "components": (2,)},
         {"node_set": "zmax", "components": (2,)}]
# tau = 6.25 MPa over the 16 mm x 1 mm loaded edge = 100 N, the benchmark load.
LOAD = [{"face_set": "loaded", "total_force": [0.0, 100.0, 0.0]}]
COOKS_TIP = (0.048, 0.060)
UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

# CoFEA reference bands quoted in the AMech validation study, tip displacement
# in mm at the top-right corner.
COFEA_COMPRESSIBLE = (32.0, 32.3)   # nu = 0.3333
COFEA_INCOMPRESSIBLE = (27.3, 28.0)  # nu = 0.4999


def tip_mm(mesh, result):
    return probe(mesh, result["displacement"][:, 1], COOKS_TIP) * 1000.0


def test_decomposition_preserves_volume():
    x, y = COOKS_CORNERS[:, 0], COOKS_CORNERS[:, 1]
    exact = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) * COOKS_THICKNESS
    for n in (2, 6, 12):
        mesh = hex_to_tet4(cooks_membrane(n))
        edges = mesh["points"][mesh["tets"][:, 1:]] - mesh["points"][mesh["tets"][:, [0]]]
        volumes = np.linalg.det(edges) / 6.0
        assert volumes.min() > 0.0, "decomposition produced an inverted tet"
        assert volumes.sum() == pytest.approx(exact, rel=1e-12)
        assert mesh["num_elements"] == 6 * n * n


def test_decomposition_is_conforming():
    """Every triangular face is used once (boundary) or twice (interior).

    A face used twice with mismatched vertices would show up here as a count
    of one on each side -- the crack that an alternating decomposition leaves.
    """
    mesh = hex_to_tet4(cooks_membrane(6))
    faces = Counter()
    for tet in mesh["tets"]:
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            faces[tuple(sorted(tet[list(face)]))] += 1
    assert set(faces.values()) <= {1, 2}


def test_patch_test():
    """A constant-strain element reproduces a linear field exactly.

    Deliberately run on a distorted *square* block, not on Cook's membrane:
    the boundary has to be identified exactly, and a bounding box does not
    find the boundary of a tapered panel -- its slanted edges are nowhere near
    the coordinate extremes. Leaving those nodes unconstrained turns the patch
    test into a different problem that any element will "fail".
    """
    import scipy.sparse.linalg as spla

    n = 3
    mesh = hex_to_tet4(bilinear_hex_mesh(UNIT_SQUARE, 1.0, n, n, n))
    points = mesh["points"]
    lo, hi = points.min(axis=0), points.max(axis=0)
    on_boundary = np.any(np.isclose(points, lo) | np.isclose(points, hi), axis=1)
    assert on_boundary.sum() < len(points), "patch test needs interior nodes"

    rng = np.random.default_rng(0)
    interior = ~on_boundary
    points[interior] += rng.uniform(-1.0, 1.0, (interior.sum(), 3)) * 0.22 / n

    D = elasticity_matrix(200e9, 0.4999)
    gradient = np.array([[1.0e-3, 7.0e-4, -3.0e-4],
                         [2.0e-4, -1.5e-3, 9.0e-4],
                         [-6.0e-4, 4.0e-4, 1.2e-3]])
    u_exact = (points @ gradient.T).ravel()

    K = _assemble_stiffness(points, mesh["tets"], D)
    boundary = np.flatnonzero(on_boundary)
    boundary_dofs = np.repeat(boundary * 3, 3) + np.tile([0, 1, 2], len(boundary))
    interior_dofs = np.setdiff1d(np.arange(3 * len(points)), boundary_dofs)

    u = np.zeros(3 * len(points))
    u[boundary_dofs] = u_exact[boundary_dofs]
    u[interior_dofs] = spla.spsolve(
        K[interior_dofs][:, interior_dofs],
        -(K[interior_dofs][:, boundary_dofs] @ u[boundary_dofs]))
    assert np.allclose(u, u_exact, atol=1e-10 * np.abs(u_exact).max())

    # ... and every element reports the same constant stress.
    strain = np.array([gradient[0, 0], gradient[1, 1], gradient[2, 2],
                       gradient[0, 1] + gradient[1, 0],
                       gradient[1, 2] + gradient[2, 1],
                       gradient[0, 2] + gradient[2, 0]])
    sigma_exact = D @ strain
    for elem_index, conn in enumerate(mesh["tets"]):
        dN_xyz, _ = _element_gradient(points[conn], elem_index)
        ue = u[np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 4)]
        sigma = D @ (_b_matrix(dN_xyz) @ ue)
        assert np.allclose(sigma, sigma_exact, rtol=1e-8,
                           atol=1e-8 * np.abs(sigma_exact).max())


def test_global_equilibrium():
    mesh = hex_to_tet4(cooks_membrane(6))
    result = solve_tet4(mesh, {"E": 70e6, "nu": 0.4999}, FIXED, LOAD)
    assert np.allclose(result["reaction"], [0.0, -100.0, 0.0], atol=1e-6)


def test_compressible_case_converges_into_the_cofea_band():
    """The external validation: nu = 0.3333 must reach the reference band."""
    tips = [tip_mm(mesh, solve_tet4(mesh, {"E": 70e6, "nu": 0.3333}, FIXED, LOAD))
            for mesh in (hex_to_tet4(cooks_membrane(n)) for n in (16, 32, 64))]
    assert tips == sorted(tips), f"should converge from below, got {tips}"
    assert tips[-1] > COFEA_COMPRESSIBLE[0] - 0.6, (
        f"tet4 should approach the CoFEA band {COFEA_COMPRESSIBLE}, got {tips}")
    assert tips[-1] < COFEA_COMPRESSIBLE[1]


def test_tet4_locks_severely_when_nearly_incompressible():
    """The demonstration: same mesh, one material change, answer collapses."""
    mesh = hex_to_tet4(cooks_membrane(16))
    compressible = tip_mm(mesh, solve_tet4(mesh, {"E": 70e6, "nu": 0.3333}, FIXED, LOAD))
    incompressible = tip_mm(mesh, solve_tet4(mesh, {"E": 70e6, "nu": 0.4999}, FIXED, LOAD))
    assert incompressible / compressible < 0.35
    assert incompressible < 0.4 * COFEA_INCOMPRESSIBLE[0], (
        "tet4 should fall far below the reference band, not near it; got "
        f"{incompressible:.2f} mm against a band of {COFEA_INCOMPRESSIBLE}")


def test_pressure_checkerboards_between_neighbouring_tets():
    """Tet4 is constant strain, so the spurious mode is element-to-element.

    Measured as the mean pressure jump across shared faces, relative to the
    range of the field. Near incompressibility drives it up sharply, which is
    the checkerboard.
    """
    def oscillation(mesh, pressure):
        faces = {}
        for index, tet in enumerate(mesh["tets"]):
            for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
                faces.setdefault(tuple(sorted(tet[list(face)])), []).append(index)
        jumps = [abs(pressure[a] - pressure[b])
                 for shared in faces.values() if len(shared) == 2
                 for a, b in [shared]]
        return np.mean(jumps) / (pressure.max() - pressure.min())

    mesh = hex_to_tet4(cooks_membrane(12))
    compressible = oscillation(
        mesh, solve_tet4(mesh, {"E": 70e6, "nu": 0.3333}, FIXED, LOAD)["pressure"])
    incompressible = oscillation(
        mesh, solve_tet4(mesh, {"E": 70e6, "nu": 0.4999}, FIXED, LOAD)["pressure"])
    assert incompressible > 2.0 * compressible
    assert incompressible > 0.15


def test_rejects_bad_input():
    hex_mesh = cooks_membrane(2)
    with pytest.raises(ValueError, match="expects a tet mesh"):
        solve_tet4(hex_mesh, {"E": 70e6, "nu": 0.3}, FIXED, LOAD)
    with pytest.raises(ValueError, match="expects a hex mesh"):
        hex_to_tet4(hex_to_tet4(hex_mesh))
    tet_mesh = hex_to_tet4(hex_mesh)
    with pytest.raises(ValueError, match="nu must be in"):
        solve_tet4(tet_mesh, {"E": 70e6, "nu": 0.5}, FIXED, LOAD)
    with pytest.raises(ValueError, match="unknown node set"):
        solve_tet4(tet_mesh, {"E": 70e6, "nu": 0.3}, [{"node_set": "fixed_end"}], LOAD)
