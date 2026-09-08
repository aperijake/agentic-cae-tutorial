"""Verification for the hex8 element and its B-bar variant.

The patch test is the one that matters. B-bar is a projection of the
strain-displacement operator, so it is fair to ask whether it is still a
legitimate element or just a convenient smudge. Reproducing an arbitrary
linear displacement field exactly, on a distorted mesh, is the answer: it is
what the mean-dilatation form inherits from its Hu-Washizu basis, and it is
the property a sloppy implementation loses first.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from minifea.cooks import COOKS_CORNERS, bilinear_hex_mesh, cooks_membrane
from minifea.hex8 import (FORMULATIONS, _assemble_stiffness, _b_matrix,
                          _element_gradients, gauss_point_pressure, probe,
                          solve_hex)
from minifea.material import elasticity_matrix

UNIT_SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

FIXED_COOKS = [{"node_set": "clamped", "components": (0, 1, 2)},
               {"node_set": "zmin", "components": (2,)},
               {"node_set": "zmax", "components": (2,)}]
SHEAR_LOAD = [{"face_set": "loaded", "total_force": [0.0, 1.0, 0.0]}]
COOKS_TIP = (0.048, 0.060)


def distorted_block(n=3, amplitude=0.22, seed=0):
    """An n^3 block of hexes with randomly perturbed interior nodes."""
    mesh = bilinear_hex_mesh(UNIT_SQUARE, 1.0, n, n, n)
    points = mesh["points"]
    lo, hi = points.min(axis=0), points.max(axis=0)
    on_boundary = np.any(np.isclose(points, lo) | np.isclose(points, hi), axis=1)

    rng = np.random.default_rng(seed)
    interior = ~on_boundary
    points[interior] += rng.uniform(-1.0, 1.0, (interior.sum(), 3)) * amplitude / n
    mesh["points"] = points
    mesh["on_boundary"] = on_boundary
    return mesh


@pytest.mark.parametrize("formulation", FORMULATIONS)
@pytest.mark.parametrize("nu", [0.3, 0.4999])
def test_patch_test(formulation, nu):
    """A linear displacement field is reproduced exactly on a distorted mesh."""
    mesh = distorted_block()
    points, hexes = mesh["points"], mesh["hexes"]
    D = elasticity_matrix(200e9, nu)

    gradient = np.array([[1.0e-3, 7.0e-4, -3.0e-4],
                         [2.0e-4, -1.5e-3, 9.0e-4],
                         [-6.0e-4, 4.0e-4, 1.2e-3]])
    u_exact = (points @ gradient.T).ravel()

    K = _assemble_stiffness(points, hexes, D, formulation)
    boundary_dofs = np.repeat(np.flatnonzero(mesh["on_boundary"]) * 3, 3) \
        + np.tile([0, 1, 2], int(mesh["on_boundary"].sum()))
    interior_dofs = np.setdiff1d(np.arange(3 * len(points)), boundary_dofs)

    u = np.zeros(3 * len(points))
    u[boundary_dofs] = u_exact[boundary_dofs]
    rhs = -(K[interior_dofs][:, boundary_dofs] @ u[boundary_dofs])
    u[interior_dofs] = spla.spsolve(K[interior_dofs][:, interior_dofs], rhs)

    scale = np.abs(u_exact).max()
    assert np.allclose(u, u_exact, atol=1e-10 * scale), (
        f"{formulation} failed the patch test: interior displacements deviate "
        f"from the exact linear field by "
        f"{np.abs(u - u_exact).max() / scale:.2e} (relative)")

    # ... and the stress it produces is constant across every Gauss point.
    strain = np.array([gradient[0, 0], gradient[1, 1], gradient[2, 2],
                       gradient[0, 1] + gradient[1, 0],
                       gradient[1, 2] + gradient[2, 1],
                       gradient[0, 2] + gradient[2, 0]])
    sigma_exact = D @ strain
    for elem_index, conn in enumerate(hexes):
        grads, _, _, mean_gradient = _element_gradients(points[conn], elem_index)
        dN_bar = mean_gradient if formulation == "bbar" else None
        ue = u[np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 8)]
        for dN_xyz in grads:
            sigma = D @ (_b_matrix(dN_xyz, dN_bar) @ ue)
            assert np.allclose(sigma, sigma_exact,
                               rtol=1e-8, atol=1e-8 * np.abs(sigma_exact).max())


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_rigid_body_motion_is_stress_free(formulation):
    """Translation plus infinitesimal rotation produces no strain energy."""
    mesh = distorted_block()
    points, hexes = mesh["points"], mesh["hexes"]
    K = _assemble_stiffness(points, hexes, elasticity_matrix(200e9, 0.3),
                            formulation)

    omega = np.array([[0.0, -3e-4, 1e-4], [3e-4, 0.0, -2e-4], [-1e-4, 2e-4, 0.0]])
    u = (points @ omega.T + np.array([1e-3, -2e-3, 5e-4])).ravel()
    reference = np.linalg.norm(K @ (points.ravel()))
    assert np.linalg.norm(K @ u) < 1e-9 * reference


@pytest.mark.parametrize("formulation", FORMULATIONS)
def test_global_equilibrium(formulation):
    """Reactions balance the applied load to machine precision."""
    mesh = cooks_membrane(6)
    result = solve_hex(mesh, {"E": 70e6, "nu": 0.3}, FIXED_COOKS, SHEAR_LOAD,
                       formulation=formulation)
    assert np.allclose(result["reaction"], [0.0, -1.0, 0.0], atol=1e-10)


def test_full_integration_locks_and_bbar_does_not():
    """The locking demonstration, as an assertion.

    Near-incompressible: full integration is dramatically too stiff.
    Compressible: the two formulations agree closely. That contrast is what
    identifies the mechanism as volumetric locking rather than a coding error
    in one of the two paths.
    """
    mesh = cooks_membrane(8)
    material_ratios = {}
    for nu in (0.3, 0.4999):
        tips = {}
        for formulation in FORMULATIONS:
            result = solve_hex(mesh, {"E": 70e6, "nu": nu}, FIXED_COOKS,
                               SHEAR_LOAD, formulation=formulation)
            tips[formulation] = probe(mesh, result["displacement"][:, 1], COOKS_TIP)
        material_ratios[nu] = tips["full"] / tips["bbar"]

    assert 0.9 < material_ratios[0.3] <= 1.0, (
        "away from incompressibility the two formulations should nearly agree; "
        f"got a stiffness ratio of {material_ratios[0.3]:.3f}")
    assert material_ratios[0.4999] < 0.35, (
        "at nu = 0.4999 full integration should lock badly; got a tip "
        f"displacement ratio of {material_ratios[0.4999]:.3f}")


def test_bbar_converges_under_refinement():
    """B-bar tip displacement converges; successive changes shrink."""
    tips = []
    for n in (4, 8, 16, 32):
        mesh = cooks_membrane(n)
        result = solve_hex(mesh, {"E": 70e6, "nu": 0.4999}, FIXED_COOKS,
                           SHEAR_LOAD, formulation="bbar")
        tips.append(probe(mesh, result["displacement"][:, 1], COOKS_TIP))

    deltas = np.abs(np.diff(tips))
    assert np.all(deltas[1:] < deltas[:-1]), (
        f"refinement should shrink successive changes, got {deltas}")
    assert deltas[-1] / abs(tips[-1]) < 0.02


def test_pressure_oscillation_is_intra_element_and_only_under_full_integration():
    """The visual signature of locking, as an assertion.

    Measured as the pressure swing between Gauss points inside one element,
    relative to the range of the element-averaged field -- that is, relative to
    the smooth picture an ordinary contour plot would show. Under full
    integration at nu = 0.4999 the hidden swing is about four times the range
    of the visible field, and that ratio holds under refinement. B-bar makes
    the pressure element-constant, so its swing is zero to round-off.

    The oscillation is deliberately measured within elements rather than
    between them: it is not the inter-element checkerboard of an unstable
    mixed pairing, and an element-averaged field does not show it.
    """
    mesh = cooks_membrane(12)
    material = {"E": 70e6, "nu": 0.4999}
    swings = {}
    for formulation in FORMULATIONS:
        result = solve_hex(mesh, material, FIXED_COOKS, SHEAR_LOAD,
                           formulation=formulation)
        _, pressure = gauss_point_pressure(mesh, material,
                                           result["displacement"], formulation)
        per_element = pressure.reshape(-1, 8)
        within = (per_element.max(axis=1) - per_element.min(axis=1)).mean()
        averaged = result["pressure"]
        swings[formulation] = within / (averaged.max() - averaged.min())

    assert swings["full"] > 2.0, (
        "the within-element pressure swing under full integration should dwarf "
        f"the averaged field it hides behind; got {swings['full']:.2f}x")
    assert swings["bbar"] < 1e-6, (
        "B-bar pressure should be element-constant by construction; got a "
        f"within-element swing of {swings['bbar']:.3e}")


def test_bbar_tip_displacement_is_insensitive_to_incompressibility():
    """B-bar holds its answer as nu -> 0.5; full integration collapses."""
    mesh = cooks_membrane(12)
    tips = {f: [] for f in FORMULATIONS}
    for nu in (0.3, 0.49, 0.499, 0.4999):
        for formulation in FORMULATIONS:
            result = solve_hex(mesh, {"E": 70e6, "nu": nu}, FIXED_COOKS,
                               SHEAR_LOAD, formulation=formulation)
            tips[formulation].append(
                probe(mesh, result["displacement"][:, 1], COOKS_TIP))

    bbar = np.array(tips["bbar"])
    assert np.ptp(bbar) / bbar.mean() < 0.2, (
        f"B-bar should be nearly insensitive to nu, got {bbar}")
    assert tips["full"][-1] / tips["full"][0] < 0.35, (
        f"full integration should stiffen sharply as nu -> 0.5, got {tips['full']}")


def test_matches_classical_cooks_benchmark():
    """Reproduce the textbook Cook's membrane value.

    The benchmark is quoted dimensionlessly (E = 1, thickness = 1, total shear
    F = 1, nu = 1/3) with a converged vertical displacement of about 23.9 at
    the midpoint of the loaded edge, under plane stress. Linear elasticity
    makes the displacement proportional to F / (E * t) at fixed shape, so that
    value maps onto this model's units by the same factor.

    Two details decide which number a reference is quoting, and mixing them up
    costs about 10% each: plane stress versus plane strain, and whether the
    displacement is read at the free-edge midpoint or the top corner. This test
    pins the combination that matches the classical value.
    """
    mesh = cooks_membrane(48)
    result = solve_hex(mesh, {"E": 70e6, "nu": 1.0 / 3.0},
                       [{"node_set": "clamped", "components": (0, 1, 2)}],
                       SHEAR_LOAD, formulation="bbar")
    scale = 1.0 / (70e6 * 0.001)  # F / (E * t), with F = 1 N
    midpoint = probe(mesh, result["displacement"][:, 1], (0.048, 0.052)) / scale
    assert midpoint == pytest.approx(23.9, rel=0.01)


def test_mesher_reproduces_corners_and_area():
    mesh = cooks_membrane(5)
    points = mesh["points"]
    for corner in COOKS_CORNERS:
        assert np.min(np.linalg.norm(points[:, :2] - corner, axis=1)) < 1e-15

    quad_area = 0.0
    for quad in mesh["face_sets"]["loaded"]:
        c = points[quad]
        quad_area += 0.5 * np.linalg.norm(np.cross(c[2] - c[0], c[3] - c[1]))
    assert quad_area == pytest.approx(mesh["loaded_area"], rel=1e-12)


def test_mesher_rejects_bad_input():
    with pytest.raises(ValueError, match="counter-clockwise"):
        bilinear_hex_mesh(UNIT_SQUARE[::-1], 1.0, 2, 2, 1)
    with pytest.raises(ValueError, match="nx must be a positive integer"):
        bilinear_hex_mesh(UNIT_SQUARE, 1.0, 0, 2, 1)
    with pytest.raises(ValueError, match="thickness must be positive"):
        bilinear_hex_mesh(UNIT_SQUARE, -1.0, 2, 2, 1)


def test_solver_rejects_bad_input():
    mesh = cooks_membrane(2)
    with pytest.raises(ValueError, match="Unknown formulation"):
        solve_hex(mesh, {"E": 70e6, "nu": 0.3}, FIXED_COOKS, formulation="bbarr")
    with pytest.raises(ValueError, match="nu must be in"):
        solve_hex(mesh, {"E": 70e6, "nu": 0.5}, FIXED_COOKS)
    with pytest.raises(ValueError, match="unknown node set"):
        solve_hex(mesh, {"E": 70e6, "nu": 0.3}, [{"node_set": "fixed_end"}])
    with pytest.raises(ValueError, match="unknown face set"):
        solve_hex(mesh, {"E": 70e6, "nu": 0.3}, FIXED_COOKS,
                  [{"face_set": "tip", "vector": [0, 1, 0]}])
    with pytest.raises(ValueError, match="exactly one of"):
        solve_hex(mesh, {"E": 70e6, "nu": 0.3}, FIXED_COOKS,
                  [{"face_set": "loaded", "vector": [0, 1, 0],
                    "total_force": [0, 1, 0]}])
