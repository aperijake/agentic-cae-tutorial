"""Linear tetrahedra (tet4): the constant-strain element.

Tet4 is what an automated meshing workflow produces by default, and it is the
most severe volumetric locker in common use. Strain is constant over the
element, so there is exactly one pressure value per element -- which is what
makes near-incompressible locking *look* different here than on hexes:

- On a hex with full integration, the spurious pressure oscillation lives
  between the Gauss points inside each element, and any element-averaged plot
  hides it.
- On a tet4 mesh there is nowhere inside an element for it to hide, so it
  appears as the textbook element-to-element checkerboard: neighboring
  tetrahedra alternating between high and low pressure.

Same pathology, two different pictures, decided by the element. There is no
B-bar variant here on purpose: with one constant-strain point per element the
mean-dilatation projection is the identity, and it changes nothing. Fixing
tet4 needs a different remedy (an averaged nodal pressure, a mixed
formulation, or simply a better element).

Conventions follow the rest of minifea: Voigt order [xx, yy, zz, xy, yz, xz]
with engineering shear strain, and SI units (meters, Pascals, Newtons).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from minifea.material import elasticity_matrix

# Reference gradients of N = [1-r-s-t, r, s, t]. Constant, hence "constant
# strain": every derivative below is independent of position in the element.
_TET4_DN_REF = np.array([
    [-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
])


def _element_gradient(coords: np.ndarray, elem_index: int):
    """Physical shape gradients (4, 3) and volume [m^3] for one tet."""
    J = _TET4_DN_REF.T @ coords
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError(
            f"Element {elem_index} has non-positive volume ({detJ / 6.0:.3e} "
            "m^3): the tet is inverted. Check the node ordering of the "
            "decomposition that produced it.")
    return _TET4_DN_REF @ np.linalg.inv(J).T, detJ / 6.0


def _b_matrix(dN_xyz: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix B (6, 12) from shape gradients (4, 3)."""
    B = np.zeros((6, 12))
    for i in range(4):
        dx, dy, dz = dN_xyz[i]
        c = 3 * i
        B[0, c] = dx
        B[1, c + 1] = dy
        B[2, c + 2] = dz
        B[3, c] = dy
        B[3, c + 1] = dx
        B[4, c + 1] = dz
        B[4, c + 2] = dy
        B[5, c] = dz
        B[5, c + 2] = dx
    return B


def _assemble_stiffness(points, tets, D):
    n_dofs = 3 * len(points)
    rows, cols, vals = [], [], []
    for elem_index, conn in enumerate(tets):
        dN_xyz, volume = _element_gradient(points[conn], elem_index)
        B = _b_matrix(dN_xyz)
        Ke = (B.T @ D @ B) * volume
        dofs = np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 4)
        rows.append(np.repeat(dofs, 12))
        cols.append(np.tile(dofs, 12))
        vals.append(Ke.ravel())
    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dofs, n_dofs))
    return K.tocsr()


def _assemble_tractions(points, face_sets, tractions):
    """Consistent nodal forces from tractions on triangular faces.

    A constant traction on a linear triangle splits equally between its three
    nodes, which is what integrating the shape functions gives exactly.
    """
    F = np.zeros(3 * len(points))
    for load in tractions:
        name = load.get("face_set")
        if name not in face_sets:
            raise ValueError(
                f"Traction refers to unknown face set {name!r}. Available "
                f"face sets in this mesh: {sorted(face_sets)}.")
        tris = face_sets[name]
        areas = 0.5 * np.linalg.norm(
            np.cross(points[tris[:, 1]] - points[tris[:, 0]],
                     points[tris[:, 2]] - points[tris[:, 0]]), axis=1)

        if ("vector" in load) == ("total_force" in load):
            raise ValueError(
                f"Traction on {name!r} needs exactly one of 'vector' "
                "(traction in Pa) or 'total_force' (resultant in N).")
        if "vector" in load:
            t = np.asarray(load["vector"], dtype=float)
        else:
            t = np.asarray(load["total_force"], dtype=float) / areas.sum()

        for tri, area in zip(tris, areas):
            for node in tri:
                F[3 * node: 3 * node + 3] += t * area / 3.0
    return F


def solve_tet4(mesh: dict, material: dict, fixed: list,
               tractions: list = ()) -> dict:
    """Solve small-displacement linear elasticity on a tet4 mesh.

    Arguments match :func:`minifea.hex8.solve_hex`, except that ``mesh`` must
    carry ``tets`` (use :func:`minifea.cooks.hex_to_tet4`) and there is no
    formulation choice.

    Returns a result dict with displacement (n_nodes, 3), the element pressure
    and von Mises fields -- one value per element, exactly as the element
    computes them, with no averaging -- the reaction resultant, and the strain
    energy. All SI.
    """
    if "tets" not in mesh:
        raise ValueError(
            "solve_tet4 expects a tet mesh with a 'tets' key. Convert a hex "
            "mesh first with minifea.cooks.hex_to_tet4.")
    if not isinstance(material, dict) or "E" not in material or "nu" not in material:
        raise ValueError(
            "material must be a dict with keys 'E' (Young's modulus, Pa) and "
            "'nu' (Poisson's ratio), e.g. {'E': 70e6, 'nu': 0.4999}.")
    E, nu = float(material["E"]), float(material["nu"])
    if not E > 0:
        raise ValueError(f"Young's modulus E must be positive, got {E!r}.")
    if not 0.0 <= nu < 0.5:
        raise ValueError(
            f"Poisson's ratio nu must be in [0, 0.5), got {nu!r}. At exactly "
            "0.5 the material is incompressible and this displacement-based "
            "formulation is singular; use 0.4999 to approach it.")
    if not fixed:
        raise ValueError(
            "No constraints given; the model has rigid body modes. Fix at "
            "least one node set, e.g. "
            "[{'node_set': 'clamped', 'components': (0, 1, 2)}].")

    points, tets = mesh["points"], mesh["tets"]
    node_sets, face_sets = mesh["node_sets"], mesh["face_sets"]
    n_dofs = 3 * len(points)

    D = elasticity_matrix(E, nu)
    K = _assemble_stiffness(points, tets, D)
    F = _assemble_tractions(points, face_sets, tractions)

    fixed_dof_list = []
    for entry in fixed:
        name = entry.get("node_set")
        if name not in node_sets:
            raise ValueError(
                f"Constraint refers to unknown node set {name!r}. Available "
                f"node sets in this mesh: {sorted(node_sets)}.")
        for comp in entry.get("components", (0, 1, 2)):
            if comp not in (0, 1, 2):
                raise ValueError(
                    f"Constraint components must be 0, 1 or 2, got {comp!r}.")
            fixed_dof_list.append(node_sets[name] * 3 + comp)

    fixed_dofs = np.unique(np.concatenate(fixed_dof_list))
    free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)

    u = np.zeros(n_dofs)
    u[free_dofs] = spla.spsolve(K[free_dofs][:, free_dofs], F[free_dofs])

    residual = K @ u - F
    reaction = np.zeros(3)
    np.add.at(reaction, fixed_dofs % 3, residual[fixed_dofs])

    pressure = np.empty(len(tets))
    von_mises = np.empty(len(tets))
    for elem_index, conn in enumerate(tets):
        dN_xyz, _ = _element_gradient(points[conn], elem_index)
        ue = u[np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 4)]
        sxx, syy, szz, sxy, syz, sxz = D @ (_b_matrix(dN_xyz) @ ue)
        pressure[elem_index] = -(sxx + syy + szz) / 3.0
        von_mises[elem_index] = np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))

    disp = u.reshape(-1, 3)
    return {
        "formulation": "tet4",
        "displacement": disp,
        "pressure": pressure,
        "von_mises": von_mises,
        "num_nodes": int(len(points)),
        "num_elements": int(len(tets)),
        "num_dofs": int(n_dofs),
        "max_displacement": float(np.linalg.norm(disp, axis=1).max()),
        "strain_energy": float(0.5 * u @ (K @ u)),
        "reaction": reaction.tolist(),
    }
