"""A small 3D linear-elastic finite element solver.

Quadratic tetrahedra (tet10), small displacements, isotropic material.
SI units everywhere: meters, Pascals, Newtons.

This solver is intentionally compact and readable rather than fast or general.
It exists so that (a) the whole simulation chain in this tutorial is open and
inspectable, and (b) an AI agent can read and extend it.

Conventions:
- Meshes come from ``minifea.meshing.mesh_box`` (gmsh .msh with named physical
  surface groups). Boundary conditions and loads refer to those group names.
- Voigt stress/strain order: [xx, yy, zz, xy, yz, xz] with engineering shear.
- Pressure loads are positive pushing INTO the surface. Outward normals are
  determined assuming a convex body (true for the box meshes used here).
"""

import os

import meshio
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from minifea.material import elasticity_matrix

# ---------------------------------------------------------------------------
# Element definitions (node ordering follows meshio/VTK conventions)
# ---------------------------------------------------------------------------

# 4-point Gauss rule for tetrahedra: exact for quadratic integrands.
_TET_A = 0.5854101966249685  # (5 + 3*sqrt(5)) / 20
_TET_B = 0.1381966011250105  # (5 - sqrt(5)) / 20
_TET_GAUSS_POINTS = np.array([
    [_TET_B, _TET_B, _TET_B],
    [_TET_A, _TET_B, _TET_B],
    [_TET_B, _TET_A, _TET_B],
    [_TET_B, _TET_B, _TET_A],
])
_TET_GAUSS_WEIGHT = 1.0 / 24.0  # reference tet volume 1/6, four equal weights

# Local (reference) coordinates of the 10 nodes, VTK order:
# 4 vertices, then edge midpoints (0,1),(1,2),(2,0),(0,3),(1,3),(2,3).
_TET10_NODE_COORDS = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
    [0.5, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
])

# 3-point Gauss rule for triangles: exact for quadratic integrands.
_TRI_GAUSS_POINTS = np.array([
    [1.0 / 6.0, 1.0 / 6.0],
    [2.0 / 3.0, 1.0 / 6.0],
    [1.0 / 6.0, 2.0 / 3.0],
])
_TRI_GAUSS_WEIGHT = 1.0 / 6.0  # reference triangle area 1/2, three equal weights


def _tet10_shape(xi: float, eta: float, zeta: float):
    """Shape functions N (10,) and reference gradients dN (10,3) for a tet10."""
    lam = np.array([1.0 - xi - eta - zeta, xi, eta, zeta])
    dlam = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    # Edge (i, j) pairs in VTK order for nodes 4..9.
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

    N = np.empty(10)
    dN = np.empty((10, 3))
    for i in range(4):
        N[i] = lam[i] * (2.0 * lam[i] - 1.0)
        dN[i] = (4.0 * lam[i] - 1.0) * dlam[i]
    for k, (i, j) in enumerate(edges):
        N[4 + k] = 4.0 * lam[i] * lam[j]
        dN[4 + k] = 4.0 * (lam[i] * dlam[j] + lam[j] * dlam[i])
    return N, dN


def _tri6_shape(r: float, s: float):
    """Shape functions N (6,) and reference gradients dN (6,2) for a tri6."""
    lam = np.array([1.0 - r - s, r, s])
    dlam = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    edges = [(0, 1), (1, 2), (2, 0)]

    N = np.empty(6)
    dN = np.empty((6, 2))
    for i in range(3):
        N[i] = lam[i] * (2.0 * lam[i] - 1.0)
        dN[i] = (4.0 * lam[i] - 1.0) * dlam[i]
    for k, (i, j) in enumerate(edges):
        N[3 + k] = 4.0 * lam[i] * lam[j]
        dN[3 + k] = 4.0 * (lam[i] * dlam[j] + lam[j] * dlam[i])
    return N, dN


# ---------------------------------------------------------------------------
# Mesh reading
# ---------------------------------------------------------------------------

def _read_mesh(mesh_file: str):
    """Read a gmsh mesh into (points, tets, surfaces).

    points:   (n_nodes, 3) float array
    tets:     (n_elems, 10) int array of node indices
    surfaces: dict name -> (n_tris, 6) int array of boundary triangle nodes
    """
    if not os.path.exists(mesh_file):
        raise ValueError(
            f"Mesh file not found: {mesh_file!r}. Run mesh_box first, and pass "
            "the 'mesh_file' path it returned.")
    mesh = meshio.read(mesh_file)

    # gmsh physical names: field_data maps name -> [tag, dimension]
    tag_to_name = {int(v[0]): name for name, v in mesh.field_data.items()
                   if int(v[1]) == 2}

    tets = []
    surfaces: dict[str, list] = {}
    for block_index, block in enumerate(mesh.cells):
        if block.type == "tetra10":
            tets.append(block.data)
        elif block.type == "triangle6":
            phys = mesh.cell_data["gmsh:physical"][block_index]
            for tri, tag in zip(block.data, phys):
                name = tag_to_name.get(int(tag))
                if name is not None:
                    surfaces.setdefault(name, []).append(tri)
        elif block.type == "tetra":
            raise ValueError(
                "Mesh contains linear tetrahedra (tet4); this solver requires "
                "quadratic tet10 meshes. Regenerate the mesh with mesh_box.")

    if not tets:
        raise ValueError(
            f"No tet10 volume elements found in {mesh_file!r}. Regenerate the "
            "mesh with mesh_box.")

    points = np.asarray(mesh.points, dtype=float)
    tets_arr = np.vstack(tets).astype(int)
    surfaces_arr = {name: np.asarray(tris, dtype=int)
                    for name, tris in surfaces.items()}
    return points, tets_arr, surfaces_arr


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(material: dict, bcs: list, loads: list, surfaces: dict):
    available = sorted(surfaces.keys())

    if not isinstance(material, dict) or "E" not in material or "nu" not in material:
        raise ValueError(
            "material must be a dict with keys 'E' (Young's modulus, Pa) and "
            "'nu' (Poisson's ratio), e.g. {'E': 210e9, 'nu': 0.3} for steel.")
    E, nu = material["E"], material["nu"]
    if not E > 0:
        raise ValueError(f"Young's modulus E must be positive, got {E!r}.")
    if E < 1e6:
        raise ValueError(
            f"Young's modulus E = {E!r} Pa looks implausibly small for a "
            "structural material. Did you enter GPa instead of Pa? "
            "E.g. steel is E=210e9 (Pa), not E=210.")
    if not 0.0 <= nu < 0.5:
        raise ValueError(
            f"Poisson's ratio nu must be in [0, 0.5), got {nu!r} "
            "(nu = 0.5 is incompressible and singular for this formulation).")

    def check_surface(entry, kind):
        name = entry.get("surface")
        if name not in surfaces:
            raise ValueError(
                f"{kind} refers to unknown surface {name!r}. Available "
                f"surface names in this mesh: {available}.")

    if not bcs:
        raise ValueError(
            "No boundary conditions given. The model has rigid body modes; "
            "fix at least one surface, e.g. "
            "[{'surface': 'xmin', 'type': 'fixed'}].")
    for bc in bcs:
        check_surface(bc, "Boundary condition")
        if bc.get("type") != "fixed":
            raise ValueError(
                f"Unsupported BC type {bc.get('type')!r}; only 'fixed' is "
                "supported.")

    for load in loads:
        check_surface(load, "Load")
        ltype = load.get("type")
        if ltype == "traction":
            vec = load.get("vector")
            if vec is None or len(vec) != 3:
                raise ValueError(
                    "Traction load needs 'vector': [tx, ty, tz] in Pa "
                    "(force per unit area).")
        elif ltype == "pressure":
            if "value" not in load:
                raise ValueError(
                    "Pressure load needs 'value' in Pa (positive pushes into "
                    "the surface).")
        else:
            raise ValueError(
                f"Unsupported load type {ltype!r}; use 'traction' or "
                "'pressure'.")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

_elasticity_matrix = elasticity_matrix


def _b_matrix(dN_xyz: np.ndarray) -> np.ndarray:
    """Strain-displacement matrix B (6, 30) from shape gradients (10, 3)."""
    B = np.zeros((6, 30))
    for i in range(10):
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
    """Assemble the global stiffness matrix (CSR)."""
    gauss = [_tet10_shape(*gp) for gp in _TET_GAUSS_POINTS]
    n_dofs = 3 * len(points)

    rows, cols, vals = [], [], []
    for elem_index, conn in enumerate(tets):
        coords = points[conn]  # (10, 3)
        Ke = np.zeros((30, 30))
        for _, dN_ref in gauss:
            J = dN_ref.T @ coords  # J[a, b] = d x_b / d xi_a
            detJ = np.linalg.det(J)
            if detJ <= 0.0:
                raise ValueError(
                    f"Element {elem_index} has non-positive Jacobian "
                    f"({detJ:.3e}): the mesh is inverted or badly distorted. "
                    "Re-mesh with a different element size.")
            dN_xyz = dN_ref @ np.linalg.inv(J).T
            B = _b_matrix(dN_xyz)
            Ke += (B.T @ D @ B) * detJ * _TET_GAUSS_WEIGHT

        dofs = np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 10)
        rows.append(np.repeat(dofs, 30))
        cols.append(np.tile(dofs, 30))
        vals.append(Ke.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dofs, n_dofs))
    return K.tocsr()


def _assemble_loads(points, surfaces, loads, body_centroid):
    """Consistent nodal forces from surface tractions and pressures."""
    n_dofs = 3 * len(points)
    F = np.zeros(n_dofs)
    gauss = [_tri6_shape(*gp) for gp in _TRI_GAUSS_POINTS]

    for load in loads:
        tris = surfaces[load["surface"]]
        for tri in tris:
            coords = points[tri]  # (6, 3)
            for N, dN in gauss:
                t_r = dN[:, 0] @ coords
                t_s = dN[:, 1] @ coords
                area_vec = np.cross(t_r, t_s)  # |.| = dA/(dr ds), dir = normal
                if load["type"] == "traction":
                    f_gp = np.asarray(load["vector"], float) \
                        * np.linalg.norm(area_vec) * _TRI_GAUSS_WEIGHT
                else:  # pressure
                    # Orient the normal outward (convex body assumption),
                    # then push into the surface: f = -p * n_outward.
                    x_gp = N @ coords
                    if np.dot(area_vec, x_gp - body_centroid) < 0.0:
                        area_vec = -area_vec
                    f_gp = -load["value"] * area_vec * _TRI_GAUSS_WEIGHT
                for i in range(6):
                    F[3 * tri[i]: 3 * tri[i] + 3] += N[i] * f_gp
    return F


def _nodal_von_mises(points, tets, D, u):
    """Nodal-averaged von Mises stress (Pa)."""
    vm_sum = np.zeros(len(points))
    vm_count = np.zeros(len(points))
    node_shapes = [_tet10_shape(*c) for c in _TET10_NODE_COORDS]

    for conn in tets:
        coords = points[conn]
        ue = u[np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 10)]
        for local, (_, dN_ref) in enumerate(node_shapes):
            J = dN_ref.T @ coords
            dN_xyz = dN_ref @ np.linalg.inv(J).T
            sigma = D @ (_b_matrix(dN_xyz) @ ue)
            sxx, syy, szz, sxy, syz, sxz = sigma
            vm = np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2
                                + (szz - sxx) ** 2)
                         + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))
            vm_sum[conn[local]] += vm
            vm_count[conn[local]] += 1
    return vm_sum / np.maximum(vm_count, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def solve(mesh_file: str, material: dict, bcs: list, loads: list,
          output_vtu: str | None = None) -> dict:
    """Solve small-displacement linear elasticity on a tet10 mesh.

    Args:
        mesh_file: path to a .msh file from ``mesh_box``.
        material:  {"E": Young's modulus [Pa], "nu": Poisson's ratio}.
        bcs:       [{"surface": <name>, "type": "fixed"}, ...]
        loads:     [{"surface": <name>, "type": "traction",
                     "vector": [tx, ty, tz]}]   (Pa) or
                   [{"surface": <name>, "type": "pressure", "value": p}]
                   (Pa, positive pushes into the surface)
        output_vtu: optional path; writes displacement and von Mises fields
                    for visualization (ParaView, pyvista).

    Returns a result summary dict (see README). All values SI.
    """
    points, tets, surfaces = _read_mesh(mesh_file)
    _validate_inputs(material, bcs, loads, surfaces)

    D = _elasticity_matrix(float(material["E"]), float(material["nu"]))
    K = _assemble_stiffness(points, tets, D)
    F = _assemble_loads(points, surfaces, loads, points.mean(axis=0))

    # Fixed supports: eliminate all three dofs of every node on the surface.
    n_dofs = 3 * len(points)
    fixed_nodes_by_surface = {
        bc["surface"]: np.unique(surfaces[bc["surface"]]) for bc in bcs}
    fixed_dofs = np.unique(np.concatenate([
        np.repeat(nodes * 3, 3) + np.tile([0, 1, 2], len(nodes))
        for nodes in fixed_nodes_by_surface.values()]))
    free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)

    u = np.zeros(n_dofs)
    u[free_dofs] = spla.spsolve(K[free_dofs][:, free_dofs], F[free_dofs])

    # Reactions: residual of the full system at the constrained dofs.
    residual = K @ u - F
    reactions = {}
    for name, nodes in fixed_nodes_by_surface.items():
        dofs = np.repeat(nodes * 3, 3) + np.tile([0, 1, 2], len(nodes))
        reactions[name] = residual[dofs].reshape(-1, 3).sum(axis=0).tolist()

    disp = u.reshape(-1, 3)
    disp_mag = np.linalg.norm(disp, axis=1)
    max_node = int(np.argmax(disp_mag))
    von_mises = _nodal_von_mises(points, tets, D, u)

    if output_vtu is not None:
        output_vtu = os.path.abspath(output_vtu)
        meshio.write(output_vtu, meshio.Mesh(
            points=points,
            cells=[("tetra10", tets)],
            point_data={"displacement": disp, "von_mises": von_mises}))

    return {
        "num_nodes": int(len(points)),
        "num_elements": int(len(tets)),
        "num_dofs": int(n_dofs),
        "max_displacement": float(disp_mag[max_node]),
        "max_displacement_vector": disp[max_node].tolist(),
        "max_von_mises": float(von_mises.max()),
        "reactions": reactions,
        "output_vtu": output_vtu,
    }
