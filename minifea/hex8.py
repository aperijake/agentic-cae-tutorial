"""Trilinear hexahedra (hex8), with and without the B-bar projection.

Two formulations of the same element, so that the difference between them can
be shown rather than asserted:

``"full"``
    Standard displacement-based hex8, 2x2x2 Gauss. As Poisson's ratio
    approaches 0.5 the dilatational constraint is imposed pointwise at every
    Gauss point, which over-constrains the element. Two symptoms appear in the
    same run: the response is far too stiff (volumetric locking), and the
    recovered pressure oscillates violently *between the Gauss points inside a
    single element* -- on Cook's membrane at nu = 0.4999 the swing within one
    element is several times the range of the whole field.

    That spatial scale matters. The oscillation is intra-element, not the
    element-to-element checkerboard associated with inf-sup-unstable mixed
    pairings, so it is invisible in any element-averaged or nodally-averaged
    plot. Use :func:`gauss_point_pressure` to see it.

``"bbar"``
    The volumetric part of the strain-displacement operator is replaced by its
    element average (Hughes 1980; the mean-dilatation form of Nagtegaal, Parks
    and Rice 1974):

        B_bar = B_dev + mean(B_vol)

    This is not a fix applied after the fact. It is what falls out of the
    three-field Hu-Washizu functional when pressure and dilatation are taken
    element-wise constant and condensed out (Simo, Taylor and Pister 1985), so
    the element still passes the patch test exactly.

B-bar addresses volumetric locking only. Hex8 also shear-locks in bending, and
Cook's membrane is bending-dominated, so a coarse B-bar mesh is still stiffer
than the converged answer. That residual is shear locking, and it needs a
different remedy (incompatible modes / enhanced assumed strain).

Conventions follow the rest of minifea: Voigt order [xx, yy, zz, xy, yz, xz]
with engineering shear strain, and SI units (meters, Pascals, Newtons).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from minifea.material import elasticity_matrix

FORMULATIONS = ("full", "bbar")

# Natural coordinates of the 8 nodes, standard VTK order: the zeta = -1 face
# counter-clockwise, then the zeta = +1 face.
_HEX8_NODE_COORDS = np.array([
    [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
])

# 2x2x2 Gauss rule, exact for trilinear integrands. Weight 1 at each point;
# they sum to 8, the volume of the reference cube [-1, 1]^3.
_G = 1.0 / np.sqrt(3.0)
_HEX_GAUSS_POINTS = np.array([[a * _G, b * _G, c * _G]
                              for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
_HEX_GAUSS_WEIGHT = 1.0

# 2x2 Gauss rule on the reference square [-1, 1]^2, for surface tractions.
_QUAD_GAUSS_POINTS = np.array([[a * _G, b * _G] for a in (-1, 1) for b in (-1, 1)])
_QUAD_GAUSS_WEIGHT = 1.0


def _hex8_shape(xi: float, eta: float, zeta: float):
    """Shape functions N (8,) and reference gradients dN (8,3) for a hex8."""
    nat = _HEX8_NODE_COORDS
    xa, ya, za = nat[:, 0], nat[:, 1], nat[:, 2]
    N = 0.125 * (1.0 + xa * xi) * (1.0 + ya * eta) * (1.0 + za * zeta)
    dN = 0.125 * np.column_stack([
        xa * (1.0 + ya * eta) * (1.0 + za * zeta),
        ya * (1.0 + xa * xi) * (1.0 + za * zeta),
        za * (1.0 + xa * xi) * (1.0 + ya * eta),
    ])
    return N, dN


def _quad4_shape(r: float, s: float):
    """Shape functions N (4,) and reference gradients dN (4,2) for a quad4."""
    nat = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    ra, sa = nat[:, 0], nat[:, 1]
    N = 0.25 * (1.0 + ra * r) * (1.0 + sa * s)
    dN = 0.25 * np.column_stack([ra * (1.0 + sa * s), sa * (1.0 + ra * r)])
    return N, dN


def _b_matrix(dN_xyz: np.ndarray, dN_bar: np.ndarray | None = None) -> np.ndarray:
    """Strain-displacement matrix B (6, 24) from shape gradients (8, 3).

    With ``dN_bar`` given, returns B_bar instead: the volumetric part of B is
    replaced by the volumetric part built from the element-averaged gradients.

    Writing the volumetric operator for node a as

        B_vol[0:3, 3a+j] = (1/3) * dN[a, j]      (zero in the shear rows)

    the substitution B_bar = B - B_vol + B_vol_bar touches only the first three
    rows, and only by the difference of the two gradients. The deviatoric part
    -- and every shear row -- is untouched.
    """
    B = np.zeros((6, 24))
    for i in range(8):
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
    if dN_bar is not None:
        correction = (dN_bar - dN_xyz) / 3.0  # (8, 3)
        for i in range(8):
            c = 3 * i
            B[0:3, c:c + 3] += correction[i]
    return B


def _element_gradients(coords: np.ndarray, elem_index: int):
    """Per-Gauss-point gradients and weights for one element.

    Returns (grads, dvols, volume, mean_gradient) where ``grads[q]`` is the
    (8, 3) physical shape gradient at Gauss point q, ``dvols[q]`` is
    detJ * w there, and ``mean_gradient`` is the volume average of the
    gradients -- the quantity the B-bar projection needs.
    """
    grads, dvols = [], []
    for gp in _HEX_GAUSS_POINTS:
        _, dN_ref = _hex8_shape(*gp)
        J = dN_ref.T @ coords  # J[a, b] = d x_b / d xi_a
        detJ = np.linalg.det(J)
        if detJ <= 0.0:
            raise ValueError(
                f"Element {elem_index} has non-positive Jacobian "
                f"({detJ:.3e}) at a Gauss point: the element is inverted or "
                "badly distorted. Check the mesh corner ordering.")
        grads.append(dN_ref @ np.linalg.inv(J).T)
        dvols.append(detJ * _HEX_GAUSS_WEIGHT)

    grads = np.array(grads)          # (8 gp, 8 node, 3)
    dvols = np.array(dvols)          # (8 gp,)
    volume = float(dvols.sum())
    mean_gradient = np.tensordot(dvols, grads, axes=(0, 0)) / volume
    return grads, dvols, volume, mean_gradient


def _assemble_stiffness(points, hexes, D, formulation):
    """Assemble the global stiffness matrix (CSR) for the given formulation."""
    n_dofs = 3 * len(points)
    rows, cols, vals = [], [], []

    for elem_index, conn in enumerate(hexes):
        coords = points[conn]
        grads, dvols, _, mean_gradient = _element_gradients(coords, elem_index)
        dN_bar = mean_gradient if formulation == "bbar" else None

        Ke = np.zeros((24, 24))
        for dN_xyz, dvol in zip(grads, dvols):
            B = _b_matrix(dN_xyz, dN_bar)
            Ke += (B.T @ D @ B) * dvol

        dofs = np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 8)
        rows.append(np.repeat(dofs, 24))
        cols.append(np.tile(dofs, 24))
        vals.append(Ke.ravel())

    K = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_dofs, n_dofs))
    return K.tocsr()


def _assemble_tractions(points, face_sets, tractions):
    """Consistent nodal forces from tractions on quad faces."""
    F = np.zeros(3 * len(points))
    gauss = [_quad4_shape(*gp) for gp in _QUAD_GAUSS_POINTS]

    for load in tractions:
        name = load.get("face_set")
        if name not in face_sets:
            raise ValueError(
                f"Traction refers to unknown face set {name!r}. Available "
                f"face sets in this mesh: {sorted(face_sets)}.")
        quads = face_sets[name]

        if ("vector" in load) == ("total_force" in load):
            raise ValueError(
                f"Traction on {name!r} needs exactly one of 'vector' "
                "(traction in Pa) or 'total_force' (resultant in N).")

        if "vector" in load:
            t = np.asarray(load["vector"], dtype=float)
        else:
            area = 0.0
            for quad in quads:
                c = points[quad]
                for _, dN in gauss:
                    area += np.linalg.norm(
                        np.cross(dN[:, 0] @ c, dN[:, 1] @ c)) * _QUAD_GAUSS_WEIGHT
            t = np.asarray(load["total_force"], dtype=float) / area
        if t.shape != (3,):
            raise ValueError(
                f"Traction on {name!r} must be a 3-vector, got {t.shape}.")

        for quad in quads:
            c = points[quad]
            for N, dN in gauss:
                dA = np.linalg.norm(
                    np.cross(dN[:, 0] @ c, dN[:, 1] @ c)) * _QUAD_GAUSS_WEIGHT
                for i in range(4):
                    F[3 * quad[i]: 3 * quad[i] + 3] += N[i] * t * dA
    return F


def _element_fields(points, hexes, D, u, formulation):
    """Element-averaged pressure [Pa] and von Mises stress [Pa].

    Pressure is p = -tr(sigma)/3, positive in compression.

    Averaging is convenient for reporting but destroys the locking signature:
    the pressure oscillation under full integration is *within* each element,
    so averaging over an element's Gauss points cancels it almost entirely.
    On Cook's membrane at nu = 0.4999 the averaged field looks smooth while
    the underlying Gauss-point field swings by several times its range.
    :func:`gauss_point_pressure` is what the locking demonstration plots.
    """
    pressure = np.zeros(len(hexes))
    von_mises = np.zeros(len(hexes))

    for elem_index, conn in enumerate(hexes):
        coords = points[conn]
        grads, dvols, volume, mean_gradient = _element_gradients(coords, elem_index)
        dN_bar = mean_gradient if formulation == "bbar" else None
        ue = u[np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 8)]

        p_acc = 0.0
        vm_acc = 0.0
        for dN_xyz, dvol in zip(grads, dvols):
            sxx, syy, szz, sxy, syz, sxz = D @ (_b_matrix(dN_xyz, dN_bar) @ ue)
            p_acc += -(sxx + syy + szz) / 3.0 * dvol
            vm_acc += np.sqrt(
                0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)) * dvol
        pressure[elem_index] = p_acc / volume
        von_mises[elem_index] = vm_acc / volume

    return pressure, von_mises


def solve_hex(mesh: dict, material: dict, fixed: list, tractions: list = (),
              formulation: str = "bbar") -> dict:
    """Solve small-displacement linear elasticity on a hex8 mesh.

    Args:
        mesh: dict from ``minifea.cooks.bilinear_hex_mesh`` (or ``cooks_membrane``).
        material: {"E": Young's modulus [Pa], "nu": Poisson's ratio}.
        fixed: [{"node_set": <name>, "components": (0, 1, 2)}, ...] -- the
            listed displacement components are held at zero. Plane strain is
            imposed by fixing component 2 on the two out-of-plane faces.
        tractions: [{"face_set": <name>, "vector": [tx, ty, tz]}] in Pa, or
            [{"face_set": <name>, "total_force": [Fx, Fy, Fz]}] in N, which is
            divided by the face area.
        formulation: "bbar" (mean-dilatation B-bar) or "full" (standard
            displacement-based hex8).

    Returns a result dict with displacement (n_nodes, 3), element pressure and
    von Mises fields, the reaction resultant, and the strain energy. All SI.

    Note that ``nu`` may be taken right up to 0.5 here: this is where the two
    formulations diverge, and refusing near-incompressible input would remove
    the phenomenon being demonstrated. The solve is still non-singular for
    nu < 0.5.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(
            f"Unknown formulation {formulation!r}; use one of {list(FORMULATIONS)}.")
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

    points, hexes = mesh["points"], mesh["hexes"]
    node_sets, face_sets = mesh["node_sets"], mesh["face_sets"]
    n_dofs = 3 * len(points)

    D = elasticity_matrix(E, nu)
    K = _assemble_stiffness(points, hexes, D, formulation)
    F = _assemble_tractions(points, face_sets, tractions)

    fixed_dof_list = []
    for entry in fixed:
        name = entry.get("node_set")
        if name not in node_sets:
            raise ValueError(
                f"Constraint refers to unknown node set {name!r}. Available "
                f"node sets in this mesh: {sorted(node_sets)}.")
        components = entry.get("components", (0, 1, 2))
        nodes = node_sets[name]
        for comp in components:
            if comp not in (0, 1, 2):
                raise ValueError(
                    f"Constraint components must be 0, 1 or 2, got {comp!r}.")
            fixed_dof_list.append(nodes * 3 + comp)

    fixed_dofs = np.unique(np.concatenate(fixed_dof_list))
    free_dofs = np.setdiff1d(np.arange(n_dofs), fixed_dofs)

    u = np.zeros(n_dofs)
    u[free_dofs] = spla.spsolve(K[free_dofs][:, free_dofs], F[free_dofs])

    # Reaction resultant, summed by component. The constrained dofs cannot be
    # reshaped into node triples: the plane-strain constraints fix only the
    # z component, so the fixed-dof list is not a whole number of nodes.
    residual = K @ u - F
    reaction = np.zeros(3)
    np.add.at(reaction, fixed_dofs % 3, residual[fixed_dofs])
    pressure, von_mises = _element_fields(points, hexes, D, u, formulation)
    disp = u.reshape(-1, 3)

    return {
        "formulation": formulation,
        "displacement": disp,
        "pressure": pressure,
        "von_mises": von_mises,
        "num_nodes": int(len(points)),
        "num_elements": int(len(hexes)),
        "num_dofs": int(n_dofs),
        "max_displacement": float(np.linalg.norm(disp, axis=1).max()),
        "strain_energy": float(0.5 * u @ (K @ u)),
        "reaction": reaction.tolist(),
    }


def probe(mesh: dict, field: np.ndarray, xy: tuple) -> np.ndarray:
    """Value of a nodal ``field`` at the mesh node nearest to ``xy`` [m].

    Used to pull out a single quantity of interest -- for Cook's membrane, the
    vertical displacement of the top-right corner.
    """
    points = mesh["points"]
    distances = np.linalg.norm(points[:, :2] - np.asarray(xy, float), axis=1)
    return field[int(np.argmin(distances))]


def gauss_point_pressure(mesh: dict, material: dict, displacement: np.ndarray,
                         formulation: str) -> tuple:
    """Pressure sampled at every Gauss point, with its physical location.

    Returns ``(coordinates, pressure)`` where ``coordinates`` is
    (n_elements * 8, 3) [m] and ``pressure`` is (n_elements * 8,) [Pa],
    positive in compression.

    This is the unaveraged field. It is what makes volumetric locking visible:
    under full integration the value flips sign between neighboring Gauss
    points inside a single element, and any averaging -- onto elements or onto
    nodes -- hides that almost completely.
    """
    if formulation not in FORMULATIONS:
        raise ValueError(
            f"Unknown formulation {formulation!r}; use one of {list(FORMULATIONS)}.")
    D = elasticity_matrix(float(material["E"]), float(material["nu"]))
    points, hexes = mesh["points"], mesh["hexes"]
    u = np.asarray(displacement, dtype=float).ravel()
    shapes = [_hex8_shape(*gp)[0] for gp in _HEX_GAUSS_POINTS]

    coordinates = np.empty((len(hexes) * 8, 3))
    pressure = np.empty(len(hexes) * 8)
    for elem_index, conn in enumerate(hexes):
        coords = points[conn]
        grads, _, _, mean_gradient = _element_gradients(coords, elem_index)
        dN_bar = mean_gradient if formulation == "bbar" else None
        ue = u[np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 8)]
        for q, (N, dN_xyz) in enumerate(zip(shapes, grads)):
            sigma = D @ (_b_matrix(dN_xyz, dN_bar) @ ue)
            row = elem_index * 8 + q
            coordinates[row] = N @ coords
            pressure[row] = -(sigma[0] + sigma[1] + sigma[2]) / 3.0
    return coordinates, pressure
