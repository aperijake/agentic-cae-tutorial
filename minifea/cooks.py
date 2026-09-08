"""Structured hexahedral meshing for Cook's membrane.

Cook's membrane is a tapered panel, clamped on its left edge and loaded in
shear on its right edge. It is the standard benchmark for volumetric locking:
bending-dominated, and at nu -> 0.5 a displacement-based low-order element
gets the answer badly wrong while producing a plot that looks fine.

There is no mesher dependency here on purpose. The panel is the bilinear image
of a unit square, so the mesh is a few lines of numpy -- which keeps the whole
tutorial chain readable end to end and keeps the install to numpy/scipy.

SI units: meters, Pascals, Newtons. The classical benchmark is stated in
dimensionless units of 48 x 44 x 60; here that is read as millimeters and
converted, so the panel spans 0.048 m in x.
"""

import numpy as np

# Cook's membrane corners in the x-y plane, counter-clockwise from the origin,
# in meters. Classical benchmark values (48, 44, 60) read as millimeters.
COOKS_CORNERS = np.array([
    [0.000, 0.000],   # lower-left  (clamped edge, bottom)
    [0.048, 0.044],   # lower-right (loaded edge, bottom)
    [0.048, 0.060],   # upper-right (loaded edge, top)
    [0.000, 0.044],   # upper-left  (clamped edge, top)
])

COOKS_THICKNESS = 0.001  # m, out-of-plane


def bilinear_hex_mesh(corners: np.ndarray, thickness: float,
                      nx: int, ny: int, nz: int = 1) -> dict:
    """Mesh the bilinear quadrilateral ``corners``, extruded by ``thickness``.

    The quadrilateral is the image of the unit square under bilinear
    interpolation of its four corners, so any straight-sided (possibly
    tapered) panel can be meshed exactly. Elements are 8-node hexahedra in
    the standard VTK node order.

    Args:
        corners: (4, 2) x-y coordinates, counter-clockwise, starting at the
            corner that maps to (xi, eta) = (0, 0).
        thickness: out-of-plane extent [m], extruded in +z from z = 0.
        nx, ny, nz: element counts along xi, eta and z.

    Returns a dict with:
        points:   (n_nodes, 3) node coordinates [m]
        hexes:    (n_elems, 8) node indices
        node_sets: name -> node index array, for
            "clamped" (xi = 0 face), "loaded" (xi = 1 face),
            "zmin", "zmax" (the two out-of-plane faces)
        face_sets: name -> (n_faces, 4) quad connectivity for "clamped" and
            "loaded", used to integrate surface tractions consistently
        loaded_area: area of the "loaded" face [m^2], for converting a total
            force into a traction

    Raises:
        ValueError: if any element count is not a positive integer, if the
            thickness is not positive, or if the corners are ordered such that
            the quadrilateral is inverted.
    """
    corners = np.asarray(corners, dtype=float)
    if corners.shape != (4, 2):
        raise ValueError(
            f"corners must be a (4, 2) array of x-y coordinates, got shape "
            f"{corners.shape}.")
    for name, value in [("nx", nx), ("ny", ny), ("nz", nz)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if not thickness > 0:
        raise ValueError(f"thickness must be positive, got {thickness!r}.")

    # Signed area (shoelace) catches clockwise / self-intersecting corner
    # orders, which would produce inverted elements and a negative Jacobian
    # much later in the solve with a far less obvious error message.
    x, y = corners[:, 0], corners[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area <= 0.0:
        raise ValueError(
            f"corners trace a clockwise or degenerate quadrilateral (signed "
            f"area {signed_area:.3e} m^2); list them counter-clockwise.")

    xi = np.linspace(0.0, 1.0, nx + 1)
    eta = np.linspace(0.0, 1.0, ny + 1)
    zeta = np.linspace(0.0, thickness, nz + 1)
    XI, ETA, Z = np.meshgrid(xi, eta, zeta, indexing="ij")

    # Bilinear map of the unit square onto the quadrilateral.
    w00 = (1.0 - XI) * (1.0 - ETA)
    w10 = XI * (1.0 - ETA)
    w11 = XI * ETA
    w01 = (1.0 - XI) * ETA
    X = (w00 * corners[0, 0] + w10 * corners[1, 0]
         + w11 * corners[2, 0] + w01 * corners[3, 0])
    Y = (w00 * corners[0, 1] + w10 * corners[1, 1]
         + w11 * corners[2, 1] + w01 * corners[3, 1])

    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Node numbering is C-order over (i, j, k) -- the same order meshgrid
    # produced above -- so a node index is recoverable arithmetically.
    def node_id(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    i0, j0, k0 = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz),
                             indexing="ij")
    i0, j0, k0 = i0.ravel(), j0.ravel(), k0.ravel()
    hexes = np.column_stack([
        node_id(i0, j0, k0), node_id(i0 + 1, j0, k0),
        node_id(i0 + 1, j0 + 1, k0), node_id(i0, j0 + 1, k0),
        node_id(i0, j0, k0 + 1), node_id(i0 + 1, j0, k0 + 1),
        node_id(i0 + 1, j0 + 1, k0 + 1), node_id(i0, j0 + 1, k0 + 1),
    ])

    all_j, all_k = np.meshgrid(np.arange(ny + 1), np.arange(nz + 1),
                               indexing="ij")
    all_i, all_k2 = np.meshgrid(np.arange(nx + 1), np.arange(nz + 1),
                                indexing="ij")
    all_i2, all_j2 = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1),
                                 indexing="ij")
    node_sets = {
        "clamped": node_id(0, all_j.ravel(), all_k.ravel()),
        "loaded": node_id(nx, all_j.ravel(), all_k.ravel()),
        "zmin": node_id(all_i2.ravel(), all_j2.ravel(), 0),
        "zmax": node_id(all_i2.ravel(), all_j2.ravel(), nz),
    }
    del all_i, all_k2  # the xi-constant faces are the only ones we name

    # Quad connectivity on the two xi-constant faces. Tractions have to be
    # integrated over these, not lumped onto the node set: a node set knows
    # which nodes are on a face but nothing about how much area each one
    # carries, and for a tapered panel that share is not uniform.
    fj, fk = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    fj, fk = fj.ravel(), fk.ravel()
    face_sets = {}
    for name, i_face in [("clamped", 0), ("loaded", nx)]:
        face_sets[name] = np.column_stack([
            node_id(i_face, fj, fk), node_id(i_face, fj + 1, fk),
            node_id(i_face, fj + 1, fk + 1), node_id(i_face, fj, fk + 1),
        ])

    loaded_edge_length = np.linalg.norm(corners[2] - corners[1])

    return {
        "points": points,
        "hexes": hexes,
        "node_sets": node_sets,
        "face_sets": face_sets,
        "loaded_area": float(loaded_edge_length * thickness),
        "num_nodes": int(len(points)),
        "num_elements": int(len(hexes)),
    }


def cooks_membrane(n: int, nz: int = 1) -> dict:
    """Cook's membrane meshed with ``n`` x ``n`` hexahedra in plane.

    Convenience wrapper around :func:`bilinear_hex_mesh` with the benchmark
    geometry. ``nz`` is normally 1: the problem is plane strain, enforced by
    fixing z-displacement on the "zmin" and "zmax" node sets.
    """
    return bilinear_hex_mesh(COOKS_CORNERS, COOKS_THICKNESS, n, n, nz)


# Kuhn decomposition of a hex into 6 tets, sharing the body diagonal 0-6.
# The other two nodes of each tet are consecutive vertices of the equatorial
# 6-cycle 1 -> 2 -> 3 -> 7 -> 4 -> 5 -> 1 around that diagonal.
#
# Using the same local pattern in every hex is what keeps the result
# conforming: on the face shared by two neighbors, the diagonal each one
# draws lands on the same pair of physical nodes. Alternating the pattern (a
# common "improvement") breaks that and leaves cracks between the tets.
_HEX_TO_TET = np.array([
    [0, 6, 1, 2], [0, 6, 2, 3], [0, 6, 3, 7],
    [0, 6, 7, 4], [0, 6, 4, 5], [0, 6, 5, 1],
])

# Each quad face splits along the diagonal already implied by _HEX_TO_TET.
_QUAD_TO_TRI = np.array([[0, 1, 2], [0, 2, 3]])


def hex_to_tet4(mesh: dict) -> dict:
    """Split every hexahedron into 6 linear tetrahedra.

    Returns a mesh dict with ``tets`` (n_elems * 6, 4) in place of ``hexes``,
    the same ``points`` and ``node_sets``, and ``face_sets`` retriangulated
    into (n_faces * 2, 3) connectivity.

    Tet4 is the element an automated workflow reaches for by default, and it
    is the worst volumetric locker of the lot: constant strain means one
    pressure value per element, so the spurious pressure mode has nowhere to
    live except between elements. That is what makes the checkerboard visible
    on a tet mesh and invisible on a hex mesh, where the same pathology hides
    inside each element instead.
    """
    if "hexes" not in mesh:
        raise ValueError(
            "hex_to_tet4 expects a hex mesh with a 'hexes' key; got keys "
            f"{sorted(mesh)}.")

    hexes = mesh["hexes"]
    tets = hexes[:, _HEX_TO_TET.ravel()].reshape(-1, 4)

    # Orientation: a negative-volume tet would fail in assembly with a much
    # less obvious message, so fix it here where the cause is visible.
    points = mesh["points"]
    edges = points[tets[:, 1:]] - points[tets[:, [0]]]
    volumes = np.linalg.det(edges) / 6.0
    flipped = volumes < 0.0
    tets[flipped] = tets[flipped][:, [0, 1, 3, 2]]

    face_sets = {name: quads[:, _QUAD_TO_TRI.ravel()].reshape(-1, 3)
                 for name, quads in mesh["face_sets"].items()}

    out = dict(mesh)
    out.pop("hexes")
    out["tets"] = tets
    out["face_sets"] = face_sets
    out["num_elements"] = int(len(tets))
    return out
