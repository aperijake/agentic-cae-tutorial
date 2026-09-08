"""Flat-shaded pressure plots for the locking demonstration.

matplotlib only -- no VTK, no trame. That is a correctness choice before it is
a convenience one. The thing being visualised is a per-element (or per-Gauss-
point) field with genuine discontinuities between neighbors, and most 3D
viewers average field values onto shared nodes by default. Averaging a
checkerboard cancels it: the pathology disappears from the picture that was
supposed to reveal it. Everything here shades each cell flat, with no
interpolation anywhere.

Plots are drawn on the z-minimum surface, which for a plane-strain panel one
element thick is the whole story. Lengths are shown in mm and pressures in
MPa, since the benchmark is quoted that way; the solver itself stays in SI.
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from minifea.hex8 import _HEX_GAUSS_POINTS, gauss_point_pressure

# Gauss points on the zeta = -1 face, in the (xi, eta) quadrant order
# (-,-), (+,-), (+,+), (-,+) that matches the face node ordering below.
_FACE_GAUSS = [0, 4, 6, 2]

# Sub-cell corners in natural coordinates, one quadrant per face Gauss point.
_SUBCELL_BOUNDS = [(-1.0, 0.0, -1.0, 0.0), (0.0, 1.0, -1.0, 0.0),
                   (0.0, 1.0, 0.0, 1.0), (-1.0, 0.0, 0.0, 1.0)]


def _bilinear(face_xy, xi, eta):
    """Point at natural coordinates (xi, eta) in [-1, 1]^2 on a quad face."""
    weights = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                               (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    return weights @ face_xy


def _zmin_polygons(mesh):
    """Polygons on the z-minimum surface, with their parent element index.

    Returns (node_indices, parent_elements): a list of node-index arrays and
    the matching element index for each, so any per-element field can be
    mapped straight onto the picture.
    """
    points = mesh["points"]
    z_min = points[:, 2].min()
    on_face = np.isclose(points[:, 2], z_min)

    polygons, parents = [], []
    if "tets" in mesh:
        # A tet contributes a face here only if three of its nodes lie on the
        # surface; the Kuhn decomposition gives two such triangles per hex.
        for elem_index, tet in enumerate(mesh["tets"]):
            face = tet[on_face[tet]]
            if len(face) == 3:
                polygons.append(face)
                parents.append(elem_index)
    else:
        for elem_index, hexa in enumerate(mesh["hexes"]):
            face = hexa[:4]  # the zeta = -1 face, in order
            if on_face[face].all():
                polygons.append(face)
                parents.append(elem_index)
    return polygons, np.array(parents, dtype=int)


def _surface_outline(mesh, polygons):
    """Boundary edges of the z-minimum surface, as node-index pairs.

    An edge on the outline belongs to exactly one surface polygon; interior
    edges are shared by two. Derived from the mesh rather than from the known
    corner coordinates, so it stays correct for any panel the mesher produces.
    """
    counts = {}
    for nodes in polygons:
        for i in range(len(nodes)):
            edge = (nodes[i], nodes[(i + 1) % len(nodes)])
            counts[tuple(sorted(edge))] = counts.get(tuple(sorted(edge)), 0) + 1
    return [edge for edge, count in counts.items() if count == 1]


def _symmetric_limits(values, percentile=98.0):
    """Color limits clipped to a percentile, centered on zero.

    The re-entrant corner of Cook's membrane is a stress singularity, so the
    extreme values live in a couple of elements next to it. Scaling the color
    map to the true min and max lets those few elements set the range and
    washes the rest of the field to a uniform pale color -- which hides
    exactly the pattern the plot exists to show.
    """
    limit = np.percentile(np.abs(values), percentile)
    if limit <= 0.0:
        limit = max(np.abs(values).max(), 1.0)
    return -limit, limit


def plot_pressure(mesh: dict, result: dict, ax=None, *, sampling: str = "auto",
                  material: dict | None = None, deformed: bool = True,
                  scale: float = 1.0, clim: tuple | None = None,
                  cmap: str = "RdBu_r", show_edges: bool = True,
                  show_undeformed: bool = True, colorbar: bool = True,
                  label: str | None = None):
    """Flat-shaded pressure on the z-minimum surface.

    Args:
        mesh: hex or tet mesh dict.
        result: output of ``solve_hex`` or ``solve_tet4``.
        sampling: "element" for the per-element field, "gauss" for the
            unaveraged Gauss-point field (hex meshes only -- the sub-element
            oscillation is invisible in the averaged one), or "auto", which
            picks "gauss" for hex meshes and "element" for tets.
        material: required for ``sampling="gauss"``; the same dict passed to
            the solver.
        deformed: draw on the deformed shape, magnified by ``scale``.
        show_undeformed: outline the original shape faintly behind it. On by
            default, and worth keeping: without it a plot of a deformed panel
            is ambiguous about which configuration it shows, and a reader has
            no way to tell a large deflection from a different geometry.
        clim: color limits. Defaults to symmetric 98th-percentile limits.
        label: colorbar label. Deliberately not a title -- see
            :func:`save_for_vision`.

    Returns the matplotlib Axes.
    """
    if sampling == "auto":
        sampling = "element" if "tets" in mesh else "gauss"
    if sampling not in ("element", "gauss"):
        raise ValueError(
            f"sampling must be 'element', 'gauss' or 'auto', got {sampling!r}.")
    if sampling == "gauss" and "tets" in mesh:
        raise ValueError(
            "Gauss-point sampling is only meaningful for hex meshes. A tet4 "
            "element has one constant-strain point, so its element field is "
            "already unaveraged; use sampling='element'.")
    if sampling == "gauss" and material is None:
        raise ValueError(
            "sampling='gauss' needs the material dict that was passed to the "
            "solver, to recompute stress at each Gauss point.")

    points = mesh["points"].copy()
    if deformed:
        points = points + result["displacement"] * scale
    xy = points[:, :2] * 1000.0  # mm

    if sampling == "element":
        node_polygons, parents = _zmin_polygons(mesh)
        vertices = [xy[nodes] for nodes in node_polygons]
        values = result["pressure"][parents]
    else:
        _, pressure = gauss_point_pressure(
            mesh, material, result["displacement"], result["formulation"])
        vertices, values = [], []
        for elem_index, hexa in enumerate(mesh["hexes"]):
            face = hexa[:4]
            if not np.isclose(points[face, 2], points[:, 2].min()).all():
                continue
            face_xy = xy[face]
            for quadrant, (x0, x1, y0, y1) in enumerate(_SUBCELL_BOUNDS):
                vertices.append(np.array([
                    _bilinear(face_xy, x0, y0), _bilinear(face_xy, x1, y0),
                    _bilinear(face_xy, x1, y1), _bilinear(face_xy, x0, y1)]))
                values.append(pressure[elem_index * 8 + _FACE_GAUSS[quadrant]])
        values = np.array(values)

    values = values / 1e6  # MPa
    if clim is None:
        clim = _symmetric_limits(values)

    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.6))
    collection = PolyCollection(
        vertices, array=values, cmap=cmap, clim=clim,
        edgecolors="0.25" if show_edges else "none",
        linewidths=0.3 if show_edges else 0.0, zorder=2)
    ax.add_collection(collection)

    if deformed and show_undeformed:
        reference = mesh["points"][:, :2] * 1000.0
        surface, _ = _zmin_polygons(mesh)
        segments = [reference[list(edge)] for edge in _surface_outline(mesh, surface)]
        ax.add_collection(LineCollection(
            segments, colors="0.55", linewidths=0.9, linestyles=(0, (4, 3)),
            zorder=3))

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if colorbar:
        bar = ax.figure.colorbar(collection, ax=ax, shrink=0.82, pad=0.03)
        bar.set_label(label or "pressure (MPa)")
    return ax
