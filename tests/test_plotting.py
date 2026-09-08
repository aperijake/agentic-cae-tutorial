"""Checks on the pressure plots.

Mostly guarding the two properties the demonstration depends on: that nothing
gets averaged or interpolated on its way into the picture, and that the image
sent to a vision model does not carry the answer in its own annotations.
"""

import matplotlib
matplotlib.use("Agg")

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import LineCollection

from minifea import cooks_membrane, hex_to_tet4, solve_hex, solve_tet4
from minifea.plotting import _symmetric_limits, plot_pressure

FIXED = [{"node_set": "clamped", "components": (0, 1, 2)},
         {"node_set": "zmin", "components": (2,)},
         {"node_set": "zmax", "components": (2,)}]
LOAD = [{"face_set": "loaded", "total_force": [0.0, 100.0, 0.0]}]
MATERIAL = {"E": 70e6, "nu": 0.4999}
N = 5


@pytest.fixture
def hex_case():
    mesh = cooks_membrane(N)
    return mesh, solve_hex(mesh, MATERIAL, FIXED, LOAD, formulation="full")


@pytest.fixture
def tet_case():
    mesh = hex_to_tet4(cooks_membrane(N))
    return mesh, solve_tet4(mesh, MATERIAL, FIXED, LOAD)


def polygons(ax):
    return ax.collections[0].get_paths()


def _field(ax):
    return ax.collections[0]


def values(ax):
    return ax.collections[0].get_array()


def test_tet_surface_has_two_triangles_per_hex(tet_case):
    mesh, result = tet_case
    ax = plot_pressure(mesh, result)
    assert len(polygons(ax)) == 2 * N * N
    plt.close(ax.figure)


def test_gauss_sampling_gives_four_subcells_per_hex(hex_case):
    """Four in-plane Gauss points per element, each shaded flat and separately.

    This is the resolution that makes the intra-element oscillation visible;
    one value per element would average it away.
    """
    mesh, result = hex_case
    ax = plot_pressure(mesh, result, sampling="gauss", material=MATERIAL)
    assert len(polygons(ax)) == 4 * N * N
    plt.close(ax.figure)


def test_gauss_sampling_shows_far_more_spread_than_element_averaging(hex_case):
    """The averaged plot really does hide it -- asserted, not just claimed."""
    mesh, result = hex_case
    averaged = plot_pressure(mesh, result, sampling="element")
    unaveraged = plot_pressure(mesh, result, sampling="gauss", material=MATERIAL)
    assert np.ptp(values(unaveraged)) > 5.0 * np.ptp(values(averaged))
    plt.close(averaged.figure)
    plt.close(unaveraged.figure)


def test_values_are_plotted_flat_and_unmodified(tet_case):
    """Every plotted value is one the solver produced, not a blend of them."""
    mesh, result = tet_case
    ax = plot_pressure(mesh, result)
    plotted = np.sort(np.unique(np.round(values(ax), 9)))
    solved = np.sort(np.unique(np.round(result["pressure"] / 1e6, 9)))
    assert np.isin(plotted, solved).all()
    plt.close(ax.figure)


def test_colour_limits_ignore_the_corner_singularity():
    """Percentile limits must not be set by a couple of extreme elements."""
    field = np.concatenate([np.random.default_rng(0).normal(0, 1, 500),
                            [500.0, -500.0]])
    low, high = _symmetric_limits(field)
    assert high < 10.0 and low == -high


def test_vision_image_carries_no_diagnosis(tet_case):
    """The frame must not give the answer away.

    A title naming the element, or an axis label mentioning locking, turns the
    vision test into a reading test. Only the field and its colorbar go in.
    """
    mesh, result = tet_case
    ax = plot_pressure(mesh, result)
    text = " ".join([ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
                    + [t.get_text() for t in ax.figure.findobj(plt.Text)]).lower()
    for banned in ("tet", "hex", "b-bar", "bbar", "lock", "checker",
                   "integration", "incompressible", "nu ="):
        assert banned not in text, f"the plot leaks {banned!r} to the model"
    plt.close(ax.figure)


def test_rejects_bad_sampling(hex_case, tet_case):
    mesh, result = hex_case
    with pytest.raises(ValueError, match="sampling must be"):
        plot_pressure(mesh, result, sampling="nodal")
    with pytest.raises(ValueError, match="needs the material"):
        plot_pressure(mesh, result, sampling="gauss")
    tet_mesh, tet_result = tet_case
    with pytest.raises(ValueError, match="only meaningful for hex"):
        plot_pressure(tet_mesh, tet_result, sampling="gauss", material=MATERIAL)


def test_undeformed_outline_is_drawn_and_closed(tet_case):
    """The reference outline must be present, and be a closed loop.

    Both blind reviews of an earlier figure flagged that they could not tell a
    large deflection from a different geometry, because the deformed shape was
    drawn with no reference. An open or partial outline would be worse than
    none, so check that every boundary node appears in exactly two segments.
    """
    mesh, result = tet_case
    ax = plot_pressure(mesh, result)
    outlines = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(outlines) == 1, "expected exactly one undeformed outline"

    segments = outlines[0].get_segments()
    assert len(segments) > 0
    endpoints = Counter(tuple(np.round(p, 9)) for s in segments for p in s)
    assert set(endpoints.values()) == {2}, "outline is not a closed loop"

    # It must be the *undeformed* shape, so it cannot coincide with the
    # deformed field it is meant to be a reference for.
    drawn = np.vstack(segments)
    moved = (mesh["points"] + result["displacement"])[:, :2] * 1000.0
    assert np.abs(drawn[:, 1].max() - moved[:, 1].max()) > 1.0
    plt.close(ax.figure)


def test_undeformed_outline_can_be_turned_off(tet_case):
    mesh, result = tet_case
    ax = plot_pressure(mesh, result, show_undeformed=False)
    assert not [c for c in ax.collections if isinstance(c, LineCollection)]
    plt.close(ax.figure)
