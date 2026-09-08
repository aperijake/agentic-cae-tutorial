"""Generate the Cook's membrane locking figures.

Nearly incompressible (nu = 0.4999), one mesh, three element formulations.
The CoFEA reference band for the tip displacement is 27.3-28.0 mm.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from minifea import cooks_membrane, hex_to_tet4, probe, solve_hex, solve_tet4
from minifea.plotting import plot_pressure

FIXED = [{"node_set": "clamped", "components": (0, 1, 2)},
         {"node_set": "zmin", "components": (2,)},
         {"node_set": "zmax", "components": (2,)}]
LOAD = [{"face_set": "loaded", "total_force": [0.0, 100.0, 0.0]}]
MATERIAL = {"E": 70e6, "nu": 0.4999}
TIP = (0.048, 0.060)
N = 8

hex_mesh = cooks_membrane(N)
tet_mesh = hex_to_tet4(hex_mesh)
cases = [
    ("tet4", tet_mesh, solve_tet4(tet_mesh, MATERIAL, FIXED, LOAD), {}),
    ("hex8, full integration", hex_mesh,
     solve_hex(hex_mesh, MATERIAL, FIXED, LOAD, formulation="full"),
     {"material": MATERIAL}),
    ("hex8, B-bar", hex_mesh,
     solve_hex(hex_mesh, MATERIAL, FIXED, LOAD, formulation="bbar"),
     {"material": MATERIAL}),
]


def shared_limits(cases, pad=0.06):
    """One set of axis limits for every panel, so the shapes are comparable."""
    xy = np.vstack([(mesh["points"] + result["displacement"])[:, :2] * 1000.0
                    for _, mesh, result, _ in cases])
    (x0, y0), (x1, y1) = xy.min(axis=0), xy.max(axis=0)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - mx, x1 + mx), (y0 - my, y1 + my)


xlim, ylim = shared_limits(cases)

# Figure 1 -- the three formulations. Each panel carries its own color scale:
# the spurious pressures in the locked solutions are an order of magnitude
# larger than the physical ones, so a shared scale would render B-bar blank.
fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
for ax, (title, mesh, result, extra) in zip(axes, cases):
    plot_pressure(mesh, result, ax, **extra)
    tip = probe(mesh, result["displacement"][:, 1], TIP) * 1000.0
    ax.set_title(f"{title}\ntip = {tip:.1f} mm", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
fig.suptitle("Cook's membrane, nu = 0.4999, same mesh throughout "
             "(reference tip displacement 27.3-28.0 mm)", fontsize=12)
fig.tight_layout()
fig.savefig("figures/cooks_three_formulations.png", dpi=140, bbox_inches="tight")

# Figure 2 -- one run, two plots. Averaging is what hides the pathology.
_, mesh, full, _ = cases[1]
fig2, axes2 = plt.subplots(1, 2, figsize=(9.5, 5.0))
plot_pressure(mesh, full, axes2[0], sampling="element")
plot_pressure(mesh, full, axes2[1], sampling="gauss", material=MATERIAL)
axes2[0].set_title("element-averaged\n(the plot you would normally make)")
axes2[1].set_title("unaveraged, at the Gauss points\n(the same solution)")
for ax in axes2:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
fig2.tight_layout()
fig2.savefig("figures/cooks_averaging_hides_it.png", dpi=140, bbox_inches="tight")

# Figures 3 and 4 -- what gets sent to a vision model. Field and colorbar
# only: no title, no formulation name, nothing that gives away the diagnosis.
#
# The filenames are deliberately neutral too. A first pass at this used
# "cooks_for_vision.png", and the model reviewing it reported that it had read
# the benchmark name off the path before opening the image. Keeping the answer
# out of the frame is not enough if it is still in the filename.
fig3, ax3 = plt.subplots(figsize=(4.8, 5.2))
plot_pressure(tet_mesh, cases[0][2], ax3)
fig3.tight_layout()
fig3.savefig("figures/pressure_field_01.png", dpi=150, bbox_inches="tight")

# The cropped view is the harder test: no domain boundary in frame, so the
# geometry cannot be recognized and only the pattern itself is left to read.
# Cook's membrane is *the* canonical locking benchmark, so a model shown the
# whole panel may be recalling what that silhouette is famous for rather than
# reading the picture. This one removes that possibility.
fine_mesh = hex_to_tet4(cooks_membrane(16))
fine = solve_tet4(fine_mesh, MATERIAL, FIXED, LOAD)
fig4, ax4 = plt.subplots(figsize=(5.0, 5.0))
plot_pressure(fine_mesh, fine, ax4)
ax4.set_xlim(8, 23)   # a window entirely interior to the deformed panel,
ax4.set_ylim(28, 46)  # checked against the deformed top and bottom edges
fig4.tight_layout()
fig4.savefig("figures/pressure_field_02.png", dpi=150, bbox_inches="tight")

for title, mesh, result, _ in cases:
    tip = probe(mesh, result["displacement"][:, 1], TIP) * 1000.0
    p = result["pressure"] / 1e6
    print(f"{title:24s} tip = {tip:6.2f} mm    element pressure "
          f"{p.min():8.2f} to {p.max():7.2f} MPa")
print("wrote figures/")
