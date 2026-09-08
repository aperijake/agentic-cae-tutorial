"""minifea: a deliberately small 3D finite element toolkit for teaching agentic workflows.

Small-displacement linear elasticity, SI units throughout. Small enough to
read in one sitting -- and small enough for an AI agent to extend.

Everything here -- meshing, hex8 with and without B-bar, tet4, the unit
consistency pipeline, plots -- needs only numpy, scipy and matplotlib.

The older gmsh/meshio tet10 toolchain now lives in ``extras/tet10`` and is
installed separately, so that a clean install of this package cannot fail on
a mesher the core material never uses.
"""

from minifea.cooks import bilinear_hex_mesh, cooks_membrane, hex_to_tet4
from minifea.hex8 import gauss_point_pressure, probe, solve_hex
from minifea.material import elasticity_matrix
from minifea.tet4 import solve_tet4

__all__ = [
    "bilinear_hex_mesh", "cooks_membrane", "hex_to_tet4",
    "solve_hex", "solve_tet4", "gauss_point_pressure", "probe",
    "elasticity_matrix",
]


def __getattr__(name):
    """Point at the tet10 path, which used to live here."""
    if name in ("mesh_box", "solve", "TOOLS", "execute_tool"):
        raise AttributeError(
            f"minifea.{name} moved to the tet10 package in extras/. Install it "
            "with: pip install -e '.[extras]'   then: from tet10 import "
            f"{name}")
    raise AttributeError(f"module 'minifea' has no attribute {name!r}")
