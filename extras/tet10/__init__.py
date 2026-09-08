"""The gmsh/meshio tet10 path: an optional companion to the core package.

This is the original tutorial toolchain -- gmsh meshing, a quadratic-tet
solver, and an LLM-facing tool layer over both. It still works and is still
worth reading, but it is no longer the front door: it needs gmsh and meshio,
and the Cook's membrane material in ``minifea`` deliberately needs neither.

Install with:  pip install -e '.[extras]'
"""

from tet10.meshing import mesh_box
from tet10.solver import solve
from tet10.tools import TOOLS, execute_tool

__all__ = ["mesh_box", "solve", "TOOLS", "execute_tool"]
