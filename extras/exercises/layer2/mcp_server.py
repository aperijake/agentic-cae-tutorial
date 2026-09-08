"""minifea as an MCP server -- SKELETON. Fill in the two TODOs.

In Layer 1 you hand-rolled the agent loop: you owned the conversation list, you
matched tool names to Python functions, you appended the results. That loop is
now somebody else's job. Here we publish the *same* tools over the Model Context
Protocol, and a harness (Claude Code) supplies the loop.

What actually changes: nothing about the tools. A tool is still a name, a typed
signature, a docstring, and a function body. MCP just standardizes how that
description travels over a pipe, so any client that speaks the protocol can use
your mesher and your solver without importing your code.

This file RUNS AS-IS. It serves one finished tool, mesh_box, as the worked
example. Your job is to add run_fea and verify_equilibrium below.

Run it directly (it talks JSON-RPC over stdin/stdout, so it will look like it
has hung -- that is a server waiting for a client, not a crash):

    python mcp_server.py

Register it with Claude Code (use the absolute path to your venv's python):

    claude mcp add minifea -- /path/to/.venv/bin/python /path/to/mcp_server.py

Units are SI everywhere: meters, newtons, pascals.
"""

from typing import Any

# The FastMCP-style server class was renamed in version 2 of the `mcp` package.
# Same decorator API, same behavior -- only the import moved.
try:
    from mcp.server.fastmcp import FastMCP  # mcp 1.x
except ImportError:  # pragma: no cover - depends on installed mcp version
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp 2.x

# The server name is what shows up in the client (e.g. `/mcp` in Claude Code).
mcp = FastMCP("minifea")


# ---------------------------------------------------------------------------
# Tool 1: meshing -- WORKED EXAMPLE, nothing to do here. Read it closely; the
# two TODOs below are the same four moves.
# ---------------------------------------------------------------------------
# The decorator reads the type annotations and the docstring and turns them into
# a JSON Schema that is sent to the model. There is no separate schema file to
# keep in sync: the signature IS the schema, and the docstring IS the prompt the
# model reads before deciding to call this. Write it for that reader -- state the
# units, name the legal string values, show one example call.


@mcp.tool()
def mesh_box(
    length: float,
    width: float,
    height: float,
    element_size: float,
    output_path: str = "box.msh",
) -> dict[str, Any]:
    """Mesh a rectangular box with quadratic tetrahedra (tet10) and write a gmsh file.

    The box spans (0, 0, 0) to (length, width, height). All lengths are in
    METRES (SI). A 200 mm beam is length=0.2, not 200.

    The mesh carries named surface groups you refer to later when setting
    boundary conditions and loads in run_fea:
      "xmin", "xmax" -- the two faces normal to x (at x=0 and x=length)
      "ymin", "ymax" -- the two faces normal to y (at y=0 and y=width)
      "zmin", "zmax" -- the two faces normal to z (at z=0 and z=height)
    The volume group is named "solid".

    Pick element_size so a few elements span the smallest dimension. Elements much
    smaller than that make the solve slow with no benefit; much larger and a
    bending model has too few elements through the thickness to be trustworthy.

    Example call, a 0.2 m x 0.02 m x 0.02 m cantilever with 5 mm elements:
        mesh_box(length=0.2, width=0.02, height=0.02, element_size=0.005)

    Args:
        length: Box dimension along x, in meters. Must be positive.
        width: Box dimension along y, in meters. Must be positive.
        height: Box dimension along z, in meters. Must be positive.
        element_size: Target element edge length, in meters. Must be positive.
        output_path: Path to write the gmsh .msh file to.

    Returns:
        A dict with "mesh_file" (path to pass to run_fea), "num_nodes",
        "num_elements", "element_type", and "surface_groups" (the names available
        for boundary conditions).
    """
    # Imported inside the function on purpose: the server can start, and the
    # client can list tools, even before minifea (or gmsh) is installed. An
    # import error then shows up as a readable tool error the agent can react
    # to, instead of a server that dies at startup with no explanation.
    from minifea.meshing import mesh_box as _mesh_box

    return _mesh_box(
        length=length,
        width=width,
        height=height,
        element_size=element_size,
        output_path=output_path,
    )


# ---------------------------------------------------------------------------
# TODO 1: expose the solver as a tool named run_fea
# ---------------------------------------------------------------------------
# Uncomment the block below and fill in the body. Keep this exact signature --
# the exercises and the solution file both assume it:
#
#     @mcp.tool()
#     def run_fea(
#         mesh_file: str,
#         material: dict[str, float],
#         bcs: list[dict[str, Any]],
#         loads: list[dict[str, Any]],
#     ) -> dict[str, Any]:
#
# The body is three lines: import minifea.solver.solve INSIDE the function (same
# reason as mesh_box above), call it with these arguments plus
# output_vtu="results.vtu", and return its dict unchanged.
#
# Do NOT wrap the call in try/except. When the solver raises
# ValueError("unknown surface 'left'; available: xmin, xmax, ..."), the MCP layer
# hands that message back to the model as a failed tool call and the model
# retries with a legal name. Swallowing the exception hides the fix from it.
#
# The docstring is the real work -- it is the only thing the model reads before
# choosing arguments. Cover, in your own words:
#   - Units: E in PASCALS (steel is 210e9, not 210), tractions and pressures in
#     Pa, lengths in m, displacements out in m, stresses in Pa, reactions in N.
#   - material: {"E": <Pa>, "nu": <-, 0 <= nu < 0.5}
#   - bcs: list of {"surface": one of "xmin"/"xmax"/"ymin"/"ymax"/"zmin"/"zmax",
#     "type": "fixed"}; at least one is required or the model floats and the
#     solve fails with a rigid-body-motion error.
#   - loads: list of either
#       {"surface": ..., "type": "traction", "vector": [tx, ty, tz]}  (Pa)
#       {"surface": ..., "type": "pressure", "value": p}  (Pa, + = into surface)
#     and say plainly that a traction is force per unit AREA: for a total force F
#     on a surface of area A, use t = F / A.
#   - One complete example call, e.g. 1 kN down on the tip of a steel cantilever:
#     mesh_file="box.msh", material={"E": 210e9, "nu": 0.3},
#     bcs=[{"surface": "xmin", "type": "fixed"}],
#     loads=[{"surface": "xmax", "type": "traction", "vector": [0, 0, -2.5e6]}]
#     (0.02 x 0.02 m face is 4e-4 m^2, so 1000 N / 4e-4 m^2 = 2.5e6 Pa).
#   - What comes back: num_nodes, num_elements, num_dofs, max_displacement (m),
#     max_displacement_vector, max_von_mises (Pa), reactions ({surface: [Rx,Ry,Rz]}
#     in N, one entry per fixed surface), output_vtu.
#
# @mcp.tool()
# def run_fea(...):
#     ...


# ---------------------------------------------------------------------------
# TODO 2: add a verification tool named verify_equilibrium
# ---------------------------------------------------------------------------
# This one calls no simulation code at all. It exists because an agent's summary
# of a run is generated text, and generated text is persuasive whether or not the
# run was any good. A tool that computes a verdict from the numbers gives both
# you and the model something that cannot be talked around.
#
# Keep this exact signature:
#
#     @mcp.tool()
#     def verify_equilibrium(
#         reactions: dict[str, list[float]],
#         applied_force: list[float],
#         tolerance: float = 0.01,
#     ) -> dict[str, Any]:
#
# The physics: for a static analysis the reactions must balance the applied load,
# sum(R) + F_applied = 0. So
#   1. sum the reaction vectors over the surfaces in `reactions` (componentwise),
#   2. residual = reaction_sum + applied_force,
#   3. relative_error = |residual| / |applied_force|,
#   4. passed = relative_error <= tolerance.
# Guard the divide: if |applied_force| is zero, fall back to the reaction
# magnitude, and if that is zero too the residual is zero and the check passes.
#
# Return a dict with at least: "passed" (bool), "reaction_sum", "applied_force",
# "residual", "residual_magnitude", "relative_error", "tolerance", and "message"
# (a one-line verdict the model can quote, and that says what to check on a FAIL).
#
# In the docstring, say what this check does and does NOT catch. Equilibrium is
# insensitive to Young's modulus: enter E in GPa instead of Pa and the reactions
# are unchanged while the displacements are wrong by 1e9. That is the point of
# exercise 2.3, and it is worth writing down where the model will read it.
# Give the units (newtons) and one example call, and note that run_fea takes
# tractions in Pa, so applied_force is traction times loaded area.
#
# @mcp.tool()
# def verify_equilibrium(...):
#     ...


if __name__ == "__main__":
    # stdio transport: the client launches this file as a subprocess and speaks
    # JSON-RPC over stdin/stdout. That is why the server prints nothing useful
    # when you run it by hand, and why anything you print to stdout elsewhere in
    # this file would corrupt the protocol stream. Use stderr for debugging.
    mcp.run()
