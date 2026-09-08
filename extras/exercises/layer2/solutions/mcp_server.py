"""minifea as an MCP server -- SOLUTION.

In Layer 1 you hand-rolled the agent loop: you owned the conversation list, you
matched tool names to Python functions, you appended the results. That loop is
now somebody else's job. Here we publish the *same* tools over the Model Context
Protocol, and a harness (Claude Code) supplies the loop.

What actually changes: nothing about the tools. A tool is still a name, a typed
signature, a docstring, and a function body. MCP just standardizes how that
description travels over a pipe, so any client that speaks the protocol can use
your mesher and your solver without importing your code.

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
# Tool 1: meshing
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
# Tool 2: the solve
# ---------------------------------------------------------------------------
# Note what we do NOT do below: no try/except that swallows the error. When the
# solver raises ValueError("unknown surface 'left'; available: xmin, xmax, ..."),
# the MCP layer hands that message back to the model as a failed tool call, and
# the model retries with a legal name. Readable exception messages are part of
# the tool contract here, not an afterthought.


@mcp.tool()
def run_fea(
    mesh_file: str,
    material: dict[str, float],
    bcs: list[dict[str, Any]],
    loads: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run a small-displacement linear elastic analysis on a mesh from mesh_box.

    Units are SI: E in PASCALS, tractions and pressures in PASCALS, lengths in
    meters, resulting displacements in meters, stresses in pascals, reactions in
    newtons. Steel is E = 210e9 Pa, NOT 210 (that would be 210 Pa, and the beam
    would deflect kilometres).

    material: {"E": Young's modulus in Pa, "nu": Poisson's ratio, 0 <= nu < 0.5}

    bcs: a list of constrained surfaces, each
        {"surface": <one of "xmin","xmax","ymin","ymax","zmin","zmax">,
         "type": "fixed"}
      "fixed" clamps all three displacement components on that surface. At least
      one fixed surface is required, otherwise the model is free to float and the
      solve fails with a rigid-body-motion error.

    loads: a list of loaded surfaces, each either
        {"surface": ..., "type": "traction", "vector": [tx, ty, tz]}  # Pa
        {"surface": ..., "type": "pressure", "value": p}              # Pa, + = into the surface
      A traction is force per unit AREA. To apply a total force F to a surface of
      area A, use t = F / A. For 1 kN downward (-z) on a 0.02 x 0.02 m end face:
      A = 4e-4 m^2, so vector = [0, 0, -2.5e6].

    Example call, 1 kN down on the tip of a steel cantilever clamped at xmin:
        run_fea(mesh_file="box.msh",
                material={"E": 210e9, "nu": 0.3},
                bcs=[{"surface": "xmin", "type": "fixed"}],
                loads=[{"surface": "xmax", "type": "traction",
                        "vector": [0.0, 0.0, -2.5e6]}])

    Args:
        mesh_file: Path to the .msh file written by mesh_box.
        material: Elastic properties, {"E": Pa, "nu": -}.
        bcs: Constrained surfaces (see above). At least one is required.
        loads: Applied surface loads (see above).

    Returns:
        A dict with "num_nodes", "num_elements", "num_dofs",
        "max_displacement" (magnitude in m), "max_displacement_vector" [ux,uy,uz],
        "max_von_mises" (Pa), "reactions" ({surface: [Rx,Ry,Rz]} in N, one entry
        per fixed surface), and "output_vtu" (results written for viewing in
        ParaView). Feed "reactions" straight into verify_equilibrium.
    """
    from minifea.solver import solve

    return solve(
        mesh_file=mesh_file,
        material=material,
        bcs=bcs,
        loads=loads,
        output_vtu="results.vtu",
    )


# ---------------------------------------------------------------------------
# Tool 3: the anchor
# ---------------------------------------------------------------------------
# This one calls no simulation code at all. It exists because an agent's summary
# of a run is generated text, and generated text is persuasive whether or not the
# run was any good. A tool that computes a verdict from the numbers gives both
# you and the model something that cannot be talked around.


@mcp.tool()
def verify_equilibrium(
    reactions: dict[str, list[float]],
    applied_force: list[float],
    tolerance: float = 0.01,
) -> dict[str, Any]:
    """Check global force balance: do the reactions sum to minus the applied load?

    For a converged static analysis, the sum of all reaction forces must balance
    the total applied force: sum(R) + F_applied = 0. This is a property of the
    assembled system, so it holds for any correct linear solve regardless of mesh
    quality, material values, or how sensible the model is physically.

    That makes it a useful anchor, and also tells you exactly what it does NOT
    catch. Equilibrium is insensitive to Young's modulus: enter E in GPa instead
    of Pa and the reactions are unchanged while the displacements are off by a
    factor of 1e9. A deterministic check is worth running even when the agent's
    write-up "looks right" -- an LLM is fluent about results it never verified --
    but one check only ever pins down one thing. Pair this with a check on the
    quantity you actually care about, such as tip deflection against beam theory.

    All forces are in NEWTONS (SI). Vectors are [Fx, Fy, Fz] in global axes.

    Getting applied_force right: run_fea takes tractions in Pa (force per area),
    so multiply by the loaded surface area. A traction of [0, 0, -2.5e6] Pa on a
    0.02 x 0.02 m face (A = 4e-4 m^2) is an applied force of [0, 0, -1000] N.

    Example call, using the "reactions" dict returned by run_fea:
        verify_equilibrium(reactions={"xmin": [0.0, 0.0, 1000.0]},
                           applied_force=[0.0, 0.0, -1000.0])
      -> passes: the reaction sum [0, 0, 1000] balances the load [0, 0, -1000].

    Args:
        reactions: The "reactions" dict from run_fea, {surface_name: [Rx, Ry, Rz]}
            in newtons. Every fixed surface must be included.
        applied_force: Total applied force [Fx, Fy, Fz] in newtons, summed over
            every load in the model.
        tolerance: Allowed relative error before the check fails. Default 0.01
            (1%); a direct linear solve normally lands far below this.

    Returns:
        A dict with "passed" (bool), "reaction_sum" [N], "applied_force" [N],
        "residual" (reaction_sum + applied_force, should be ~zero) [N],
        "residual_magnitude" [N], "relative_error" (residual magnitude divided by
        the applied force magnitude), "tolerance", and "message" (a one-line
        human-readable verdict).
    """
    if len(applied_force) != 3:
        raise ValueError(
            f"applied_force must be [Fx, Fy, Fz] in newtons; got {applied_force!r}"
        )

    # Sum the reaction vectors over every fixed surface.
    reaction_sum = [0.0, 0.0, 0.0]
    for surface, vector in reactions.items():
        if len(vector) != 3:
            raise ValueError(
                f"reaction for surface '{surface}' must be [Rx, Ry, Rz] in newtons; "
                f"got {vector!r}"
            )
        for i in range(3):
            reaction_sum[i] += float(vector[i])

    applied = [float(component) for component in applied_force]

    # Static equilibrium: reactions + applied load = 0, so the residual is what
    # is left over, and it should be zero to solver round-off.
    residual = [reaction_sum[i] + applied[i] for i in range(3)]
    residual_magnitude = sum(r * r for r in residual) ** 0.5
    applied_magnitude = sum(f * f for f in applied) ** 0.5

    # Normalize by the applied load. With no applied load there is nothing to
    # normalize by, so fall back to the reaction magnitude, and if that is zero
    # too the residual is zero and the check trivially passes.
    reference = applied_magnitude
    if reference == 0.0:
        reference = sum(r * r for r in reaction_sum) ** 0.5
    relative_error = residual_magnitude / reference if reference > 0.0 else 0.0

    passed = relative_error <= tolerance
    if passed:
        message = (
            f"PASS: reactions balance the applied load to "
            f"{relative_error:.3e} relative error (tolerance {tolerance:g})."
        )
    else:
        message = (
            f"FAIL: force balance is off by {residual_magnitude:.6g} N, "
            f"{relative_error:.3e} relative error (tolerance {tolerance:g}). "
            f"Reactions sum to {reaction_sum}, applied load is {applied}. "
            f"Check that every fixed surface is included in 'reactions' and that "
            f"'applied_force' is the traction times the loaded area, in newtons."
        )

    return {
        "passed": passed,
        "reaction_sum": reaction_sum,
        "applied_force": applied,
        "residual": residual,
        "residual_magnitude": residual_magnitude,
        "relative_error": relative_error,
        "tolerance": tolerance,
        "message": message,
    }


if __name__ == "__main__":
    # stdio transport: the client launches this file as a subprocess and speaks
    # JSON-RPC over stdin/stdout. That is why the server prints nothing useful
    # when you run it by hand, and why anything you print to stdout elsewhere in
    # this file would corrupt the protocol stream. Use stderr for debugging.
    mcp.run()
