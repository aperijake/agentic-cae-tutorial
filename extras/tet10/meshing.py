"""Box meshing with gmsh.

Produces quadratic tetrahedral (tet10) meshes with named surface groups so that
boundary conditions can be referred to by name ("xmin", "xmax", ...) instead of
by node id lists. SI units: all dimensions in meters.
"""

import os

import gmsh

# Names for the six faces of the box, identified by which coordinate is pinned.
_FACE_NAMES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]


def mesh_box(length: float, width: float, height: float,
             element_size: float, output_path: str = "box.msh") -> dict:
    """Mesh a rectangular box spanning (0,0,0) to (length,width,height) [m]
    with quadratic tetrahedra (tet10) of target size ``element_size`` [m].

    The six faces are tagged as physical surface groups "xmin", "xmax",
    "ymin", "ymax", "zmin", "zmax"; the volume is tagged "solid". These names
    are how the solver's boundary conditions and loads refer to geometry.

    Returns a summary dict: mesh_file, num_nodes, num_elements, element_type,
    surface_groups.
    """
    for name, value in [("length", length), ("width", width),
                        ("height", height), ("element_size", element_size)]:
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{name} must be a positive number, got {value!r}")
    if element_size > min(length, width, height):
        raise ValueError(
            f"element_size ({element_size} m) is larger than the smallest box "
            f"dimension ({min(length, width, height)} m); the mesh would have "
            "no elements through the thickness. Use a smaller element_size.")

    dims = (length, width, height)
    # interruptible=False keeps gmsh from installing signal handlers, which
    # only work on the main thread -- and MCP servers run tools on worker
    # threads. Without this, mesh_box works in a plain script but dies inside
    # an MCP server with "signal only works in main thread".
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0.0, 0.0, 0.0, length, width, height)
        gmsh.model.occ.synchronize()

        # Identify each face by its center of mass and tag it with a name.
        tol = 1e-9 * max(dims)
        for dim, tag in gmsh.model.getEntities(2):
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            face_name = None
            for axis in range(3):
                if abs(com[axis] - 0.0) < tol:
                    face_name = _FACE_NAMES[2 * axis]
                elif abs(com[axis] - dims[axis]) < tol:
                    face_name = _FACE_NAMES[2 * axis + 1]
                if face_name:
                    break
            if face_name is None:  # pragma: no cover - cannot happen for a box
                raise RuntimeError(f"could not classify face at {com}")
            group = gmsh.model.addPhysicalGroup(2, [tag])
            gmsh.model.setPhysicalName(2, group, face_name)

        vol_group = gmsh.model.addPhysicalGroup(3, [1])
        gmsh.model.setPhysicalName(3, vol_group, "solid")

        gmsh.option.setNumber("Mesh.MeshSizeMin", element_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.setOrder(2)  # promote to quadratic (tet10)

        # msh 2.2 is the simplest widely-readable format with physical names.
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        output_path = os.path.abspath(output_path)
        gmsh.write(output_path)

        node_tags, _, _ = gmsh.model.mesh.getNodes()
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        num_elements = sum(len(tags) for tags in elem_tags)
    finally:
        gmsh.finalize()

    return {
        "mesh_file": output_path,
        "num_nodes": len(node_tags),
        "num_elements": int(num_elements),
        "element_type": "tet10",
        "surface_groups": list(_FACE_NAMES),
    }
