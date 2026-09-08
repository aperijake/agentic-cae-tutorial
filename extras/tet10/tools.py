"""The LLM-facing tool layer.

This module is the bridge between an LLM agent and the simulation code. Two
things live here, and together they demonstrate the core idea of tool use:

1. ``TOOLS`` -- tool *descriptions*: names, plain-language documentation, and
   JSON Schemas for the arguments. This is pure data. It is all the LLM ever
   sees; the model never touches the Python underneath.
2. ``execute_tool`` -- the dispatcher that runs a described tool. Errors are
   returned to the model as data instead of raised, so an agent loop can feed
   failures back and let the model correct itself.

The schemas use the OpenAI function-calling format, which every
OpenAI-compatible provider understands.
"""

import json

from tet10.meshing import mesh_box
from tet10.solver import solve

_SURFACE_NAMES = ["xmin", "xmax", "ymin", "ymax", "zmin", "zmax"]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mesh_box",
            "description": (
                "Mesh a rectangular solid box spanning (0,0,0) to "
                "(length,width,height), in meters, with quadratic tetrahedra. "
                "The six faces get named surface groups (xmin, xmax, ymin, "
                "ymax, zmin, zmax) which boundary conditions and loads refer "
                "to. Returns the mesh file path and mesh statistics. Example: "
                "a 0.2 m long beam with 0.02 m square cross-section at 5 mm "
                "resolution -> {\"length\": 0.2, \"width\": 0.02, "
                "\"height\": 0.02, \"element_size\": 0.005}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "length": {"type": "number",
                               "description": "Box size in x, meters."},
                    "width": {"type": "number",
                              "description": "Box size in y, meters."},
                    "height": {"type": "number",
                               "description": "Box size in z, meters."},
                    "element_size": {
                        "type": "number",
                        "description": ("Target element edge length, meters. "
                                        "Must be smaller than the smallest "
                                        "box dimension.")},
                    "output_path": {
                        "type": "string",
                        "description": "Output mesh file path (.msh). "
                                       "Default box.msh."},
                },
                "required": ["length", "width", "height", "element_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_fea",
            "description": (
                "Run a small-displacement linear elastic finite element "
                "analysis on a mesh produced by mesh_box. SI units: E in Pa "
                "(steel is 210e9, NOT 210), tractions and pressures in Pa, "
                "results in meters and Pa. Surfaces are named xmin, xmax, "
                "ymin, ymax, zmin, zmax. Example cantilever: fix xmin, apply "
                "a downward traction on xmax. Returns max displacement, max "
                "von Mises stress, and reaction forces at fixed surfaces; "
                "also writes results.vtu for visualization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mesh_file": {
                        "type": "string",
                        "description": "Path to the .msh file from mesh_box."},
                    "material": {
                        "type": "object",
                        "description": ("Isotropic material: E = Young's "
                                        "modulus in Pa, nu = Poisson's ratio."),
                        "properties": {
                            "E": {"type": "number"},
                            "nu": {"type": "number"},
                        },
                        "required": ["E", "nu"],
                    },
                    "bcs": {
                        "type": "array",
                        "description": "Boundary conditions. At least one "
                                       "fixed surface is required.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "surface": {"type": "string",
                                            "enum": _SURFACE_NAMES},
                                "type": {"type": "string",
                                         "enum": ["fixed"]},
                            },
                            "required": ["surface", "type"],
                        },
                    },
                    "loads": {
                        "type": "array",
                        "description": (
                            "Surface loads. traction: force per unit area "
                            "vector [tx,ty,tz] in Pa. pressure: scalar value "
                            "in Pa, positive pushes into the surface."),
                        "items": {
                            "type": "object",
                            "properties": {
                                "surface": {"type": "string",
                                            "enum": _SURFACE_NAMES},
                                "type": {"type": "string",
                                         "enum": ["traction", "pressure"]},
                                "vector": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3, "maxItems": 3},
                                "value": {"type": "number"},
                            },
                            "required": ["surface", "type"],
                        },
                    },
                },
                "required": ["mesh_file", "material", "bcs", "loads"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    """Run one tool call and return the result as a JSON string.

    Exceptions become {"error": ...} results instead of crashing the loop:
    the error message goes back to the model, which gets a chance to fix its
    inputs and try again. Readable error messages are what make that loop
    converge -- they are written for the model as much as for you.
    """
    try:
        if name == "mesh_box":
            result = mesh_box(**arguments)
        elif name == "run_fea":
            result = solve(**arguments, output_vtu="results.vtu")
        else:
            result = {"error": f"Unknown tool: {name!r}. "
                               f"Available tools: mesh_box, run_fea."}
    except Exception as exc:  # noqa: BLE001 - the model handles the error
        result = {"error": str(exc)}
    return json.dumps(result)
