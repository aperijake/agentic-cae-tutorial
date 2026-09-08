# extras

Material that is still useful but is no longer the front door.

Everything here needs dependencies the core package deliberately avoids, so it
is installed separately:

```bash
pip install -e '.[extras]'
```

Nothing in `minifea/` imports anything from here. That is the point: a clean
install of the core package cannot fail on a mesher the main material never
uses, and the Cook's membrane demonstration runs on numpy, scipy and
matplotlib alone.

## `tet10/`

The original toolchain: gmsh box meshing, a quadratic-tetrahedron linear
elastic solver, and an LLM-facing tool layer over both.

```python
from tet10 import mesh_box, solve, TOOLS, execute_tool
```

Worth reading for the tool layer in particular (`tools.py`), which is the
clearest small example in the repo of what an LLM actually sees when you give
it a tool: a name, a description written for a reader who cannot see your
code, and a JSON Schema. The simulation code underneath is invisible to the
model.

Its tests live alongside it and skip automatically if gmsh is not installed.

## `exercises/`

The three-layer build-an-agent sequence:

- **layer1** — the agent loop written by hand, from a single API call up to a
  tool-using loop that recovers from a planted failure.
- **layer2** — the same tools exposed over MCP and driven from a coding
  harness.
- **layer3** — asking the agent to read the solver and extend it.

These drive the `tet10` path, so they need the extras installed too.
