# Layer 3 — The agent extends its own tooling (stretch)

In Layers 1 and 2 the agent *used* tools someone wrote for it. In practice, a
large share of agentic engineering today is the other direction: agents
*writing and extending* the tools themselves. This repository is small enough
that Claude Code can read all of it — so let it.

Pick one challenge, open Claude Code in the repo root (with your MCP server
from Layer 2 still registered), and drive.

## Challenge A: mesh convergence study tool

> Read minifea and the MCP server in exercises/layer2. Add a new MCP tool
> `convergence_study` that re-meshes a box at a series of element sizes, runs
> the same analysis on each, and reports max displacement and max von Mises
> per size along with the relative change between successive refinements.
> Then use it: is a 5 mm mesh converged for the cantilever from Layer 2, if
> my question is tip deflection? What if my question is peak stress?

Watch for: does the agent run the study *through its own new tool*, and does
it give you the (correct) different answer for deflection vs peak stress?

## Challenge B: a capability the solver doesn't have

> Read minifea/solver.py. Add support for a body-force load (self-weight):
> loads like {"type": "gravity", "vector": [0, 0, -9.81], "density": 7850}.
> Extend the tool schema and the MCP server you registered, add a test against
> the exact solution for a bar hanging under its own weight
> (tip displacement = rho * g * L^2 / (2E) -- exact only when gravity acts
> along the bar's long axis and nu = 0, like test_uniaxial_tension_exact),
> run the test suite, and then use the new capability to compare stress in the
> Layer 2 cantilever with and without self-weight.

Watch for: does the agent add a *verification test* without being reminded
(it was asked); does the test actually pass; do the existing 13 tests still
pass? Then read the tool-schema diff yourself: the load items previously
required a "surface" key, gravity has none, and **no test catches a schema
that still demands it** -- the schema is prose the model reads, so a green
test suite proves nothing about it. That review is the exercise.

Expected physics: for this beam the self-weight (about 6 N) is noise next to
the 1 kN tip load -- peak stress rises about 0.3%. An agent that reports
"negligible, and here is why" beats one that reports numbers without judgment.

## The point

When the demo works, notice what you just did: described a capability in a
sentence, and reviewed the diff of a solver you understand. That review step
is not optional — you are the engineer of record. The agent's test passing is
evidence, not proof; read what it wrote.
