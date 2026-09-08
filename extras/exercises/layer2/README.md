# Layer 2 -- MCP and a real harness

Layer 1 left you with a working agent loop that you wrote by hand. Layer 2 throws
your loop away and keeps your tools.

Files here:

- `mcp_server.py` -- the skeleton you complete (runs as-is, serves one tool)
- `solutions/mcp_server.py` -- the finished server, three tools

---

## 1. What MCP is

In `ex3_agent_loop.py` you had three things: a list of tool schemas, a dispatch
function that mapped a tool name to a Python call, and a `while` loop that fed
results back to the model. Only the first two are about *your* tools. The loop is
plumbing, and everyone who writes an agent writes the same plumbing.

The Model Context Protocol (MCP) is that split made official. A **server** owns
the tools: their names, their argument schemas, their descriptions, and the code
that runs them. A **client** (Claude Code, an IDE, your own script) owns the
conversation, the loop, the retries, and the user interface. Between them runs a
small JSON-RPC protocol with a handful of methods -- `initialize`, `tools/list`,
`tools/call`.

That is the whole idea, and the payoff is worth being explicit about:

- **The tool contract is promoted to a protocol.** In Layer 1 your schemas were a
  Python list that only your script could read. Now any MCP client can call
  `tools/list` and discover them, without importing minifea or even being written
  in Python.
- **The loop stops being yours.** Retries, streaming, permission prompts,
  conversation history, multi-tool calls: all of that lives in the harness now.
  You maintain solver code, not agent code.
- **Tools stay data.** Look at `mesh_box` in `mcp_server.py`. There is no schema
  literal anywhere in the file. `@mcp.tool()` reads the type annotations to build
  the JSON Schema and the docstring to build the description, and ships both to
  the model. Your signature is the API, and your docstring is the prompt.

That last point is the real lesson of this layer. The model never sees your
function body. It sees a name, a schema, and your prose, and from those it
decides whether to call your tool and with what arguments. A vague docstring is a
bug: if you do not say that `E` is in pascals, expect to be handed 210.

Transport: these servers speak **stdio**. The client launches your script as a
subprocess and talks JSON-RPC over its stdin and stdout. Two consequences worth
remembering. Running `python mcp_server.py` in a terminal looks like a hang --
that is a server waiting for a client that will never connect, and Ctrl-C is the
correct exit. And a stray `print()` in your tool code corrupts the protocol
stream, so debug to stderr.

---

## 2. Claude Code and a key

Claude Code is the harness we use as the MCP client. It talks to Anthropic's
API, so you need an Anthropic account: a Claude subscription or an API key.
Run `claude` once and follow the login prompt.

If you have ever pointed Claude Code at another endpoint, make sure those
variables are not set (`unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN
ANTHROPIC_MODEL`), or you will be routed there instead of your own account.

If you have not installed Claude Code yet, do that first; Anthropic's docs
cover every platform.

---

## 3. Registering your server

From the tutorial repo root, with your virtualenv active:

```bash
claude mcp add minifea -- /absolute/path/to/.venv/bin/python /absolute/path/to/exercises/layer2/mcp_server.py
```

`minifea` is the name the server appears under. Everything after `--` is the
command Claude Code will run to start it.

**Use the absolute path to your virtualenv's Python.** This is the single most
common failure in this exercise. Claude Code launches the server as a subprocess
with its own environment; a bare `python` may resolve to a system interpreter
that has never heard of `mcp`, `gmsh`, or `minifea`, and the server dies at
startup with an import error you will not see. The same goes for the script path:
give the full path, not a relative one.

Print the right paths with:

```bash
python -c "import sys; print(sys.executable)"      # venv python
python -c "import os; print(os.path.abspath('exercises/layer2/mcp_server.py'))"
```

Then start `claude` and type `/mcp`. You should see `minifea` connected, and
expanding it should list your tools. Useful commands:

```bash
claude mcp list                 # what is registered
claude mcp remove minifea       # unregister (do this before re-adding)
```

---

## 4. Exercises

### 2.1 -- Complete the server

Open `mcp_server.py`. `mesh_box` is finished as the worked example. Two TODO
blocks follow it, each with the exact signature to use.

1. **`run_fea`** -- wrap `minifea.solver.solve`. Import it inside the function
   (see `mesh_box` for why), pass `output_vtu="results.vtu"`, and return the dict
   unchanged. Do not catch exceptions: minifea's error messages are written to be
   read by the model, and swallowing them hides the fix.
2. **`verify_equilibrium`** -- pure arithmetic, no simulation. Sum the reaction
   vectors, compare against the applied load, report a relative error and a
   pass/fail.

Spend real effort on the docstrings. Units, the legal surface names
(`"xmin"`..`"zmax"`), and one worked example call each. You are writing for a
reader who has your signature and your prose and nothing else.

Check your work by running the server directly:

```bash
python exercises/layer2/mcp_server.py     # should hang silently; Ctrl-C to stop
```

A traceback here means a syntax or import error at module scope. Silence means it
started. Then register it and confirm all three tools appear under `/mcp`.

`solutions/mcp_server.py` has a full version if you get stuck or want to compare
docstrings afterwards.

### 2.2 -- Drive the cantilever conversationally

Same problem you solved in Layer 1. In `claude`, ask for it in plain language:

> Using the minifea tools, mesh a 0.2 x 0.02 x 0.02 m steel cantilever, fix the
> xmin face, and apply 1 kN downward at the xmax face. Report the tip deflection
> and the maximum von Mises stress.

Watch the tool calls scroll past. Then compare against Layer 1 honestly:

- **The same:** the tools, the schemas, the model's job of choosing arguments,
  and the fact that a traction in Pa had to be worked out from a force in N.
- **Different:** you wrote no loop. Nothing about turn limits, message
  accumulation, or JSON parsing is in your code any more.
- **Also different:** the harness brought its own tools. It can read your files
  and run shell commands, so it may inspect `results.vtu` or write a script
  without being asked. More capability, and less predictability -- the same
  trade-off you will make with any harness.

Ask a follow-up in the same session ("now refine the mesh to 2.5 mm elements and
tell me how much the tip deflection moved"). Convergence checks are cheap when
the loop is somebody else's problem.

### 2.3 -- The anchoring exercise

This is the part to take home.

**Step 1.** Re-run the cantilever, then ask the agent to check its own work with
`verify_equilibrium`. It has to turn the traction it applied back into a total
force in newtons (traction times loaded area) and hand that in along with the
`reactions` dict. Expect a pass with a relative error near machine precision.

**Step 2.** Now plant a units bug. Start a fresh conversation and give the same
problem, but specify the material as **E = 210, nu = 0.3** -- the classic slip of
entering steel in GPa when the solver wants pascals. Ask for the same two
outputs, then ask for `verify_equilibrium` again.

Watch which symptom each check catches:

| Check | Catches the GPa mistake? |
| --- | --- |
| Force equilibrium (`verify_equilibrium`) | **No.** Reactions are set by the applied load and the constraints. Stiffness does not enter the balance, so the check passes cleanly on a model that is wrong by nine orders of magnitude. |
| The agent's narrative summary | Sometimes. It may flag the deflection as implausible -- or explain it fluently and move on. Not something to rely on. |
| Tip deflection against beam theory (`PL^3/3EI`) | **Yes, immediately.** The prediction scales with `1/E`, so the error is a clean factor of 1e9. |
| A sanity bound on displacement (say, deflection < 10% of span) | **Yes.** Crude, but it fires. |

Two conclusions:

1. **Deterministic checks beat plausible prose.** The agent's summary is
   generated text, and it reads exactly as confident whether or not the numbers
   are any good. A tool that computes a verdict is something neither of you can
   talk around.
2. **A check only pins down what it is sensitive to.** Equilibrium verifies
   assembly and load application. It says nothing about whether the material was
   entered correctly. Anchoring an agentic workflow means picking checks that are
   sensitive to the errors you actually expect, and knowing what each one leaves
   uncovered.

Optional, and the natural next step: add a `check_beam_theory` tool that computes
`PL^3/3EI` and compares it against `max_displacement`. Then re-run the broken
case and watch it fail loudly. That is Layer 3 territory -- and a good candidate
for asking the agent to write it for you.

---

## 5. Troubleshooting

**`minifea` does not appear in `/mcp`.**
Run `claude mcp list` to confirm it is registered at all. If it is listed but not
connecting, the server is failing at startup -- almost always the wrong Python.
Re-add it with the absolute path from `python -c "import sys; print(sys.executable)"`.

**It appeared, then stopped appearing.**
`claude mcp add` registers per project directory by default. Launch `claude` from
the same directory you registered from, or re-add it there.

**I edited `mcp_server.py` and Claude Code still shows the old tools.**
The server process is started once per session. Exit `claude` and relaunch. If a
tool list still looks stale, `claude mcp remove minifea` then add it again.

**Tools appear but every call fails with `No module named 'minifea'`.**
The server is running under a Python that cannot see the package. Activate the
venv and `pip install -e .` from the repo root, and check the interpreter path in
your `claude mcp add` command.

**Running `python mcp_server.py` just sits there.**
That is correct. A stdio server waits for a client on stdin. Ctrl-C to exit.

**Weird JSON errors, or the server drops mid-call.**
Something printed to stdout. `print()` inside a tool corrupts the JSON-RPC
stream; write to `sys.stderr` instead, or return the value in the tool's result
dict where the model can actually see it.

**Claude Code cannot reach the model at all (auth or connection errors).**
Check that `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_MODEL`
are unset in the shell you launched `claude` from, then run `claude` and log
in again.
