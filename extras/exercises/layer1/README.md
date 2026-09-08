# Layer 1 - Building the agent loop from scratch

Three short scripts, in order. By the end of the third one you will have written
(and understood) the same loop that sits underneath every agent framework on the
market.

| Script | What it adds |
| --- | --- |
| `ex1_first_call.py` | A single chat completion. No tools, no loop. |
| `ex2_one_tool.py` | One tool + the hand-rolled loop. **This is the core file.** |
| `ex3_agent_loop.py` | The same loop, generalized, driving a real mesher and FE solver. |

Everything uses the `openai` Python SDK against an OpenAI-compatible endpoint.
The provider is interchangeable; only three environment variables change.

---

## 1. Set the three environment variables

Every script reads exactly these:

| Variable | Meaning |
| --- | --- |
| `LLM_BASE_URL` | The API endpoint, including the `/v1` suffix. |
| `LLM_API_KEY` | Your key. |
| `LLM_MODEL` | Model name. Optional; defaults to `gemini-2.5-flash`. |

### Gemini through AI Studio (free tier)

Google's OpenAI-compatible endpoint works with the standard SDK unmodified.
Get a key at https://aistudio.google.com/apikey and substitute it below.

```bash
export LLM_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export LLM_API_KEY="<your Gemini key>"
export LLM_MODEL="gemini-2.5-flash"
```

### Generic OpenAI (or anything else OpenAI-compatible)

```bash
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o-mini"
```

The same pattern covers a local server. For Ollama:

```bash
export LLM_BASE_URL="http://localhost:11434/v1"
export LLM_API_KEY="ollama"          # ignored, but the SDK requires something
export LLM_MODEL="qwen2.5:14b"
```

On Windows PowerShell use `$env:LLM_BASE_URL = "..."` instead of `export`.

Check the variables took:

```bash
echo $LLM_BASE_URL $LLM_MODEL
```

> **Tool calling is not universal.** Any chat model runs `ex1`. `ex2` and `ex3`
> need a model trained for function calling. If a small local model replies with
> a chatty description of the tool instead of actually calling it, that is the
> model, not your code.

---

## 2. Run the exercises

From this directory:

```bash
python ex1_first_call.py
python ex2_one_tool.py
```

`ex3` imports the `minifea` package, so run it from the repository root (or with
the package installed via `pip install -e .`):

```bash
cd ../..
python exercises/layer1/ex3_agent_loop.py
```

---

## Exercise 1.1 - `ex1_first_call.py`

A single request: a system message setting the model's standing instructions, a
user message asking a cantilever question, one response printed.

**What to observe**

- The request is just a *list of messages*. There is no session, no server-side
  history. Statelessness is the fact that makes the rest of this course work the
  way it does.
- The model answers the beam question from memory. Run it two or three times and
  compare. Sometimes the arithmetic is right; sometimes a factor of `3` or a unit
  conversion quietly goes missing. Nothing in the response tells you which one
  you got. That uncertainty is the problem tools solve.

**Try this**

1. Ask for the answer in millimeters, then in inches. Does the physics survive
   the unit change, or does the number drift?
2. Delete the system message entirely and rerun. Watch the answer get longer and
   less structured. System prompts are cheap steering.
3. Ask something with no closed-form answer, like "what is the tip deflection if
   the beam has a 3 mm surface crack at the root?" Note whether the model hedges
   or invents a number.

---

## Exercise 1.2 - `ex2_one_tool.py`

The heart of the course. One tool (`cantilever_tip_deflection`, the
`F*L^3/(3*E*I)` formula) plus a `while`-style loop that runs it on request. The
file is commented as a lecture. Read it top to bottom *before* running it.

**What to observe**

- **The schema is prose, not code.** The model never sees your Python. It sees
  the `description` strings. Those descriptions are prompt engineering.
- **The turn structure.** Turn 1: the model asks for the tool, with arguments it
  pulled from the problem statement. Turn 2: it has the number and writes the
  answer. Compare against ex1's hand-waved number for the same beam.
- **The tool boundary.** The tool takes width and height and computes the
  second moment of area internally. That is deliberate: see the design note on
  the function. The model decides *what* to compute; the tool does *all* the
  arithmetic.
- **`tool_call_id`.** Each result message carries the id of the request it
  answers. This is what lets a model fire off several tool calls at once and
  still know which result is which.
- **The exit condition.** The loop ends when the model returns a message with no
  `tool_calls`. Nobody tells it to stop. It stops because it decided it was done.

**Try this**

1. **Break the units in the schema.** Change the `E` description to say
   "gigapascals" but leave the function expecting pascals. Rerun. You get a
   confident answer that is off by 10^9, and nothing flags it. This is the single
   most common failure mode in agentic CAE, and it is a documentation bug, not a
   code bug.
2. **Take the tool away.** Delete `tools=TOOLS` from the API call and rerun. The
   loop terminates on turn 1 because there are no tool calls, and the model
   guesses the arithmetic instead. Compare its number to the tool's.
3. **Ask for two beams at once.** Change the user message to ask about a 0.2 m
   beam *and* a 0.4 m beam. Many models will request both tool calls in a single
   assistant message. Watch the inner `for` loop handle both, and confirm two
   `role: "tool"` messages go back.
4. **Make the tool fail.** Add `raise ValueError("dimensions must be positive")`
   for non-positive width/height and ask about a beam with zero thickness. The
   error string goes back to the model as a tool result. Watch whether it
   recovers, asks you a question, or loops.
5. **Move the arithmetic outside the boundary.** Change the tool to take `I`
   directly and make the model compute it from the cross-section. When we ran
   that version live, the model slipped two orders of magnitude on `I` and the
   tool amplified it into a confident 0.1 m answer (truth: 1 mm). Where you
   draw the tool boundary is a reliability decision, not a style choice.

---

## Exercise 1.3 - `ex3_agent_loop.py`

Same loop, two real tools: `mesh_box` (gmsh, quadratic tets) and `run_fea`
(linear elastic solve), imported from `minifea.tools`. The dispatch collapses to
one generic `execute_tool(name, arguments)` call.

The task: mesh a 0.2 x 0.02 x 0.02 m cantilever, fix `xmin`, load `xmax`
downward with 1 kN, report max deflection and von Mises stress.

**What to observe**

- **Multi-step planning.** The agent has to mesh before it can solve, and it has
  to carry the mesh filename from the first tool result into the second call.
  Nobody scripted that ordering.
- **Unit reasoning.** The solver takes tractions in Pa, not point loads in N. The
  agent has to divide 1 kN by the 0.02 x 0.02 m end face area on its own. Check
  its arithmetic against the printed arguments. This is exactly the step that
  goes wrong silently in practice.
- **Error recovery.** If it invents a surface name, the solver's `ValueError`
  lists the real ones and comes back as a tool result. Watch the next turn: it
  usually fixes itself without you saying a word. Good error messages are agent
  infrastructure.
- **The turn cap.** 15 turns, then a bail-out message. Real agents do get stuck;
  an unbounded loop is an unbounded bill.
- Compare the final max deflection to the ex2 beam-theory number. They should be
  close but not identical. Ask yourself which one you trust and why.

**Try this**

1. **Ask for a comparison.** Change `TASK` to end with "then compare the result
   to Euler-Bernoulli beam theory and explain any difference." The agent now has
   to reason about shear deformation and the fixed-end constraint, not just
   report a number.
2. **Give it an impossible task.** Ask it to run the model with no boundary
   conditions at all, or to apply a load to a surface called `top`. The solver
   raises, the error comes back as a tool result, and you get to watch the retry
   behavior. Does it fix the problem, argue with you, or spin until the turn cap?
3. **Ask for a mesh convergence study.** "Repeat at three element sizes and tell
   me whether the deflection has converged." Now a single prompt produces many
   tool calls, and the 15-turn cap starts to matter. Note how much of the loop
   code had to change to support this: none.
4. **Turn off the units hint.** Delete the sentence in the system prompt about
   converting point loads to tractions. Does the agent still figure it out from
   the tool descriptions alone? That comparison tells you how much of your
   reliability lives in the system prompt versus the schema.

---

## What carries forward

By the end of `ex3` you have written the entire agent pattern:

1. Describe your tools in a schema the model can read.
2. Send the conversation plus the schemas.
3. If the reply contains tool calls, run them and append one `role: "tool"`
   message per call, each tagged with its `tool_call_id`.
4. Repeat until the reply has no tool calls. Cap the iterations.

Layer 2 replaces steps 1 and 3 with MCP, so the tools live in a separate process
behind a standard protocol and any client can use them. The loop itself does not
change, because there is nothing left to change.
