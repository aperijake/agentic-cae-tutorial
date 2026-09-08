# Agentic CAE Tutorial

Companion code for Jake Koester's session of the USACM AESCAPE 2026 short
course SC03, Introduction to Agentic Workflows. The theme is anchored agents:
the LLM interprets intent, deterministic code handles whatever must be exact.

## What's here

| Piece | What it is |
|---|---|
| `playground.py` | Three mocked agents (main, geometry, setup) sharing one small state. Edit their instructions, watch the unit system get lost across the handoff, then let code carry it. Needs a key. |
| `look.py` | Solve Cook's membrane, snapshot the pressure field, send the picture to the model, ask. Three one-line edits show what it sees and what it doesn't. Needs a key. |
| `unit_demo.py` | The scripted version of the same failure, plus the unit consistency pipeline. Runs with no key, from recorded model responses. |
| `minifea/` | A small FE toolkit you can read in a sitting: structured hex mesher, hex8 with and without B-bar, tet4, Cook's membrane, plots, the unit pipeline, the two-agent handoff, and a thin LLM client with a recorder. |
| `make_figures.py` | The Cook's membrane locking figures: three formulations, one mesh. |
| `capture_recordings.py` | Re-records the model responses `unit_demo.py` replays. Needs a key. |
| `tests/` | Patch tests, equilibrium, locking, the unit pipeline, the playground's check. `pytest`. |
| `extras/` | Optional and older: the gmsh tet10 toolchain and exercises that build an agent loop by hand and wrap it in an MCP server. `pip install -e '.[extras]'`. |

numpy, scipy, matplotlib, openai. That is the whole dependency list.

## Setup

Python 3.10 or newer, about five minutes.

```bash
git clone https://github.com/aperijake/agentic-cae-tutorial
cd agentic-cae-tutorial
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python setup_check.py
```

`setup_check.py` runs the offline chain: imports, a small Cook's membrane
solve against the reference band, and the unit demo replayed from recordings.
It needs no key.

## A key

Everything that calls a model uses Google's Gemini through AI Studio, which
has a free tier: https://aistudio.google.com/apikey

```bash
export GEMINI_API_KEY=...
```

Any OpenAI-compatible endpoint works instead. See the top of `playground.py`
and `minifea/llm.py`.

## The playground

```bash
python playground.py             # both requests, one run each
python playground.py --runs 3    # a fix that works once is not a fix
python playground.py --anchor    # the code fix, without editing the file
```

The things to try, in order, are in the file's docstring.

## Look at the result

```bash
python look.py             # solve, open snapshot.png, then ask the model about it
python look.py --runs 5    # ask five times; count how often it names the artifact
```

Three one-line edits at the top of the file, marked EDIT ME: the element
formulation, whether the plot is averaged, and the question. Each changes
what the model says, and one of them changes what it can see.

## Slides

The talk itself is in `slides/`, as a Quarto reveal.js deck with its figures
and theme. To build it:

```bash
cd slides && quarto render talk.qmd
```

That writes `slides/talk.html`. The AMPlify demo video is not included; the
slide that plays it renders blank.

## Units

`minifea` is SI throughout (m, Pa, N). The unit failures the demos show live
in the handoff between agents, not in the solver.
