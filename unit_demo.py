"""Where unit consistency actually breaks, and what fixes it.

Beats, matching the slides:

  1. Two isolated agents. The setup agent gets the shared state and its task,
     nothing else. Wrong, every time, the way production runs are wrong.
  2. Give it everything: forward the analyst's whole request and the engine's
     bounding box. Still wrong, every time. It does not connect them.
  3. Write the unit down -- on the geometry or as a field, same thing. Right
     every time.
  4. What has to happen for that field to be trustworthy: the pipeline.

Run it:

    python unit_demo.py            # straight through
    python unit_demo.py --step     # pause between beats, to run alongside slides

With no API key set it replays responses captured from a real model, so it
works on any laptop. To drive it live, get a free key at
https://aistudio.google.com/apikey and:

    export GEMINI_API_KEY=...
    python unit_demo.py

Other providers: MINIFEA_PROVIDER=openai|anthropic, MINIFEA_MODEL=<name>.
"""

import sys

from minifea.handoff import (SharedState, classify_deck, geometry_agent,
                             setup_agent)
from minifea.llm import LLMClient
from minifea.units import (PERMISSIVE_INTERPRET_SYSTEM, UnitError,
                           check_plausibility, convert_to_system,
                           determine_unit_system, extract_quantities,
                           interpret_roles, validate_grounding)

# The exemplar problem this is drawn from: a 10 mm bar with the modulus quoted
# in GPa. A consistent mm-N-MPa deck needs E = 200000 and traction = 100.
GEOMETRY_TASK = ("Create a rectangular bar 10mm x 2mm x 2mm. The bar extends "
                 "from x=0 to x=10mm.")
SETUP_TASK = ("Write the material and loading values. Fix the end at x=0, apply "
              "100 MPa tension at the x=10 face. Material: steel, E=200 GPa, nu=0.3.")
# What the setup agent gets when the analyst's request is forwarded whole.
FULL_REQUEST_TASK = (f'The analyst\'s original request: "{GEOMETRY_TASK} {SETUP_TASK}"'
                     "\n\nNow write the material and loading values for the deck.")

# What the engine reports for the part the geometry agent actually built
# (create brick x_interval 0 10 y_interval -1 1 z_interval -1 1): ten units
# long, centered on the x axis. No unit anywhere.
BBOX = "(0, -1, -1) to (10, 1, 1)"

# The full statement, for the pipeline section.
PIPELINE_PROMPT = ("Panel 48 mm wide, 44 mm tall, 1 mm thick. Steel, E = 200 GPa, "
                   "nu = 0.3, density 7850 kg/m^3. Apply 100 N shear on the right edge.")
GROUNDING_PROMPT = "Steel bar, 10 mm long, under 100 MPa applied stress."

# The handoff conditions, in the order the demo runs them. Every entry is
# (label, shared state, task, reason_first). capture_recordings.py records
# exactly these, so the offline replay matches the live run.
HANDOFF_CONDITIONS = [
    ("bare state",                     SharedState(),                    SETUP_TASK,        False),
    ("full request + bounding box",    SharedState(bounding_box=BBOX),   FULL_REQUEST_TASK, False),
    ("unit_system field",              SharedState(unit_system="mm-N-MPa"), SETUP_TASK,     False),
]

# Measured on gemini-2.5-flash at temperature 0, 6 runs per condition, with
# the setup agent in a fresh context each time. Re-measure before quoting.
MEASURED = {
    "bare":        "6/6 inconsistent",
    "everything":  "6/6 inconsistent",
    "unit_field":  "6/6 consistent",
}

STEP = "--step" in sys.argv


def rule(title):
    if STEP:
        input("\n  [enter] ")
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def show(label, values, note):
    verdict, why = classify_deck(values)
    mark = {"consistent": "ok ", "inconsistent": "XX ", "other": "?? "}.get(verdict, "?? ")
    print(f"  {mark}{label}")
    print(f"      {why}")
    print(f"      measured: {note}")


def main():
    client = LLMClient()
    print(f"model: {client.description}")
    print(f"\ngeometry agent's task: {GEOMETRY_TASK}")
    print(f"setup agent's task:    {SETUP_TASK}")
    conditions = dict((c[0], c) for c in HANDOFF_CONDITIONS)

    rule("1. THE GEOMETRY AGENT")
    print(f"  task: {GEOMETRY_TASK}\n")
    commands = geometry_agent(GEOMETRY_TASK, client)
    print("  it issued:\n    " + commands.strip().replace("\n", "\n    "))
    print("\n  Ten. Not ten millimeters: the engine has no such thing. The part")
    print("  in the CAD kernel is ten units long, and nothing anywhere says")
    print("  what a unit is.")

    rule("2. THE SETUP AGENT, IN A FRESH CONTEXT")
    print("  It never sees the geometry agent's conversation, only the shared")
    print("  state. That isolation is deliberate -- it is what stops engine")
    print("  entity ids leaking between phases and being reused wrongly.\n")
    _, state, task, _ = conditions["bare state"]
    print("  " + state.render().replace("\n", "\n  ") + "\n")
    show("no unit system in the state", setup_agent(state, task, client), MEASURED["bare"])
    print("\n  The geometry agent read '10mm', typed 10 into the engine, and")
    print("  moved on. The setup agent read '200 GPa' and converted to SI,")
    print("  like any sensible engineer. Neither did anything wrong alone.")
    print("  The deck is millimeters and pascals: out by a factor of a million,")
    print("  in a file no schema rejects.")
    print("\n  This is the dominant failure in our exemplar runs. Of 95 unit")
    print("  failures, 78 read: geometry=1x, E=1e+06x, traction=1e+06x.")

    rule("3. GIVE IT EVERYTHING")
    print("  Forward the analyst's whole request. Hand over the engine's")
    print("  bounding box. Now the setup agent sees:\n")
    _, state, task, _ = conditions["full request + bounding box"]
    print("  " + (state.render() + "\n\n" + task).replace("\n", "\n  ") + "\n")
    show("full request and bounding box",
         setup_agent(state, task, client), MEASURED["everything"])
    print("\n  It has everything it needs to see the part was built in")
    print("  millimeters: '10mm' in the request, '10' from the engine.")
    print("  It does not connect them.")

    rule("4. WRITE IT DOWN")
    print("  The engine returns bare numbers. The only thing in the system")
    print("  that ever knew the bar was in millimeters was the geometry agent,")
    print("  reading the request. Record it in the shared state:\n")
    _, state, task, _ = conditions["unit_system field"]
    print("  " + state.render().replace("\n", "\n  ") + "\n")
    show("unit_system: mm-N-MPa", setup_agent(state, task, client), MEASURED["unit_field"])
    print("\n  It is the only fix here that can be checked rather than hoped for.")
    print("  Not a better prompt and not a better model: this failure spans")
    print("  every model family in our runs, because it is not a property of")
    print("  the model.")

    rule("5. SO WHO DECIDES THAT FIELD, AND HOW WOULD YOU KNOW IT IS RIGHT?")
    print(f"  problem statement:\n    {PIPELINE_PROMPT}\n")
    inventory = extract_quantities(PIPELINE_PROMPT)
    print("  1. extract   (regex)      " + ", ".join(str(q) for q in inventory))
    interpreted = interpret_roles(PIPELINE_PROMPT, client)
    print("  2. interpret (the model)  "
          + ", ".join(f"{q}={q.role}" for q in interpreted))
    validate_grounding(inventory, interpreted)
    print("  3. ground    (code)       every value traces back to the text")
    system, reason = determine_unit_system(interpreted)
    print(f"  4. system    (code)       {system}  <-- the field from beat 4")
    print(f"                            {reason}")
    print("  5. convert   (code)")
    for quantity, value, unit in convert_to_system(interpreted, system):
        print(f"       {quantity.role:14s} {str(quantity):14s} -> {value:>12g} {unit}")
    problems = check_plausibility(interpreted)
    print("  6. plausible (code)       "
          + ("; ".join(problems) if problems else "all values within physical bounds"))
    print("\n  Decided once, from the geometry, by code. Then carried, not")
    print("  re-derived by whoever needs it next. The model was never asked")
    print("  to multiply anything.")

    rule("A NOTE ON STAGE 3, WHILE WE ARE HERE")
    print(f"  problem statement: {GROUNDING_PROMPT}")
    print("  It names a material and gives none of its properties.\n")
    print("  Our interpretation prompt tells the model to report only values")
    print("  present in the text, and today, on this model, that holds. Here")
    print("  is the same stage with the prompt somebody writes first, which")
    print("  differs by one line: 'make sure the solver has everything it")
    print("  needs to run.'\n")
    invented = interpret_roles(GROUNDING_PROMPT, client,
                               system=PERMISSIVE_INTERPRET_SYSTEM)
    print("  the model returned: " + ", ".join(f"{q}={q.role}" for q in invented))
    try:
        validate_grounding(extract_quantities(GROUNDING_PROMPT), invented)
        print("\n  -> grounding passed: it added nothing this time.")
    except UnitError as error:
        print(f"\n  -> REJECTED by stage 3:\n     {error}")
    print("\n  It does not add values every run; this one was recorded because")
    print("  it did. Intermittent is the point: you cannot rely on it either way.")
    print("\n  Those are good numbers for steel, which is what makes them")
    print("  dangerous. You cannot verify a prompt. You can verify a check.")


if __name__ == "__main__":
    main()
