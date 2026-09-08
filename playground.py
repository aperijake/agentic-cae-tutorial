"""Follow one request through three agents, then check their unit handling.

These are real model calls. CAD and input-deck creation are simulated;
no FE solver runs. Each call starts a fresh conversation.

Start with the prompts and requests below, then read run_one_request().
The helpers below that function explain the details when you need them.

Try:
    python playground.py --step
    python playground.py --runs 3
    python playground.py --anchor --step

Exercise:
1. Run both requests and inspect what the setup agent receives.
2. Add "Always use mm-N-MPa." to SETUP_AGENT. Predict what changes. Run. Undo.
3. Ask MAIN_AGENT to put the unit system in both tasks. Repeat runs. Undo.
4. Uncomment the shared_state assignment in step 5, or use --anchor.
   Inspect the setup input again. What no longer depends on the main agent?

Code chooses the system. The setup model converts the deck values.
Code independently converts reference values to check the model's output.
This checks stress-value membership, not geometry, property roles, or a full deck.

Needs openai and GEMINI_API_KEY. Connection settings are below the helpers.
"""

import json
import os
import re
import sys

# ---- 1. Instructions students can edit -------------------------------------

MAIN_AGENT = """\
You are the main agent. Read the analyst's request and write a task for each
sub-agent: one for the geometry agent, who builds the part, and one for the
setup agent, who writes the material and loading values.

Return JSON only:
{"geometry_task": "<text>", "setup_task": "<text>"}
"""

GEOMETRY_AGENT = """\
You are the geometry agent. Build the requested part in the CAD engine.
The engine takes bare numbers.

Return JSON only:
{"length": <number>, "width": <number>, "height": <number>}
"""

SETUP_AGENT = """\
You are the solver setup agent. Write the material and loading values for the
input deck, using the shared analysis state you are given.

Return JSON only:
{"youngs_modulus": <number>, "poisson": <number>, "traction": <number>}
"""

# ---- 2. Same physical problem, two ways to express lengths -----------------

REQUESTS = {
    "millimeter bar": (
        "Create a rectangular bar 10mm x 2mm x 2mm. The bar extends from x=0 "
        "to x=10mm. Fix the end at x=0, apply 100 MPa tension at the x=10 "
        "face. Material: steel, E=200 GPa, nu=0.3."),
    "meter bar": (
        "Create a rectangular bar 0.01 m x 0.002 m x 0.002 m. The bar extends "
        "from x=0 to x=0.01 m. Fix the end at x=0, apply 100 MPa tension at "
        "the x=0.01 face. Material: steel, E=200 GPa, nu=0.3."),
}

# ---- 3. One request, in execution order: read this first --------------------

def run_one_request(request, anchored=False):
    # Step 1: the analyst supplies the request.
    show("1. Analyst's request", request)

    # Step 2: CODE chooses a system from the length unit (mm or m).
    # We need this for the checker even when the agents are not told the system.
    unit_system = choose_unit_system(request)
    show("2. Code chooses the system", unit_system)

    # Step 3: MODEL splits the request into tasks. Only this call is directly
    # given the whole request; sub-agents receive whatever task text it writes.
    # json.dumps(..., indent=2) formats the dictionary for easy reading.
    tasks = ask_for_json(MAIN_AGENT, request)
    show("3. Main agent writes tasks", json.dumps(tasks, indent=2))

    # Step 4: MODEL writes dimensions; our fake CAD engine stores bare numbers.
    # .get() reads a dictionary entry; "" is the fallback if it is missing.
    geometry_task = tasks.get("geometry_task", "")
    dimensions = ask_for_json(GEOMETRY_AGENT, geometry_task)
    geometry = describe_geometry(dimensions)
    show("4. Geometry agent writes dimensions", json.dumps(dimensions, indent=2))

    # Step 5: CODE assembles the information the setup agent will receive.
    # A dictionary stores named values. Agents cannot see this Python object
    # automatically: we must put its contents into a message for them.
    shared_state = {"geometry": geometry}
    # THE STUDENT CHANGE: uncomment the following line to carry the system.
    # shared_state["unit_system"] = unit_system
    if anchored:  # --anchor makes the same change without editing this file.
        shared_state["unit_system"] = unit_system
    # The anchor stores the decision; it does not convert any numbers.
    # Turn the record into text and append the main agent's setup task.
    setup_task = tasks.get("setup_task", "")
    setup_input = format_setup_input(shared_state, setup_task)
    show("5. Everything the setup agent receives", setup_input)

    # Step 6: MODEL writes the deck values, including its unit conversions.
    model_values = ask_for_json(SETUP_AGENT, setup_input)
    show("6. Setup agent writes values", json.dumps(model_values, indent=2))

    # Step 7: CODE converts reference values, then checks the model's numbers.
    # These reference values are never sent to the model or used to repair its deck.
    expected_values = convert_request_values(request, unit_system)
    return compare_values(model_values, expected_values, unit_system)


# ---- 4. Helpers used above, in the same order ------------------------------

def choose_unit_system(request):
    """Teaching rule: mm -> mm-N-MPa; m -> m-N-Pa. Refuse mixed/missing units."""
    # Find mm or m after a number, e.g. "10mm". set() removes duplicates:
    # three dimensions in mm still mean just one distinct length unit.
    length_units = set(re.findall(r"\d\s*(mm|m)\b", request))
    if len(length_units) != 1:
        raise ValueError(f"cannot choose a unit system: lengths in {length_units or 'no unit'}")
    # Take the one remaining unit and look up our convention. Both use Newtons.
    length_unit = length_units.pop()
    systems = {"mm": "mm-N-MPa", "m": "m-N-Pa"}
    return systems[length_unit]


def ask_for_json(instructions, task):
    # The API returns text. Parse its JSON into a dictionary Python can read.
    reply_text = ask(instructions, task)
    return as_json(reply_text)


def describe_geometry(dimensions):
    """Fake CAD: keep the numbers the model wrote, with no unit attached."""
    length = dimensions.get("length", "?")
    width = dimensions.get("width", "?")
    height = dimensions.get("height", "?")
    return f"bar, meshed, bounding box (0, 0, 0) to ({length}, {width}, {height})"


def format_setup_input(shared_state, setup_task):
    # The setup agent sees these fields and its task, not the other conversations.
    # This function turns the Python dictionary into ordinary message text.
    shared = "Shared analysis state:\n"
    # Keep the original script's field order for comparable model inputs.
    for field in ("unit_system", "geometry"):
        if field in shared_state:
            shared += f"  {field}: {shared_state[field]}\n"
    return shared.rstrip("\n") + "\n\n" + setup_task


def convert_request_values(request, unit_system):
    """CODE converts the supplied stress quantities for the checker only.

    Example: 200 GPa -> 200000 MPa in mm-N-MPa.
    It does not decide which value is the modulus or the traction.
    """
    pascals_per_unit = {"GPa": 1e9, "MPa": 1e6, "Pa": 1.0}
    pascals_per_target_unit = {"mm-N-MPa": 1e6, "m-N-Pa": 1.0}
    # Extract (number, unit) pairs such as ("200", "GPa") from the request.
    # This teaching parser supports the notation used in these examples.
    quantities = re.findall(r"(\d+(?:\.\d+)?)\s*(GPa|MPa|Pa)\b", request)
    expected_values = []
    for value, unit in quantities:
        # float() turns extracted text into a number. Convert to Pa first,
        # then divide by the size of the target unit (1e6 Pa per MPa).
        value_in_pascals = float(value) * pascals_per_unit[unit]
        converted_value = value_in_pascals / pascals_per_target_unit[unit_system]
        expected_values.append(converted_value)
    return expected_values


def compare_values(model_values, expected_values, unit_system):
    """A limited check: each stress must match an allowed value.

    Swapped or duplicated values can pass. This is a unit-value check,
    not a check of material roles, geometry, Poisson's ratio, or solver correctness.
    """
    # Only these two fields are checked. A missing field becomes None and fails.
    wrote = [model_values.get("youngs_modulus"), model_values.get("traction")]
    passed = True
    for value in wrote:
        if not isinstance(value, (int, float)):
            passed = False
            continue
        # any() accepts a match to ANY allowed number, within a small tolerance.
        # It does not check roles: swapped modulus/traction or duplicate values
        # can pass. Geometry and Poisson's ratio are not checked here either.
        matches = any(abs(value - expected) <= 1e-6 * expected
                      for expected in expected_values)
        if not matches:
            passed = False
    show("7. Code checks stress values",
         f"allowed values in {unit_system}: {numbers(expected_values)}\n"
         f"model wrote: {numbers(wrote)}\n"
         f"{'ok' if passed else 'WRONG'}")
    return passed


# ---- Connection, printing, and command-line helpers ------------------------
# You can stop reading here during the walkthrough.

MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
KEY_ENV = "GEMINI_API_KEY"


STEP = False    # --step: wait for enter after each piece
QUIET = False   # later runs of the same request: verdict only

_client = None


def ask(instructions, message):
    """One model call, fresh context, temperature 0. Returns the reply text."""
    # Reuse the API connection, not a conversation. No chat history is stored.
    global _client
    if _client is None:
        from openai import OpenAI
        # Read the key from your terminal environment. This client talks to
        # Gemini through its OpenAI-compatible endpoint.
        _client = OpenAI(api_key=os.environ[KEY_ENV], base_url=BASE_URL)
    # Each call sends only these two messages: system = standing instructions;
    # user = this task and its context. Previous replies are not included.
    # Temperature 0 reduces sampling variation but does not guarantee correctness.
    reply = _client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": instructions},
                  {"role": "user", "content": message}])
    # The API returns a list of responses. Python starts numbering at zero.
    first_response = reply.choices[0]
    # Extract its text; ask_for_json() parses that text afterward.
    return first_response.message.content


def as_json(reply):
    """The JSON object in a reply, code fences and all. {} if there is none."""
    # Find the outer braces in case the model wraps its JSON in code fences.
    start, end = reply.find("{"), reply.rfind("}")
    try:
        # Slices exclude the ending position, so +1 includes the closing brace.
        json_text = reply[start:end + 1]
        return json.loads(json_text)
    except ValueError:
        print(f"  (no JSON in this reply: {reply.strip()[:160]!r})")
        return {}


def show(label, text):
    """Print one piece of the story; with --step, wait for enter."""
    if QUIET:
        return
    print(f"\n  {label}:")
    print("    " + text.replace("\n", "\n    "))
    if STEP:
        input("  [enter] ")


def numbers(values):
    return ", ".join(f"{v:g}" if isinstance(v, (int, float)) else repr(v) for v in values)


def main():
    import argparse

    global STEP, QUIET
    # argparse reads command-line options. __doc__ is the introductory text
    # at the top of this file, displayed when students run --help.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", action="store_true", help="pause after each step")
    parser.add_argument("--anchor", action="store_true", help="put the unit system in shared state")
    parser.add_argument("--runs", type=int, default=1, help="runs per request")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if KEY_ENV not in os.environ:
        sys.exit(f"Set {KEY_ENV} first. Free key: https://aistudio.google.com/apikey")
    STEP = args.step
    tally = {}
    for name, request in REQUESTS.items():
        print(f"\n== {name} ==")
        passed = 0
        for index in range(args.runs):
            QUIET = index > 0
            try:
                ok = run_one_request(request, anchored=args.anchor)
            except ValueError as refusal:
                print(f"Code refused: {refusal}. Ask the user.")
                break
            # True counts as 1 and False as 0: total the successful checks.
            passed += ok
            if QUIET:
                print(f"  run {index + 1}: {'ok' if ok else 'WRONG'}")
        QUIET = False
        tally[name] = passed
    print("\n" + "   ".join(f"{name}: {passed}/{args.runs} right"
                             for name, passed in tally.items()))


# Start when launched as a script, but not when imported by tests.
if __name__ == "__main__":
    main()
