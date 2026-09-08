"""Exercise 1.2 -- one tool, and the agent loop that drives it.

This file is the whole lecture. Everything else in this course -- MCP servers,
Claude Code, commercial agent frameworks -- is this loop with more plumbing.

The idea in one paragraph:
    A language model cannot run your code. It can only emit text. So we hand it
    a *menu* of functions it is allowed to request (the tool schemas), and when
    it asks for one, WE run it and hand the answer back. The model then decides
    what to do next: ask for another tool, or write the final answer. That
    back-and-forth is the loop, and the loop is what makes it an "agent" rather
    than a chatbot.

The tool here is Euler-Bernoulli cantilever tip deflection, delta = F*L^3/(3*E*I).
Simple on purpose: you already know the right answer, so you can watch the
mechanism instead of squinting at the physics.

Run:  python ex2_one_tool.py

Units: SI throughout (m, N, Pa).
"""

import json
import os

from openai import OpenAI

client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

# A runaway loop is a real failure mode: a model can get stuck calling the same
# tool forever. Always bound it.
MAX_TURNS = 8


# ---------------------------------------------------------------------------
# 1. The tool itself -- ordinary Python. Nothing about it knows an LLM exists.
# ---------------------------------------------------------------------------


def cantilever_tip_deflection(F: float, L: float, E: float,
                              width: float, height: float) -> float:
    """Euler-Bernoulli tip deflection of an end-loaded rectangular cantilever.

    F: point load at the free end [N]
    L: beam length [m]
    E: Young's modulus [Pa]
    width:  cross-section width  (horizontal, perpendicular to the load) [m]
    height: cross-section height (vertical, along the load) [m]

    Returns deflection magnitude [m].

    Design note -- why width/height and not I: an earlier version of this tool
    took the second moment of area I directly. In live testing the model
    computed I wrong by a factor of 100 (1.33e-10 instead of 1.33e-8 m^4) and
    the tool faithfully amplified the garbage input into a confident,
    wrong answer. The model is good at deciding WHAT to compute and terrible
    at arithmetic, so the tool boundary belongs around the arithmetic. Give
    tools raw, hard-to-mangle inputs and do the math inside.
    """
    I = width * height**3 / 12.0
    return F * L**3 / (3.0 * E * I)


# ---------------------------------------------------------------------------
# 2. The tool *schema* -- the same function, described in a form the model reads.
# ---------------------------------------------------------------------------
#
# This is the part that trips people up, so be deliberate about it. The schema is
# not code the model executes. It is documentation, sent along with every request,
# that tells the model: here is a function you may request, here is what it does,
# here are its arguments and their types. The model uses `description` fields the
# way a new grad student uses a manual page -- they are prompt text, not comments.
#
# Rules of thumb that matter far more than they look:
#   * State the UNITS. A model that guesses GPa vs Pa will hand you a 1000x error
#     and sound completely confident about it.
#   * Say what the function is FOR, not just what it computes. "Use this for
#     quick hand-check estimates" tells the model when to reach for it.
#   * Mark required arguments. Anything optional will eventually be omitted.
#   * Draw the boundary so the model passes raw facts (dimensions, loads) and
#     the tool does the arithmetic. Any derived quantity you make the model
#     compute (like I) is a place it will eventually slip a decimal -- see the
#     design note on the function above; that lesson cost us a factor of 100.
#
# The structure below (type/function/name/description/parameters) is OpenAI's
# function-calling format, and the `parameters` block is plain JSON Schema.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "cantilever_tip_deflection",
            "description": (
                "Compute the tip deflection of an end-loaded cantilever beam with "
                "a solid rectangular cross-section, using Euler-Bernoulli beam "
                "theory. The second moment of area is computed internally from "
                "the cross-section dimensions. Use this for quick analytical "
                "checks of slender beams. All inputs and outputs are SI: force "
                "in newtons, lengths in meters, modulus in pascals (steel is "
                "about 2.0e11 Pa, not 200), deflection in meters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "F": {"type": "number", "description": "Point load at the free end, in newtons (N)."},
                    "L": {"type": "number", "description": "Beam length, in meters (m)."},
                    "E": {"type": "number", "description": "Young's modulus, in pascals (Pa)."},
                    "width": {
                        "type": "number",
                        "description": "Cross-section width, perpendicular to the load, in meters (m).",
                    },
                    "height": {
                        "type": "number",
                        "description": "Cross-section height, along the load direction, in meters (m).",
                    },
                },
                "required": ["F", "L", "E", "width", "height"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# 3. The conversation. This list IS the agent's entire memory.
# ---------------------------------------------------------------------------
#
# The API is stateless. Every call ships the full message list back to the server.
# So "the agent remembers what it did" really means "we appended to this list."
# Nothing else is going on.

messages = [
    {
        "role": "system",
        "content": (
            "You are a structural analysis assistant. Use the provided tools for "
            "any numerical calculation rather than doing arithmetic yourself. "
            "State units in your final answer."
        ),
    },
    {
        "role": "user",
        "content": (
            "I have a steel cantilever, 0.2 m long, with a 0.02 m x 0.02 m square "
            "cross-section, carrying a 1 kN load at the free end. What is the tip "
            "deflection?"
        ),
    },
]


# ---------------------------------------------------------------------------
# 4. The loop.
# ---------------------------------------------------------------------------

for turn in range(MAX_TURNS):
    print(f"\n--- turn {turn + 1} ---")

    # Send the whole conversation plus the tool menu. `tools=` is the only thing
    # that separates this from the ex1 chatbot call.
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    assistant_message = response.choices[0].message

    # THE TERMINATION CONDITION. The model either asks for tools or it doesn't.
    # No tool calls means it believes it has everything it needs, and the content
    # is the final answer. That is the only clean way out of this loop.
    if not assistant_message.tool_calls:
        print("\nFinal answer:")
        print(assistant_message.content)
        break

    # Otherwise: the model requested one or more tool calls. Note that we append
    # the assistant's message to the history FIRST, before running anything. The
    # API requires the request and its results to appear as a matched pair --
    # a tool result with no preceding request is a protocol error.
    #
    # We rebuild it as a plain dict so you can see exactly what goes on the wire.
    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content or "",
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        # NOTE: `arguments` arrives as a JSON *string*, not a dict.
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in assistant_message.tool_calls
            ],
        }
    )

    # Now run each requested call. A model may request several at once (here there
    # is only one tool, but it could still ask for two beams in one turn), and
    # every single one needs a reply.
    for tool_call in assistant_message.tool_calls:
        arguments = json.loads(tool_call.function.arguments)
        print(f"tool call: {tool_call.function.name}({arguments})")

        # Dispatch. With one tool this is trivial; ex3 generalizes it.
        # We catch exceptions instead of letting them escape, because a readable
        # error string fed back to the model is something it can recover from --
        # a traceback that kills the process is not.
        try:
            result = cantilever_tip_deflection(**arguments)
            result_text = json.dumps({"tip_deflection_m": result})
        except Exception as exc:
            result_text = json.dumps({"error": f"{type(exc).__name__}: {exc}"})

        print(f"tool result: {result_text}")

        # THE HANDBACK. The result goes in as its own message with role "tool".
        #
        # Why a separate role instead of just telling the model the answer in a
        # user message? Because the model needs to distinguish "the human said
        # this" from "the machine computed this," and because `tool_call_id` is
        # the thread that ties this result to the specific request above. If the
        # model asked for three calls, you send three tool messages, each carrying
        # the id it is answering. Mismatched or missing ids are the single most
        # common bug when people write this loop by hand.
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_text,
            }
        )

    # Loop back around. The model now sees its own request AND the result, and
    # decides what to do next. That decision -- unprompted, based on what came
    # back -- is the whole of "agency."

else:
    # `for/else` runs only if we never hit `break`, i.e. we burned every turn
    # without the model settling on an answer.
    print(f"\nStopped after {MAX_TURNS} turns without a final answer.")
