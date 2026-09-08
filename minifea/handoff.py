"""Two agents, one analysis, and what has to travel between them.

This is a deliberately small stand-in for a phased multi-agent system: a
geometry agent that builds the part, and a solver-setup agent that writes the
material and loading block. The point being demonstrated is the handoff, so
everything else is stripped away.

The setup agent runs in a *fresh context*. It never sees the geometry agent's
conversation -- only the shared analysis state it is handed. That isolation is
deliberate and correct: it is what stops engine-native identifiers from one
phase leaking into another and being reused where they mean something else.

It also removes the geometry's unit system from view, and that is where this
goes wrong. The geometry agent reads "10mm", types 10 into the CAD engine and
moves on. The setup agent reads "E = 200 GPa", converts to SI like any
sensible engineer, and writes 2e11. Each is locally reasonable. Together they
produce a deck whose geometry is in millimeters and whose material is in
pascals, wrong by a factor of a million, in a file that no schema rejects.

The fix is not a better prompt or a better model. It is one more field in the
shared state.
"""

from dataclasses import dataclass, field

GEOMETRY_SYSTEM = """\
You are the geometry agent. Build the requested part in the CAD engine.
Reply with Cubit journal commands only, no commentary.
"""

SETUP_SYSTEM = """\
You are the solver setup agent. Write the material and loading values for the
input deck, using the shared analysis state you are given.

Return JSON only:
{"youngs_modulus": <number>, "poisson": <number>, "traction": <number>}
"""

# The same agent, told to think before it answers. One sentence is enough to
# change the outcome -- which is the point made about it in the demo: a
# prompt that works is not the same thing as a check that can be verified.
REASON_FIRST_SYSTEM = SETUP_SYSTEM.replace(
    "Return JSON only:",
    "First, in ONE sentence, state which unit system you chose and why. "
    "Then return the JSON:")


@dataclass
class SharedState:
    """What travels between phases. The agents share nothing else.

    ``unit_system`` is the field this whole demonstration is about. Without
    it, a downstream agent has no way to know what the geometry's numbers
    mean. ``bounding_box`` is what a CAD engine reports: bare numbers. "10" is
    10 of something, and unless the string carries a unit, nothing in the
    state says of what. A bounding box *labeled* "mm" works exactly as well
    as the field -- it is the same decision, written in a different place.
    """
    geometry: str = "volume_1 (bar), meshed, 40 hex elements"
    surfaces: tuple = ("fixed_end", "load_face")
    bounding_box: str | None = None
    unit_system: str | None = None

    def render(self) -> str:
        lines = ["Shared analysis state:"]
        if self.unit_system:
            lines.append(f"  unit_system: {self.unit_system}")
        lines.append(f"  geometry: {self.geometry}")
        if self.bounding_box:
            lines.append(f"  bounding box: {self.bounding_box}")
        lines.append(f"  named surfaces: {', '.join(self.surfaces)}")
        return "\n".join(lines)


def geometry_agent(task: str, client) -> str:
    """Build the part. Returns the journal commands it issued."""
    return client.complete(GEOMETRY_SYSTEM, task)


def _split_reply(reply: str) -> tuple[str, dict]:
    """Separate any prose the model wrote from the JSON object it returned."""
    import json

    start, end = reply.find("{"), reply.rfind("}")
    preamble = reply[:start] if start > 0 else ""
    preamble = preamble.split("```")[0].strip()
    if start < 0 or end < 0:
        return preamble, {"raw": reply[:120]}
    try:
        return preamble, json.loads(reply[start:end + 1])
    except Exception:
        return preamble, {"raw": reply[:120]}


def setup_agent(state: SharedState, task: str, client,
                system: str = SETUP_SYSTEM) -> dict:
    """Write the material block, seeing only ``state`` -- never the geometry
    agent's conversation. ``system`` selects answer-only or reason-first."""
    return _split_reply(client.complete(system, state.render() + "\n\n" + task))[1]


def setup_agent_with_reasoning(state: SharedState, task: str, client) -> tuple[str, dict]:
    """As :func:`setup_agent` under ``REASON_FIRST_SYSTEM``; also returns the
    sentence of reasoning the model gave before its answer."""
    return _split_reply(client.complete(REASON_FIRST_SYSTEM, state.render() + "\n\n" + task))


def classify_deck(values: dict, geometry_unit: str = "mm") -> tuple[str, str]:
    """Is the deck consistent with the unit the geometry was built in?

    Returns (verdict, explanation). The check is deliberately about internal
    consistency rather than about matching one preferred system: a deck in SI
    throughout is perfectly valid. What is not valid is millimeter geometry
    with pascal stresses, and that combination is what the exemplar runs
    actually produce.
    """
    E, traction = values.get("youngs_modulus"), values.get("traction")
    if E is None or traction is None:
        return "unparsable", f"no usable values in {values!r}"
    if abs(E - 200000.0) < 1.0 and abs(traction - 100.0) < 1e-6:
        return "consistent", "MPa throughout, matching the mm geometry"
    if abs(E - 2e11) < 1e6 and abs(traction - 1e8) < 1e3:
        return "inconsistent", (
            f"stresses in Pa (E={E:g}, traction={traction:g}) against "
            f"{geometry_unit} geometry: out by a factor of 1e6")
    return "other", f"E={E:g}, traction={traction:g}"


ONE_AGENT_SYSTEM = GEOMETRY_SYSTEM.replace(
    "Reply with Cubit journal commands only, no commentary.",
    "You also write the solver input deck when asked.")


def one_agent(geometry_task: str, setup_task: str, client) -> dict:
    """The same agent builds the part and then writes the deck, in one context.

    The comparison case for :func:`setup_agent`. Everything the geometry step
    decided is still visible when the material block is written -- including,
    implicitly, that the part was built with millimeter-valued coordinates.
    """
    import json

    messages = [{"role": "user", "content": geometry_task}]
    messages.append({"role": "assistant",
                     "content": client.chat(ONE_AGENT_SYSTEM, messages)})
    messages.append({"role": "user", "content": setup_task + "\n" + SETUP_SYSTEM})
    reply = client.chat(ONE_AGENT_SYSTEM, messages).strip()
    if reply.startswith("```"):
        reply = reply.split("```")[1]
        reply = reply[4:] if reply.lower().startswith("json") else reply
    try:
        return json.loads(reply)
    except Exception:
        return {"raw": reply[:120]}
