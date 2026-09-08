"""A unit consistency pipeline: six stages, exactly one of which is the model.

FEA solvers are dimensionless. If the geometry is in millimeters, the modulus
has to be in MPa and the density in Mg/mm^3, or the answer is wrong by a
factor of a thousand and the contour plot looks entirely reasonable.

(1 Mg = 1000 kg, the same unit some codes and textbooks call a tonne. The mm
system forces it: 1 N = 1 kg m/s^2 = 1 Mg mm/s^2, so once length is mm and
force is N, the mass unit is fixed and it is not the kilogram. That constraint
is exactly what a model has to work out, and often does not.)

Language models are good at reading a problem statement and working out that
"70 GPa" is a Young's modulus rather than a yield stress. They are much worse
at carrying that number through a unit conversion without dropping a factor,
and worse still at noticing when they have. So the model is given exactly one
job here -- deciding what each number *means* -- and every other stage is
ordinary code:

    1. extract      regex inventory of every (value, unit) literally in the text
    2. interpret    the model labels each inventory entry with a physical role
    3. ground       every labeled value must trace back to the inventory
    4. system       choose the target unit system from the geometry's scale
    5. convert      lookup-table arithmetic into that system
    6. plausibility check the converted values against physical bounds

Stage 3 is the one worth dwelling on. Stage 2 is the only place a model can
invent something, so stage 3 checks that it did not: the model is allowed to
be a source of *labels*, never a source of *numbers*. A value that does not
appear in the text is rejected even when it is a perfectly sensible guess.
"""

import json
import math
import re
from dataclasses import dataclass, field

# Conversion factors into SI base units, by dimension. Deliberately a plain
# lookup table: this is arithmetic that must be exactly right, so it is not
# somewhere to be clever.
_UNITS = {
    "length": {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3,
               "in": 0.0254, "ft": 0.3048},
    "pressure": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9,
                 "psi": 6894.757293168, "ksi": 6894757.293168},
    "force": {"N": 1.0, "kN": 1e3, "MN": 1e6, "lbf": 4.4482216152605},
    # 1 Mg (megagram) = 1000 kg, which is the same unit as a tonne. Both
    # spellings are accepted on input because real decks use both; Mg is what
    # gets written back out, since it is the SI-prefixed name for it.
    "density": {"kg/m^3": 1.0, "kg/m3": 1.0, "g/cm^3": 1e3, "g/cm3": 1e3,
                "Mg/mm^3": 1e12, "Mg/mm3": 1e12,
                "tonne/mm^3": 1e12, "tonne/mm3": 1e12, "t/mm^3": 1e12},
}

# Which dimension each physical role is measured in, and the plausible range
# of that role in SI base units. The ranges are wide on purpose: this stage
# catches order-of-magnitude errors, not questionable engineering.
_ROLES = {
    "length":       ("length",   1e-6, 1e3),
    "thickness":    ("length",   1e-6, 1e3),
    "modulus":      ("pressure", 1e6, 1e13),
    "yield_stress": ("pressure", 1e4, 1e11),
    "stress":       ("pressure", 1e-3, 1e11),
    "pressure":     ("pressure", 1e-3, 1e11),
    "force":        ("force",    1e-6, 1e12),
    "density":      ("density",  1e0, 1e6),
    "poisson":      (None,       0.0, 0.5),
    "dimensionless": (None,     -math.inf, math.inf),
}

# Target unit systems. A solver needs one consistent set; which one is a
# choice, and stage 4 makes it from the geometry rather than from the prose.
UNIT_SYSTEMS = {
    "SI":     {"length": "m",  "pressure": "Pa",  "force": "N", "density": "kg/m^3"},
    "SI-mm":  {"length": "mm", "pressure": "MPa", "force": "N", "density": "Mg/mm^3"},
}

# The order stage 4 tries them in. Both systems suit a part between 1 m and
# 1000 m, so something has to break the tie, and it should be a stated
# preference rather than whichever key happens to come first in a dict.
SYSTEM_PREFERENCE = ("SI", "SI-mm")

ROLE_NAMES = sorted(_ROLES)


class UnitError(ValueError):
    """A unit problem that must stop the run rather than be passed downstream."""


@dataclass
class Quantity:
    """One number found in the problem statement.

    ``source`` is the exact substring it was found in, which is what makes
    grounding possible: a value with no source did not come from the user.
    """
    value: float
    unit: str | None
    source: str
    role: str | None = None

    def __str__(self):
        return f"{self.value:g} {self.unit}" if self.unit else f"{self.value:g}"


@dataclass
class UnitResolution:
    """The output of the pipeline, plus the trail of how it got there."""
    system: str
    quantities: list[Quantity]
    converted: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# Stage 1 -- extraction
# --------------------------------------------------------------------------

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_ALL_UNITS = sorted({u for table in _UNITS.values() for u in table},
                    key=len, reverse=True)
_UNIT_PATTERN = "|".join(re.escape(u) for u in _ALL_UNITS)
_QUANTITY_RE = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT_PATTERN})?(?![A-Za-z0-9/^])")


def extract_quantities(text: str) -> list[Quantity]:
    """Every number in ``text``, with its unit if it has one.

    Regex, not the model. This builds the inventory that stage 3 checks
    against, so it has to be a mechanical reading of the text: if the model
    were asked to do it, there would be nothing independent left to check.

    Numbers with no unit are kept too. Poisson's ratio has no unit and still
    has to be carried through, and a bare number that the model later claims
    is a modulus is exactly the kind of thing worth catching.
    """
    found = []
    for match in _QUANTITY_RE.finditer(text):
        found.append(Quantity(value=float(match.group("value")),
                              unit=match.group("unit"),
                              source=match.group(0).strip()))
    return found


# --------------------------------------------------------------------------
# Stage 3 -- grounding
# --------------------------------------------------------------------------

def validate_grounding(inventory: list[Quantity],
                       interpreted: list[Quantity]) -> None:
    """Check that the model labeled the text rather than adding to it.

    Every interpreted quantity must match an inventory entry in both value and
    unit. Raises on the first violation, naming the value, because a
    hallucinated number is not something to warn about and continue past.

    This is the whole anti-hallucination mechanism, and it is string and float
    comparison. The model gets to say what a number is for; it does not get to
    say what the numbers are.
    """
    available = [(q.value, q.unit) for q in inventory]
    for quantity in interpreted:
        key = (quantity.value, quantity.unit)
        if key not in available:
            raise UnitError(
                f"Ungrounded quantity: the interpretation produced "
                f"{quantity} (as {quantity.role or 'unlabeled'}), which does "
                f"not appear in the problem statement. Values found in the "
                f"text were: {', '.join(str(q) for q in inventory)}. "
                "The model may only assign meaning to values the user wrote; "
                "it may not supply values of its own.")


# --------------------------------------------------------------------------
# Stage 4 -- unit system
# --------------------------------------------------------------------------

def determine_unit_system(quantities: list[Quantity]) -> tuple[str, str]:
    """Choose a target system from the size of the geometry.

    The geometry decides, not the prose, because the geometry is the thing
    that can be measured. A part described as "48 mm" and one described as
    "0.048 m" are the same part and should land in the same system.

    The rule: pick the first system in ``SYSTEM_PREFERENCE`` in which the
    characteristic length is a human-scale number, in [1, 1000). A 48 mm panel
    is 0.048 in SI and 48 in SI-mm, so SI-mm wins. A 1.5 m beam qualifies in
    both -- 1.5 m or 1500 mm -- and SI wins on preference.

    Returns (system name, the reason), because a choice made silently is a
    choice nobody checks.
    """
    lengths = [q for q in quantities if q.role in ("length", "thickness")]
    if not lengths:
        raise UnitError(
            "No geometric dimension was identified, so there is nothing to "
            "set the unit system from. A solver needs one consistent system; "
            "state at least one length, or choose a system explicitly.")

    characteristic = max(to_si(q) for q in lengths)
    for name in SYSTEM_PREFERENCE:
        system = UNIT_SYSTEMS[name]
        scaled = characteristic / _UNITS["length"][system["length"]]
        if 1.0 <= scaled < 1000.0:
            return name, (f"characteristic length {characteristic:g} m is "
                          f"{scaled:g} {system['length']}")
    return "SI", (f"characteristic length {characteristic:g} m fits no "
                  "preferred system; defaulting to SI")


# --------------------------------------------------------------------------
# Stage 5 -- conversion
# --------------------------------------------------------------------------

def to_si(quantity: Quantity) -> float:
    """Convert one quantity to SI base units. Pure table lookup."""
    if quantity.role is None:
        raise UnitError(f"Quantity {quantity} has no role, so its dimension "
                        "is unknown and it cannot be converted.")
    if quantity.role not in _ROLES:
        raise UnitError(
            f"Unknown role {quantity.role!r} for {quantity}. Known roles: "
            f"{', '.join(ROLE_NAMES)}.")

    dimension = _ROLES[quantity.role][0]
    if dimension is None:
        if quantity.unit is not None:
            raise UnitError(
                f"{quantity.role} is dimensionless but was given a unit "
                f"({quantity.unit}) in {quantity!s}.")
        return quantity.value
    if quantity.unit is None:
        raise UnitError(
            f"{quantity} was labeled {quantity.role}, which is measured in "
            f"{dimension}, but no unit was given in the problem statement. "
            "An unlabeled number cannot be converted; state the unit.")
    if quantity.unit not in _UNITS[dimension]:
        raise UnitError(
            f"{quantity.unit!r} is not a {dimension} unit, but {quantity} was "
            f"labeled {quantity.role}. Known {dimension} units: "
            f"{', '.join(_UNITS[dimension])}.")
    return quantity.value * _UNITS[dimension][quantity.unit]


def convert_to_system(quantities: list[Quantity],
                      system: str) -> list[tuple[Quantity, float, str]]:
    """Convert every quantity into ``system``.

    Returns one (quantity, value, unit) row per input, in order -- not a dict
    keyed by role. A problem statement routinely carries several lengths, and
    collapsing them into one entry per role silently discards all but the last
    of them.

    The model is not asked to multiply anything. Anywhere a number has to be
    exactly right, take the arithmetic away from the model.
    """
    if system not in UNIT_SYSTEMS:
        raise UnitError(f"Unknown unit system {system!r}; "
                        f"known systems: {', '.join(UNIT_SYSTEMS)}.")
    target = UNIT_SYSTEMS[system]
    rows = []
    for quantity in quantities:
        si_value = to_si(quantity)
        dimension = _ROLES[quantity.role][0]
        if dimension is None:
            rows.append((quantity, si_value, ""))
        else:
            unit = target[dimension]
            rows.append((quantity, si_value / _UNITS[dimension][unit], unit))
    return rows


# --------------------------------------------------------------------------
# Stage 6 -- plausibility
# --------------------------------------------------------------------------

def check_plausibility(quantities: list[Quantity]) -> list[str]:
    """Compare each converted value against physical bounds, in SI.

    Stage 5 guarantees the arithmetic was done right. It cannot notice that
    the input was absurd -- a steel modulus of 70 Pa converts perfectly. This
    is the last stage that can catch a value that is self-consistent and
    still wrong.

    Returns a list of complaints rather than raising: several may be true at
    once, and seeing all of them is more useful than seeing the first.
    """
    problems = []
    for quantity in quantities:
        low, high = _ROLES[quantity.role][1:]
        si_value = to_si(quantity)
        if not low <= si_value <= high:
            problems.append(
                f"{quantity.role} of {quantity} is {si_value:g} in SI base "
                f"units, outside the plausible range [{low:g}, {high:g}]. "
                + ("Poisson's ratio must be in [0, 0.5)."
                   if quantity.role == "poisson" else
                   f"Check whether the unit is right: a factor of 1000 here "
                   f"is the usual cause."))
    return problems


# --------------------------------------------------------------------------
# Stage 2 -- interpretation: the only stage the model touches
# --------------------------------------------------------------------------

_INTERPRET_SYSTEM = """\
You label the numbers in an engineering problem statement.

Return JSON only, with this shape:
{"quantities": [{"value": <number>, "unit": <string or null>, "role": <string>}]}

Use exactly these roles: """ + ", ".join(ROLE_NAMES) + """

Rules:
- Report every numeric value that appears in the statement.
- Copy each value and unit exactly as written. Do not convert anything.
- Poisson's ratio has role "poisson" and unit null.
- If a number's purpose is unclear, use role "dimensionless".
"""


# The interpretation prompt somebody writes first. It is not a straw man -- it
# is careful, specific, and names the right roles. Its one extra instruction,
# "make sure the solver has everything it needs", is enough to make the model
# supply material properties the user never mentioned. Kept here because the
# demonstration needs it, and because it is the honest argument for stage 3:
# the strict prompt above happens to hold today, on this model. You cannot
# verify a prompt. You can verify a check.
PERMISSIVE_INTERPRET_SYSTEM = """\
You extract the material and geometry parameters an FEA solver needs from a
problem statement.

Return JSON only:
{"quantities": [{"value": <number>, "unit": <string or null>, "role": <string>}]}

Roles: """ + ", ".join(ROLE_NAMES) + """
Make sure the solver has everything it needs to run.
"""


def interpret_roles(text: str, client, system: str | None = None) -> list[Quantity]:
    """Ask the model what each number in ``text`` means.

    Note what is *not* asked for: no conversion, no unit system, no arithmetic.
    The model reads English and assigns meaning, which is the one part of this
    job it is genuinely better at than a regex.

    The model is given the raw statement rather than the stage-1 inventory, on
    purpose. Handing it the inventory would constrain it to labeling and
    leave stage 3 with nothing independent to check. Reading the text itself
    means its output can disagree with the inventory -- and catching that
    disagreement is the entire point of grounding.

    ``system`` overrides the prompt, which the demonstration uses to show what
    happens with a less careful one.
    """
    reply = client.complete(system or _INTERPRET_SYSTEM, text)
    payload = reply.strip()
    if payload.startswith("```"):  # models like to fence their JSON
        payload = payload.split("```")[1]
        payload = payload[4:] if payload.lower().startswith("json") else payload
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as err:
        raise UnitError(
            f"The interpretation stage did not return valid JSON: {err}. "
            f"Raw reply: {reply[:200]!r}") from err

    quantities = []
    for entry in parsed.get("quantities", []):
        try:
            quantities.append(Quantity(
                value=float(entry["value"]), unit=entry.get("unit"),
                source="", role=entry.get("role")))
        except (KeyError, TypeError, ValueError) as err:
            raise UnitError(
                f"Malformed quantity from the interpretation stage: "
                f"{entry!r} ({err})") from err
    return quantities


# --------------------------------------------------------------------------
# The pipeline, and the thing it is meant to be compared against
# --------------------------------------------------------------------------

_NAIVE_SYSTEM = """\
You are preparing input for a finite element solver. The solver is
dimensionless: every quantity must be given in one consistent unit system.

Read the problem statement, choose a consistent unit system, convert every
value into it, and return JSON only:
{"unit_system": <string>, "values": {<name>: <number>, ...}}
"""


def resolve_units(text: str, client) -> UnitResolution:
    """Run all six stages. Raises ``UnitError`` rather than passing a bad
    value downstream.

    This is the anchored path. Compare with :func:`resolve_units_naive`, which
    asks a model to do the whole job in one step -- the difference between the
    two is the entire argument for building it this way.
    """
    inventory = extract_quantities(text)                    # 1. extract
    if not inventory:
        raise UnitError(
            "No numeric quantities found in the problem statement, so there "
            "is nothing to check. State the geometry and material as numbers "
            "with units.")

    interpreted = interpret_roles(text, client)             # 2. interpret
    validate_grounding(inventory, interpreted)              # 3. ground
    system, reason = determine_unit_system(interpreted)     # 4. system
    rows = convert_to_system(interpreted, system)           # 5. convert
    problems = check_plausibility(interpreted)              # 6. plausibility
    if problems:
        raise UnitError("Implausible values after conversion:\n  "
                        + "\n  ".join(problems))

    return UnitResolution(
        system=system, quantities=interpreted,
        converted={f"{q.role}[{i}]": value for i, (q, value, _) in enumerate(rows)},
        notes=[f"unit system chosen: {system} ({reason})",
               f"{len(inventory)} quantities found, all grounded in the text"])


def resolve_units_naive(text: str, client) -> dict:
    """Ask the model to do the whole job, conversions included.

    This is the unanchored path, and it is not a straw man: it is what you get
    from an afternoon of wiring an LLM to a solver, and it usually works. When
    it does not, the failure is a plausible number in the right shape, which
    no schema and no solver will reject.
    """
    reply = client.complete(_NAIVE_SYSTEM, text)
    payload = reply.strip()
    if payload.startswith("```"):
        payload = payload.split("```")[1]
        payload = payload[4:] if payload.lower().startswith("json") else payload
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"unit_system": None, "values": {}, "raw": reply}
