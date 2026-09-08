"""Checks on the unit consistency pipeline.

The stages that matter most are the ones a model cannot influence: grounding,
the unit-system choice, and the conversion arithmetic. Those are tested
directly with hand-built quantities, so that no test depends on what a model
happened to return on the day.
"""

import math

import pytest

from minifea.units import (UNIT_SYSTEMS, Quantity, UnitError, check_plausibility,
                           convert_to_system, determine_unit_system,
                           extract_quantities, to_si, validate_grounding)

STEEL_MM = [
    Quantity(48.0, "mm", "48 mm", "length"),
    Quantity(44.0, "mm", "44 mm", "length"),
    Quantity(1.0, "mm", "1 mm", "thickness"),
    Quantity(200.0, "GPa", "200 GPa", "modulus"),
    Quantity(0.3, None, "0.3", "poisson"),
    Quantity(7850.0, "kg/m^3", "7850 kg/m^3", "density"),
    Quantity(100.0, "N", "100 N", "force"),
]


# --- stage 1 -------------------------------------------------------------

def test_extraction_finds_values_units_and_bare_numbers():
    found = extract_quantities(
        "Panel 48 mm wide, 1 mm thick. E = 200 GPa, nu = 0.3, "
        "density 7850 kg/m^3, load 1.5e3 N.")
    assert [(q.value, q.unit) for q in found] == [
        (48.0, "mm"), (1.0, "mm"), (200.0, "GPa"), (0.3, None),
        (7850.0, "kg/m^3"), (1500.0, "N")]


def test_extraction_prefers_the_longest_unit_spelling():
    """'kg/m^3' must not be read as a bare number followed by junk."""
    found = extract_quantities("density 7850 kg/m^3")
    assert len(found) == 1 and found[0].unit == "kg/m^3"


# --- stage 3 -------------------------------------------------------------

def test_grounding_accepts_values_present_in_the_text():
    validate_grounding(STEEL_MM, STEEL_MM)


def test_grounding_rejects_an_invented_value():
    invented = Quantity(200.0, "GPa", "", "modulus")
    inventory = [Quantity(10.0, "mm", "10 mm", "length"),
                 Quantity(100.0, "MPa", "100 MPa", "stress")]
    with pytest.raises(UnitError, match="Ungrounded quantity"):
        validate_grounding(inventory, inventory + [invented])


def test_grounding_rejects_a_value_whose_unit_was_altered():
    """Same number, different unit, is a different quantity."""
    inventory = [Quantity(200.0, "GPa", "200 GPa", "modulus")]
    altered = [Quantity(200.0, "MPa", "", "modulus")]
    with pytest.raises(UnitError, match="Ungrounded quantity"):
        validate_grounding(inventory, altered)


# --- stage 4 -------------------------------------------------------------

@pytest.mark.parametrize("length_mm, expected", [
    (48.0, "SI-mm"),      # 0.048 m is too small for SI; 48 mm is human-scale
    (1.0, "SI-mm"),       # 1 mm part
    (1500.0, "SI"),       # a 1.5 m beam suits both; SI wins on preference
    (12000.0, "SI"),      # 12 m
])
def test_unit_system_follows_the_geometry(length_mm, expected):
    quantities = [Quantity(length_mm, "mm", "", "length")]
    system, reason = determine_unit_system(quantities)
    assert system == expected
    assert "characteristic length" in reason


def test_the_tie_break_is_a_stated_preference_not_dict_order():
    """A 1.5 m beam is human-scale in both systems, so the rule must decide.

    Guarding this because the tie-break used to fall out of dict ordering,
    which would have made a physics decision depend on where a key was typed.
    """
    from minifea.units import SYSTEM_PREFERENCE
    assert SYSTEM_PREFERENCE[0] == "SI"
    system, _ = determine_unit_system([Quantity(1.5, "m", "", "length")])
    assert system == "SI"


def test_same_part_written_two_ways_lands_in_the_same_system():
    """48 mm and 0.048 m are the same panel and must not diverge."""
    in_mm = determine_unit_system([Quantity(48.0, "mm", "", "length")])[0]
    in_m = determine_unit_system([Quantity(0.048, "m", "", "length")])[0]
    assert in_mm == in_m == "SI-mm"


def test_unit_system_needs_a_length():
    with pytest.raises(UnitError, match="No geometric dimension"):
        determine_unit_system([Quantity(200.0, "GPa", "", "modulus")])


# --- stage 5 -------------------------------------------------------------

def test_conversion_into_the_mm_system():
    rows = convert_to_system(STEEL_MM, "SI-mm")
    converted = {q.role: (value, unit) for q, value, unit in rows}
    assert converted["modulus"] == (pytest.approx(200000.0), "MPa")
    assert converted["density"][0] == pytest.approx(7.85e-9)
    assert converted["density"][1] == "Mg/mm^3"
    assert converted["force"] == (pytest.approx(100.0), "N")
    assert converted["poisson"][0] == pytest.approx(0.3)


def test_conversion_keeps_every_quantity_not_one_per_role():
    """Two lengths must survive as two rows.

    Keying the output by role collapses them and silently drops all but the
    last, which is easy to miss because the result still looks reasonable.
    """
    rows = convert_to_system(STEEL_MM, "SI-mm")
    assert len(rows) == len(STEEL_MM)
    lengths = [value for q, value, _ in rows if q.role == "length"]
    assert sorted(lengths) == [44.0, 48.0]


def test_mg_and_tonne_are_the_same_unit():
    """1 Mg = 1000 kg = 1 tonne. Both spellings are accepted on input."""
    as_mg = Quantity(7.85e-9, "Mg/mm^3", "", "density")
    as_tonne = Quantity(7.85e-9, "tonne/mm^3", "", "density")
    assert to_si(as_mg) == pytest.approx(to_si(as_tonne)) == pytest.approx(7850.0)


def test_mm_system_reports_density_as_mg():
    assert UNIT_SYSTEMS["SI-mm"]["density"] == "Mg/mm^3"


def test_the_mm_system_mass_unit_is_forced_by_length_and_force():
    """1 N = 1 Mg * mm / s^2, which is why kg/mm^3 would be inconsistent.

    Checked as arithmetic rather than asserted in a comment: one megagram
    accelerated at one mm/s^2 has to come out as exactly one newton.
    """
    megagram_kg = 1e3
    mm_per_s2 = 1e-3
    assert megagram_kg * mm_per_s2 == pytest.approx(1.0)


def test_conversion_rejects_a_unit_of_the_wrong_dimension():
    with pytest.raises(UnitError, match="not a pressure unit"):
        to_si(Quantity(200.0, "mm", "", "modulus"))


def test_conversion_rejects_a_missing_unit():
    with pytest.raises(UnitError, match="no unit was given"):
        to_si(Quantity(200.0, None, "", "modulus"))


def test_dimensionless_role_rejects_a_unit():
    with pytest.raises(UnitError, match="dimensionless but was given a unit"):
        to_si(Quantity(0.3, "MPa", "", "poisson"))


# --- stage 6 -------------------------------------------------------------

def test_plausibility_catches_a_modulus_that_converted_perfectly():
    """70 Pa is arithmetically fine and physically absurd."""
    problems = check_plausibility([Quantity(70.0, "Pa", "70 Pa", "modulus")])
    assert len(problems) == 1 and "outside the plausible range" in problems[0]
    assert "factor of 1000" in problems[0]


def test_plausibility_catches_an_impossible_poisson_ratio():
    problems = check_plausibility([Quantity(0.7, None, "0.7", "poisson")])
    assert len(problems) == 1 and "[0, 0.5)" in problems[0]


def test_plausibility_reports_every_problem_not_just_the_first():
    problems = check_plausibility([Quantity(70.0, "Pa", "", "modulus"),
                                   Quantity(0.7, None, "", "poisson")])
    assert len(problems) == 2


def test_plausibility_passes_a_correct_set():
    assert check_plausibility(STEEL_MM) == []
