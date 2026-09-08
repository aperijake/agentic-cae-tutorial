"""Checks on the code half of the playground: the unit-system choice and the check.

The agents are model calls and are not tested here. What is tested is that
the code which chooses the unit system and judges the deck does what the
docstrings say, on exactly the two requests the playground ships with.
"""

import pytest

import playground as p

MM, M = p.REQUESTS["millimeter bar"], p.REQUESTS["meter bar"]
PASCALS = {"youngs_modulus": 2e11, "poisson": 0.3, "traction": 1e8}
MEGAPASCALS = {"youngs_modulus": 200000, "poisson": 0.3, "traction": 100}


def test_the_two_requests_resolve_to_different_systems():
    assert p.choose_unit_system(MM) == "mm-N-MPa"
    assert p.choose_unit_system(M) == "m-N-Pa"


def test_stresses_and_bare_numbers_are_not_lengths():
    with pytest.raises(ValueError, match="no unit"):
        p.choose_unit_system("E=200 GPa, 100 MPa, nu=0.3, x=10 face")


def test_mixed_length_units_are_refused():
    with pytest.raises(ValueError, match="cannot choose"):
        p.choose_unit_system("a bar 10mm long and 0.002 m wide")


def test_request_stresses_convert_into_the_chosen_system():
    assert p.convert_request_values(MM, "mm-N-MPa") == [100.0, 200000.0]
    assert p.convert_request_values(M, "m-N-Pa") == [1e8, 2e11]


def test_pascal_deck_is_wrong_for_millimeter_geometry_and_right_for_meters():
    assert p.compare_values(PASCALS, p.convert_request_values(MM, "mm-N-MPa"), "mm-N-MPa") is False
    assert p.compare_values(PASCALS, p.convert_request_values(M, "m-N-Pa"), "m-N-Pa") is True


def test_megapascal_deck_is_right_for_millimeter_geometry_and_wrong_for_meters():
    assert p.compare_values(MEGAPASCALS, p.convert_request_values(MM, "mm-N-MPa"), "mm-N-MPa") is True
    assert p.compare_values(MEGAPASCALS, p.convert_request_values(M, "m-N-Pa"), "m-N-Pa") is False


def test_missing_or_non_numeric_values_fail_the_check():
    expected = p.convert_request_values(MM, "mm-N-MPa")
    assert p.compare_values({}, expected, "mm-N-MPa") is False
    assert p.compare_values({"youngs_modulus": "200 GPa", "traction": 100}, expected, "mm-N-MPa") is False


def test_the_check_shows_its_work(capsys):
    p.compare_values(PASCALS, p.convert_request_values(MM, "mm-N-MPa"), "mm-N-MPa")
    out = capsys.readouterr().out
    assert "allowed values in mm-N-MPa: 100, 200000" in out
    assert "model wrote: 2e+11, 1e+08" in out
    assert "WRONG" in out


def test_the_setup_agent_is_given_the_state_then_its_task():
    given = p.format_setup_input({"unit_system": "mm-N-MPa", "geometry": "bar"}, "Write the deck.")
    assert given == ("Shared analysis state:\n  unit_system: mm-N-MPa\n  geometry: bar"
                     "\n\nWrite the deck.")


def test_json_is_found_inside_code_fences_and_prose():
    assert p.as_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert p.as_json('Sure. {"a": 1} Done.') == {"a": 1}
    assert p.as_json("no json here") == {}


def test_mixed_units_are_refused_before_any_agent_runs(monkeypatch):
    monkeypatch.setattr(p, "ask", lambda *a: (_ for _ in ()).throw(AssertionError("model called")))
    with pytest.raises(ValueError, match="cannot choose"):
        p.run_one_request("a bar 10mm x 0.002 m, E=200 GPa")
