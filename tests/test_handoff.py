"""Checks on the two-agent handoff demonstration.

The measured behavior of a model is not something a test can pin down, so
what is checked here is the machinery around it: that the shared state says
what it claims to, that a deck is judged on internal consistency rather than
on matching one preferred unit system, and that the whole demonstration runs
offline from recorded responses.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from minifea.handoff import SharedState, _split_reply, classify_deck
from minifea.llm import RECORDINGS, LLMClient

REPO = Path(__file__).resolve().parent.parent


def test_bare_state_names_no_unit_system():
    rendered = SharedState().render()
    assert "unit_system" not in rendered
    assert "geometry" in rendered and "named surfaces" in rendered


def test_state_renders_the_unit_system_when_it_has_one():
    assert "unit_system: mm-N-MPa" in SharedState(unit_system="mm-N-MPa").render()


def test_bounding_box_is_rendered_without_a_unit():
    """The point of the second beat: the box is bare numbers.

    If a unit ever gets appended here the demonstration stops demonstrating
    anything, so it is worth a test rather than a comment.
    """
    rendered = SharedState(bounding_box="(0,0,0) to (10, 2, 2)").render()
    assert "(0,0,0) to (10, 2, 2)" in rendered
    assert "mm" not in rendered


def test_a_millimetre_deck_is_consistent():
    verdict, why = classify_deck(
        {"youngs_modulus": 200000.0, "poisson": 0.3, "traction": 100.0})
    assert verdict == "consistent" and "MPa" in why


def test_pascal_stresses_against_millimetre_geometry_are_inconsistent():
    verdict, why = classify_deck(
        {"youngs_modulus": 2e11, "poisson": 0.3, "traction": 1e8})
    assert verdict == "inconsistent" and "1e6" in why


def test_a_deck_is_judged_on_consistency_not_on_a_preferred_system():
    """Pa values are only wrong *relative to* millimeter geometry.

    Guarding a mistake made repeatedly while building this: checking only the
    modulus, and calling a perfectly valid SI deck wrong because it was not
    the system expected.
    """
    verdict, _ = classify_deck({"youngs_modulus": 2e11, "poisson": 0.3,
                                "traction": 1e8}, geometry_unit="m")
    assert verdict == "inconsistent"  # still mm-keyed thresholds
    assert classify_deck({"youngs_modulus": 2e11})[0] == "unparsable"


def test_recordings_exist_and_carry_real_conversations():
    recordings = json.loads(RECORDINGS.read_text())
    assert recordings, "no recordings captured"
    for entry in recordings.values():
        assert entry["messages"] and entry["response"]
        assert entry["provider"] and entry["model"]


def test_offline_client_replays_and_says_so():
    client = LLMClient(offline=True)
    assert "replayed" in client.description


def test_offline_client_explains_a_missing_recording():
    client = LLMClient(offline=True)
    with pytest.raises(RuntimeError, match="No recorded response"):
        client.complete("a system prompt never recorded", "nor this user turn")


def test_the_whole_demo_runs_with_no_api_key():
    """The condition most attendees will actually be in."""
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home())}
    result = subprocess.run([sys.executable, "unit_demo.py"], cwd=REPO,
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    for expected in ("THE GEOMETRY AGENT", "GIVE IT EVERYTHING",
                     "WRITE IT DOWN", "REJECTED by stage 3"):
        assert expected in result.stdout


def test_reply_splitting_keeps_the_reasoning_and_the_json():
    preamble, values = _split_reply(
        'I chose mm-N-MPa because the geometry is in millimeters.\n'
        '```json\n{"youngs_modulus": 200000, "poisson": 0.3, "traction": 100}\n```')
    assert preamble.startswith("I chose mm-N-MPa")
    assert values["youngs_modulus"] == 200000


def test_reply_splitting_survives_bare_json():
    preamble, values = _split_reply('{"youngs_modulus": 2e11, "poisson": 0.3, "traction": 1e8}')
    assert preamble == "" and values["traction"] == 1e8

