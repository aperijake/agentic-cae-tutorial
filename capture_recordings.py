"""Capture real model responses so the demo runs without an API key.

Recordings are captured from live calls, never written by hand. Re-run this
whenever a prompt in unit_demo.py changes, with a key set:

    GEMINI_API_KEY=... python capture_recordings.py

Every response here is whatever the model returned on the day, with one
exception that is worth stating plainly. The grounding beat needs the
permissive prompt to invent a material property, and whether it does is
stochastic even at temperature 0. This script retries that one call, a
bounded number of times, until it records a run that invented something,
and the demo says on screen that the behavior is intermittent. The handoff
conditions are not retried: they reproduce 6/6 in each direction.
"""

import json

from minifea.llm import RECORDINGS, LLMClient
from minifea.handoff import geometry_agent, setup_agent, setup_agent_with_reasoning
from minifea.llm import _fingerprint
from minifea.units import (PERMISSIVE_INTERPRET_SYSTEM, UnitError,
                           extract_quantities, interpret_roles,
                           validate_grounding)
from unit_demo import (GEOMETRY_TASK, GROUNDING_PROMPT, HANDOFF_CONDITIONS,
                       PIPELINE_PROMPT)

def main():
    client = LLMClient(record=True, offline=False)
    if client.api_key is None:
        raise SystemExit(
            "No API key found. Set GEMINI_API_KEY (free tier at "
            "https://aistudio.google.com/apikey) and re-run.")
    print(f"capturing from {client.description}")

    geometry_agent(GEOMETRY_TASK, client)
    print("  recorded geometry agent")

    # The handoff conditions, exactly as the demo runs them.
    for label, state, task, reason_first in HANDOFF_CONDITIONS:
        if reason_first:
            setup_agent_with_reasoning(state, task, client)
        else:
            setup_agent(state, task, client)
        print(f"  recorded handoff: {label}")

    # The pipeline section.
    interpret_roles(PIPELINE_PROMPT, client)
    print("  recorded interpretation: pipeline")
    # The permissive prompt has to invent something for the beat to land.
    key = _fingerprint(client.model, PERMISSIVE_INTERPRET_SYSTEM,
                       [{"role": "user", "content": GROUNDING_PROMPT}])
    for attempt in range(1, 13):
        invented = interpret_roles(GROUNDING_PROMPT, client,
                                   system=PERMISSIVE_INTERPRET_SYSTEM)
        try:
            validate_grounding(extract_quantities(GROUNDING_PROMPT), invented)
        except UnitError:
            print(f"  recorded interpretation: grounding (permissive prompt), "
                  f"invented values on attempt {attempt}")
            break
        recordings = json.loads(RECORDINGS.read_text())
        recordings.pop(key, None)
        RECORDINGS.write_text(json.dumps(recordings, indent=2) + "\n")
        print(f"  attempt {attempt}: the permissive prompt added nothing, retrying")
    else:
        print("  WARNING: the permissive prompt never invented a value in 12 tries; "
              "the demo's grounding beat will show a pass")

    print(f"wrote {RECORDINGS}")
    print(f"  {len(json.loads(RECORDINGS.read_text()))} recordings total")


if __name__ == "__main__":
    main()
