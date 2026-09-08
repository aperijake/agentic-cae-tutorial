"""Pre-session smoke test. Run: python setup_check.py

Checks the offline chain end to end: imports, a small Cook's membrane solve
against the published reference band, and the unit demo replayed from
recorded model responses. Needs no API key; says whether one is set.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


def main() -> int:
    print("1/3 imports...", end=" ", flush=True)
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        import openai  # noqa: F401
        import scipy  # noqa: F401
        from minifea import cooks_membrane, probe, solve_hex
    except ImportError as exc:
        print(f"\nFAILED: {exc}\nDid you run pip install -e '.[dev]' inside the "
              "activated virtual environment?")
        return 1
    print("ok")

    print("2/3 Cook's membrane, hex8 B-bar, nu = 0.4999...", end=" ", flush=True)
    mesh = cooks_membrane(24)
    fixed = [{"node_set": "clamped", "components": (0, 1, 2)},
             {"node_set": "zmin", "components": (2,)},
             {"node_set": "zmax", "components": (2,)}]
    load = [{"face_set": "loaded", "total_force": [0.0, 100.0, 0.0]}]
    result = solve_hex(mesh, {"E": 70e6, "nu": 0.4999}, fixed, load, formulation="bbar")
    tip_mm = probe(mesh, result["displacement"][:, 1], (0.048, 0.060)) * 1000.0
    if not 27.3 <= tip_mm <= 28.0:
        print(f"\nFAILED: tip displacement {tip_mm:.2f} mm is outside the "
              "CoFEA reference band 27.3 to 28.0 mm")
        return 1
    print(f"ok ({tip_mm:.2f} mm)")

    print("3/3 unit demo, replayed from recordings...", end=" ", flush=True)
    offline = {k: v for k, v in os.environ.items() if k != "GEMINI_API_KEY"}
    run = subprocess.run([sys.executable, "unit_demo.py"], cwd=REPO, env=offline,
                         capture_output=True, text=True)
    if run.returncode != 0 or "WRITE IT DOWN" not in run.stdout:
        print(f"\nFAILED:\n{run.stderr[-1500:]}")
        return 1
    print("ok")

    print("\nALL CHECKS PASSED")
    if os.environ.get("GEMINI_API_KEY"):
        print("GEMINI_API_KEY is set: playground.py and live runs will work.")
    else:
        print("No GEMINI_API_KEY yet. Free key: https://aistudio.google.com/apikey")
    return 0


if __name__ == "__main__":
    sys.exit(main())
