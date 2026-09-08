"""Give the agent eyes: solve, snapshot the result, send the picture, ask.

One model call, like the playground's agents. The only new thing is the
image in the message. Read the two ask functions side by side: the user
message goes from a string to a list with a text part and an image part.
Nothing else changes.

Run:

    python look.py               # solve, inspect the snapshot, then ask with the picture
    python look.py --runs 5      # ask with the picture five times; count the verdicts

Things to try, one line each, marked EDIT ME:

    1. FORMULATION "full" -> "bbar". The picture changes. Does the verdict?
    2. AVERAGED False -> True. The plot you would normally make. Does it still notice?
    3. Change QUESTION. Run five times. A prompt is a prompt.

After the verdict, the model is offered the formulations the code has and
asked to pick one for the next run. Code reruns with its choice and asks
again. The choice is limited to that list: it cannot name an element we do
not have. Then one last question, text only: is that element inf-sup
stable? Run it more than once before you trust the answer.

Needs the package installed (pip install -e .) and GEMINI_API_KEY.
"""

import base64
import io
import os
import sys

import matplotlib

# Draw to an image file rather than opening an interactive plotting window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from minifea import cooks_membrane, probe, solve_hex
from minifea.plotting import plot_pressure

# ------------------------------------------------------------------ EDIT ME --

FORMULATION = "full"   # fully integrated hex8 locks at nu = 0.4999; "bbar" does not
AVERAGED = False       # True: element-averaged pressure, the plot you would normally make
QUESTION = ("This is a pressure field from a finite element analysis. Describe it in two "
            "sentences. If you see a numerical artifact rather than physics, name it.")
REVIEWER = "You are reviewing a finite element result for an engineer."

# ----------------------------------------------------------------------------


# ---- 1. The text call you already know --------------------------------------

def ask(instructions, text):
    """One model call. The user message is a string."""
    return chat(instructions, text)


# ---- 2. Take the snapshot ---------------------------------------------------

def snapshot(formulation, averaged):
    """Solve Cook's membrane, plot the pressure field, return it as PNG bytes.

    No title, no labels, nothing in the frame that gives the diagnosis away.
    The PNG is what a screenshot of the viewer would be.
    """
    # Build an 8-by-8 in-plane mesh. The solver uses meters, Pascals, and Newtons.
    mesh = cooks_membrane(8)
    material = {"E": 70e6, "nu": 0.4999}
    # Components 0, 1, 2 mean x, y, z displacement. Clamp the left edge;
    # prevent out-of-plane displacement on the front and back faces.
    fixed = [{"node_set": "clamped", "components": (0, 1, 2)},
             {"node_set": "zmin", "components": (2,)},
             {"node_set": "zmax", "components": (2,)}]
    load = [{"face_set": "loaded", "total_force": [0.0, 100.0, 0.0]}]
    result = solve_hex(mesh, material, fixed, load, formulation=formulation)
    # [:, 1] selects the y displacement at every node (Python counts from 0).
    # Read it at the top right corner, then convert meters to millimeters.
    y_displacement = result["displacement"][:, 1]
    tip_mm = probe(mesh, y_displacement, (0.048, 0.060)) * 1000.0

    # fig is the whole image; ax is the plotting area inside it.
    fig, ax = plt.subplots(figsize=(5.0, 5.2))
    # Choose one pressure per element, or the unaveraged integration-point values.
    plot_pressure(mesh, result, ax, sampling="element" if averaged else "gauss",
                  material=material)
    fig.tight_layout()
    # BytesIO is a file held in memory. Save the plot into it as PNG data.
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # Return two things: the image bytes to send, and the tip value to print.
    return png.getvalue(), tip_mm


# ---- 3. Send the snapshot: the only change is the user message --------------

def ask_with_image(instructions, text, png_bytes):
    """One model call. The user message is a list: a text part and an image part."""
    # Base64 represents the image bytes as text that fits inside an API message.
    # It is an encoding, not encryption. decode() turns that encoded data into
    # a Python text string; it does not decode the picture.
    encoded_image = base64.b64encode(png_bytes).decode("ascii")

    # This data URL contains the image itself, rather than a website address.
    image = "data:image/png;base64," + encoded_image

    # The user message has two parts: our question and the image.
    # "image_url" is the API field name even when the image is embedded here.
    content = [{"type": "text", "text": text},
               {"type": "image_url", "image_url": {"url": image}}]
    return chat(instructions, content)


# ---- 4. Let the model choose the next run -----------------------------------

FORMULATIONS = {"full": "standard hex8, full integration",
                "bbar": "hex8 with mean-dilatation B-bar"}
CHOICE = ("Formulations available for the next run: "
          + "; ".join(f"{name} ({what})" for name, what in FORMULATIONS.items())
          + ". Given what you saw in this picture, which one should the next run use? "
          "Answer with one word.")


def choose_formulation(png_bytes):
    """Ask the model to pick from the formulations the code actually offers.

    Returns (name or None, the full answer). Only names in FORMULATIONS count,
    so the model cannot send us to an element that does not exist here.
    """
    answer = ask_with_image(REVIEWER, CHOICE, png_bytes)
    # "B-bar", "bbar" and "BBAR" should all count as bbar. Check it first, since
    # an explanation may mention "full integration" while choosing bbar.
    text = answer.lower().replace("-", "")
    picked = "bbar" if "bbar" in text else ("full" if "full" in text else None)
    return picked, answer


# ---- 5. One more question, no picture --------------------------------------

THEORY = ("The rerun used an 8-node hexahedron with mean-dilatation B-bar for a nearly "
          "incompressible material. Is that element inf-sup stable? Answer yes or no "
          "first, then explain in three sentences.")


# ---- API connection and running the exercise ------------------------------

MODEL = "gemini-2.5-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
KEY_ENV = "GEMINI_API_KEY"
_client = None

FLAGS = ("checkerboard", "oscillat", "spurious", "hourglass", "locking")   # symptom words


def chat(instructions, user_content):
    # Reuse the connection object between calls. This does not preserve chat history.
    global _client
    if _client is None:
        from openai import OpenAI
        # This client can talk to Gemini through its OpenAI-compatible endpoint.
        # Read the key from the environment variable set in your terminal.
        _client = OpenAI(api_key=os.environ[KEY_ENV], base_url=BASE_URL)
    # Every call sends a fresh two-message conversation:
    # system = the reviewer's role; user = the question (and optionally the image).
    # Temperature 0 reduces sampling variation; it does not guarantee correctness.
    reply = _client.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "system", "content": instructions},
                  {"role": "user", "content": user_content}])
    # The API returns a list of responses. [0] selects the first one.
    first_response = reply.choices[0]

    # Extract its text and remove whitespace at the beginning and end.
    answer = first_response.message.content
    return answer.strip()


def flagged(answer):
    """Did the answer name a spurious pattern? Crude keyword scan, for counting only."""
    # Compare in lowercase so capitalization does not matter.
    # This only counts word mentions: "no checkerboard" also matches.
    # Read the answer yourself to decide whether its diagnosis is supported.
    text = answer.lower()
    return any(flag in text for flag in FLAGS)


def main():
    # sys.argv contains the command-line words. Read the number after --runs,
    # or make one image call if that option was not supplied.
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1
    if KEY_ENV not in os.environ:
        sys.exit(f"Set {KEY_ENV} first. Free key: https://aistudio.google.com/apikey")

    # Solve and make the image once. Repeated model calls will see this same image.
    png, tip_mm = snapshot(FORMULATION, AVERAGED)
    # "wb" means write binary data. This is the exact PNG sent to the model.
    with open("snapshot.png", "wb") as f:
        f.write(png)
    print(f"solved: hex8 {FORMULATION}, nu = 0.4999, tip displacement {tip_mm:.1f} mm")
    print(f"snapshot: {'element-averaged' if AVERAGED else 'unaveraged'} pressure -> snapshot.png")

    print("\nOpen snapshot.png—the exact image we will send to the model.")
    input("What do you notice? Press Enter when ready to ask the model. ")

    # Optional text-only comparison, kept here for reference but not executed.
    # log = (f"The linear static solve completed with no errors or warnings. "
    #        f"Tip displacement {tip_mm:.1f} mm.")
    # print("\n== without the picture ==\n  given: " + log)
    # print("  " + ask(REVIEWER, log + " " + QUESTION).replace("\n", "\n  "))

    print("\n== with the picture ==")
    count = 0
    # range(runs) counts from 0 to runs - 1; display i + 1 for human numbering.
    for i in range(runs):
        answer = ask_with_image(REVIEWER, QUESTION, png)
        # Python counts True as 1 and False as 0. Add one for a keyword match.
        count += flagged(answer)
        print(f"\n  run {i + 1}:\n  " + answer.replace("\n", "\n  "))
    print(f"\nnamed a spurious pattern in {count} of {runs} (keyword scan; read the answers)")

    # Step 4: offer the formulations we have, run the one it picks, ask again.
    print("\n== choose ==")
    print("  given: " + CHOICE)
    picked, answer = choose_formulation(png)
    print("  model: " + answer.replace("\n", "\n  "))
    if picked is None:
        print("  no formulation from the list in that answer; nothing to run")
    elif picked == FORMULATION:
        print(f"  same as the current run ({FORMULATION}); nothing to change")
    else:
        print(f"\n== rerun with {picked} ==")
        png_after, tip_after = snapshot(picked, AVERAGED)
        with open("snapshot_after.png", "wb") as f:
            f.write(png_after)
        print(f"  solved: hex8 {picked}, nu = 0.4999, tip displacement {tip_after:.1f} mm "
              f"(reference 27.3 to 28.0) -> snapshot_after.png")
        answer_after = ask_with_image(REVIEWER, QUESTION, png_after)
        print("  " + answer_after.replace("\n", "\n  "))
        print(f"  named a spurious pattern: {'yes' if flagged(answer_after) else 'no'}")

    # Step 5: a theory question, no picture. The same text call as step 1.
    print("\n== one more question ==")
    print("  given: " + THEORY)
    print("  model: " + ask(REVIEWER, THEORY).replace("\n", "\n  "))


# Start the exercise when this file is run, but not when imported by tests.
if __name__ == "__main__":
    main()
