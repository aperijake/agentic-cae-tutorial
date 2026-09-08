"""Checks on the code half of look.py: the snapshot and the shape of the image message.

The model call itself is not tested. What is tested is that the snapshot is
a real PNG of a real solve, and that sending it changes exactly one thing
about the message: the user content becomes a text part plus an image part.
"""

import base64

import look


def test_snapshot_is_a_png_of_a_locked_solve():
    png, tip_mm = look.snapshot("full", averaged=False)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert tip_mm < 10.0            # fully integrated hex8 locks at nu = 0.4999


def test_bbar_snapshot_is_not_locked():
    _, tip_mm = look.snapshot("bbar", averaged=False)
    assert tip_mm > 20.0


def test_image_message_is_text_part_plus_image_part(monkeypatch):
    sent = {}

    def fake_chat(instructions, user_content):
        sent["instructions"], sent["content"] = instructions, user_content
        return "NO."

    monkeypatch.setattr(look, "chat", fake_chat)
    png = b"\x89PNG\r\n\x1a\nfake"
    look.ask_with_image("reviewer", "question", png)
    text_part, image_part = sent["content"]
    assert text_part == {"type": "text", "text": "question"}
    assert image_part["type"] == "image_url"
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == png


def test_text_message_is_a_plain_string(monkeypatch):
    sent = {}
    monkeypatch.setattr(look, "chat", lambda i, c: sent.setdefault("content", c) and "NO.")
    look.ask("reviewer", "question")
    assert sent["content"] == "question"


def test_flag_scan_keys_on_symptom_words_not_verdicts():
    assert look.flagged("YES. A clear checkerboard pattern across adjacent elements.")
    assert not look.flagged("YES, the mesh is coarse and I would not accept this result.")


def test_the_choice_is_limited_to_formulations_the_code_has(monkeypatch):
    answers = iter(["B-bar", "bbar", "Full integration locks here, so use bbar.", "full", "tet10"])
    monkeypatch.setattr(look, "ask_with_image", lambda *a: next(answers))
    png = b"\x89PNG\r\n\x1a\nfake"
    assert [look.choose_formulation(png)[0] for _ in range(5)] == ["bbar", "bbar", "bbar", "full", None]
