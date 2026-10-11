"""T-175: the stage card's keys match /judge's key handler, its three lines and fallbacks are present, and
the rendered PDF is one page. Static checks only (the key behaviour itself: tests/test_judge_keys.py)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = (ROOT / "docs" / "STAGE_CARD.md").read_text(encoding="utf-8")
JUDGE = (ROOT / "web" / "console" / "judge.js").read_text(encoding="utf-8")
MODES = (ROOT / "web" / "console" / "judge-modes.js").read_text(encoding="utf-8")


def _keys_table() -> list[str]:
    section = CARD.split("## Keys", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^\| \*\*([^*]+)\*\*", section, re.MULTILINE)


def test_every_card_key_is_handled_by_judge_js():
    guarded = re.search(r'if \("([a-z]+)"\.includes\(key\)', JUDGE).group(1)
    handled = set(re.findall(r'key === "(.)"', JUDGE)) | set(re.findall(r'key: "(.)"', MODES))
    # T-177: Release is Shift+0 via isReleaseKey, guarded inside its own branch (Shift or drawer open)
    if "isReleaseKey(key, e.code)" in JUDGE and "(e.shiftKey || $(\"drawer\").open) && !e.repeat) release()" in JUDGE:
        handled.add("0")
        guarded += "0"
    keys = _keys_table()
    assert keys[:1] == ["Start"] and 'id="start"' in (ROOT / "web/console/judge.html").read_text()
    for k in keys[1:]:
        letter = {"Space": " "}.get(k, k.removeprefix("Shift+").lower())
        assert letter in handled, k
        # a guarded key must be printed with Shift (it does nothing alone with the drawer closed), others without
        assert k.startswith("Shift+") == (letter in guarded), k


def test_three_lines_and_fallback_order():
    for line in (
        '"Turn down the world without disconnecting from it."',
        '"noise suppression, not person selection"',
        (
            "So the person you want to hear wears a clip mic, and the app does the part no product does: "
            'pick who, mute the rest, caption only them."'
        ),
    ):
        assert line in CARD
    boxes = re.findall(r"^☐ \*\*(.)\. ([^*]+)\*\*", CARD, re.MULTILINE)
    assert boxes == [("a", "Quiet captions."), ("b", "Natural noise suppression."), ("c", "Labelled replay.")]
    assert len(re.findall(r"^\d\. ", CARD.split("## The 8 steps", 1)[1].split("\n## ", 1)[0], re.MULTILINE)) == 8


def test_pdf_is_one_page():
    pdf = (ROOT / "docs" / "STAGE_CARD.pdf").read_bytes()
    assert len(re.findall(rb"/Type\s*/Page[^s]", pdf)) == 1
