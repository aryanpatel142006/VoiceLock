"""T-150: /judge Focus / Quiet buttons and Shift+F / Shift+Q send the right control messages, Quiet shows the
captions with the speaker label, and a telemetry frame whose caption feed fell back to the mixture shows the
notice. The page's WebSocket is mocked (no session); Chromium with fake media, no real device. Screenshots at
1280 and 390 px go to $HEARING_JUDGE_SHOTS (default: the test's tmp dir)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_device_picker import _serve

MOCK_WS = """
window.mockSent = [];
window.WebSocket = class {
  static OPEN = 1;
  constructor() { this.readyState = 1; window.mockWs = this; setTimeout(() => this.onopen?.(), 10); }
  send(data) { if (typeof data === "string") window.mockSent.push(JSON.parse(data)); }
  close() {}
};
"""
LAST = "window.mockSent.filter(m => m.type === 'control').at(-1)"


def _push(page, msg: dict) -> None:
    page.evaluate("m => window.mockWs.onmessage({data: m})", json.dumps(msg))


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
def test_judge_focus_quiet(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    shots = Path(os.environ.get("HEARING_JUDGE_SHOTS") or tmp_path)
    shots.mkdir(parents=True, exist_ok=True)
    base, proc = _serve()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
            page = browser.new_context(viewport={"width": 1280, "height": 800}).new_page()
            page.set_default_timeout(15000)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.add_init_script(MOCK_WS)
            page.goto(base + "/judge")
            page.wait_for_function("window.hearingJudge.connected")
            first = page.evaluate(LAST)  # the judge profile, in Focus
            assert first["profile"] == "judge" and first["mode"] == "focus_hq"
            assert page.locator("#captions-panel").is_hidden()
            assert page.locator("#modes button[data-mode=focus_hq]").get_attribute("aria-pressed") == "true"
            assert "hear who matters" in page.locator("#modes").inner_text()
            assert "stay connected without listening" in page.locator("#modes").inner_text()

            page.click("#modes button[data-mode=quiet]")
            assert page.evaluate(LAST) == {"type": "control", "mode": "quiet"}
            assert page.locator("#modes button[data-mode=quiet]").get_attribute("aria-pressed") == "true"
            page.wait_for_function("!document.getElementById('captions-panel').hidden")
            for el in ("#ab", "#pill", "#modes"):  # Quiet hides nothing else
                assert page.locator(el).is_visible(), el

            page.click("#modes button[data-mode=focus_hq]")
            assert page.evaluate(LAST) == {"type": "control", "mode": "focus_hq"}
            page.wait_for_function("document.getElementById('captions-panel').hidden")

            # keys are guarded like C/R/N (T-135): alone nothing, with Shift they switch
            n = page.evaluate("window.mockSent.length")
            page.keyboard.press("q")
            assert page.evaluate("window.mockSent.length") == n
            page.keyboard.press("Shift+Q")
            assert page.evaluate(LAST) == {"type": "control", "mode": "quiet"}
            page.keyboard.press("Shift+F")
            assert page.evaluate(LAST) == {"type": "control", "mode": "focus_hq"}
            page.keyboard.press("Shift+Q")

            tel = {"type": "telemetry", "mode": "quiet", "effective_mode": "quiet", "synth_feed": "focus",
                   "synth_feed_request": "focus", "quiet": {"feed": "focus"}, "enrollment": {"state": "ready", "source": "profile", "profile": "Aryan"},
                   "acquisition": {"state": "locked", "detail": "", "since_s": 1.0}}
            _push(page, tel)
            _push(page, {"type": "caption", "text": "see you at noon", "final": True, "t0": 1.0, "t1": 2.0})
            page.wait_for_function("document.getElementById('caption-speaker').textContent === 'Aryan'")
            assert page.locator("#caption-warning").is_hidden()
            assert "see you at noon" in page.locator("#captions").inner_text()
            for i in range(6):  # trimming old lines never removes the speaker label
                _push(page, {"type": "caption", "text": f"line {i}", "final": True, "t0": 1.0, "t1": 2.0})
            assert page.locator("#caption-speaker").is_visible()

            _push(page, {**tel, "synth_feed": "mixture"})  # the ladder failed: the ASR hears the room
            page.wait_for_function("!document.getElementById('caption-warning').hidden")
            assert "whole room" in page.locator("#caption-warning").inner_text()

            page.evaluate("document.getElementById('start').setAttribute('aria-pressed', 'true')")  # hide the overlay
            page.screenshot(path=str(shots / "judge-quiet-1280.png"))
            page.set_viewport_size({"width": 390, "height": 844})
            assert page.evaluate("document.documentElement.scrollWidth") <= 390
            page.screenshot(path=str(shots / "judge-quiet-390.png"))
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
