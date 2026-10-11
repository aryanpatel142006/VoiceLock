"""T-135: /judge keys are guarded (C/R/N need Shift or the open drawer), the Name Call form hides until
N arms it, and the pill goes through the page's hysteresis. Chromium with fake media; no real device."""

from __future__ import annotations

import pytest

from tests.test_device_picker import _serve

SENT = "window.hearingJudge.sent.map(m => JSON.stringify(m)).join('|')"


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
def test_judge_keys_need_shift_or_the_drawer():
    sync_api = pytest.importorskip("playwright.sync_api")
    base, proc = _serve()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
            page = browser.new_context(permissions=["microphone", "camera"]).new_page()
            page.set_default_timeout(15000)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base + "/judge")
            page.wait_for_function("window.hearingJudge.connected")
            assert page.locator("#namecall").is_hidden()
            page.evaluate("window.hearingJudge.sent.length = 0")

            for key in ("c", "r", "n"):  # alone: nothing fires
                page.keyboard.press(key)
            assert '"enroll"' not in page.evaluate(SENT)
            assert page.evaluate("window.hearingJudge.nameCall") is False
            assert page.locator("#namecall").is_hidden()

            page.keyboard.press("Shift+C")
            assert '"enroll"' in page.evaluate(SENT)
            page.keyboard.press("Shift+N")
            assert page.evaluate("window.hearingJudge.nameCall") is True
            page.wait_for_function("!document.getElementById('namecall').hidden")
            page.keyboard.press("Shift+N")
            page.wait_for_function("document.getElementById('namecall').hidden")

            page.keyboard.press("d")  # drawer open: C alone works, and the form is reachable there
            page.evaluate("window.hearingJudge.sent.length = 0")
            page.keyboard.press("c")
            assert '"enroll"' in page.evaluate(SENT)
            page.wait_for_function("!document.getElementById('namecall').hidden")

            # the pill goes through PillHysteresis (shown changes are counted for judge_check)
            assert page.evaluate("typeof window.hearingJudge.pill.changes") == "number"
            assert page.locator("#pill-text").inner_text() != ""
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
