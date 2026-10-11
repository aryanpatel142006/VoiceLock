"""T-177: /judge "Choose saved person" buttons send what keys 1/2 send and the target line follows, the Raw /
Isolated buttons toggle ab, Release (button and Shift+0) sends enroll clear and the pill reads Searching, and the
"Keys off: click the page" banner follows window blur/focus and text fields. Mocked WebSocket, Chromium with fake
media; no real device, no session."""

from __future__ import annotations

import json

import pytest

from tests.test_device_picker import _serve
from tests.test_judge_modes import LAST, MOCK_WS


def _push(page, msg: dict) -> None:
    page.evaluate("m => window.mockWs.onmessage({data: m})", json.dumps(msg))


def _tel(profile: str | None, source: str | None = "profile") -> dict:
    return {"type": "telemetry", "mode": "focus_hq", "effective_mode": "focus_hq",
            "acquisition": {"state": "locked", "detail": f"known voice: {profile}", "since_s": 1.0},
            "enrollment": {"state": "ready", "source": source, "profile": profile, "profiles": ["aryan", "joseph"]}}


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
def test_judge_people_release_and_keys_off_banner():
    sync_api = pytest.importorskip("playwright.sync_api")
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
            page.evaluate("document.getElementById('start').setAttribute('aria-pressed', 'true')")  # hide the overlay
            # T-158's "Press Start" outranks every pill until the mic streams; a replay counts as streaming
            page.evaluate("window.hearingJudge.replaying = true")
            assert page.locator("#people h2").inner_text().lower() == "choose saved person"
            assert page.locator("#release").inner_text() == "Release"
            assert page.locator("#calibrate").count() == 0  # live calibration stays Shift+C, off the stage
            assert page.locator("#people button[data-person='1']").is_disabled()  # no profiles yet

            _push(page, _tel("aryan"))
            joseph = page.locator("#people button[data-person='1']")
            page.wait_for_function("!document.querySelector(\"#people button[data-person='1']\").disabled")
            assert "Joseph" in joseph.inner_text() and "Aryan" in page.locator("#people").inner_text()
            assert page.locator("#target-line").inner_text() == "Target: Aryan"

            joseph.click()
            assert page.evaluate(LAST) == {"type": "control", "enroll": {"action": "profile", "name": "joseph"}}
            page.keyboard.press("2")  # the key sends the same message
            assert page.evaluate(LAST) == {"type": "control", "enroll": {"action": "profile", "name": "joseph"}}
            _push(page, _tel("joseph"))
            page.wait_for_function("document.getElementById('target-line').textContent === 'Target: Joseph'")
            assert joseph.get_attribute("aria-pressed") == "true"

            page.click("#ab button[data-ab=raw]")
            assert page.evaluate(LAST) == {"type": "control", "ab": "raw"}
            assert page.evaluate("window.hearingJudge.ab") == "raw"
            page.click("#ab button[data-ab=processed]")
            assert page.evaluate(LAST) == {"type": "control", "ab": "processed"}

            page.click("#release")
            assert page.evaluate(LAST) == {"type": "control", "enroll": {"action": "clear"}}
            page.wait_for_function("document.getElementById('pill-text').textContent === 'Searching'")
            assert page.locator("#target-line").inner_text().startswith("Target: none")
            assert page.locator("#release").get_attribute("aria-pressed") == "true"
            page.keyboard.press("1")  # a saved person ends the release
            assert page.evaluate(LAST) == {"type": "control", "enroll": {"action": "profile", "name": "aryan"}}
            page.wait_for_function("document.getElementById('pill-text').textContent !== 'Searching'")
            n = page.evaluate("window.mockSent.length")
            page.keyboard.press("0")  # guarded like C/R/N: alone nothing
            assert page.evaluate("window.mockSent.length") == n
            page.keyboard.press("Shift+0")
            assert page.evaluate(LAST) == {"type": "control", "enroll": {"action": "clear"}}
            page.wait_for_function("document.getElementById('pill-text').textContent === 'Searching'")
            page.keyboard.press("Shift+C")  # the backup calibration also ends the release
            assert page.evaluate(LAST) == {"type": "control", "enroll": {"action": "start", "seconds": 3}}
            page.wait_for_function("document.getElementById('pill-text').textContent !== 'Searching'")
            assert page.locator("#release").get_attribute("aria-pressed") == "false"
            assert not page.locator("#target-line").inner_text().startswith("Target: none")

            banner = page.locator("#keys-off")
            assert banner.is_hidden()
            page.evaluate("window.dispatchEvent(new Event('blur'))")  # another window or tab took the keys
            page.wait_for_function("!document.getElementById('keys-off').hidden")
            assert banner.inner_text() == "Keys off: click the page"
            page.evaluate("window.dispatchEvent(new Event('focus'))")
            page.wait_for_function("document.getElementById('keys-off').hidden")
            page.keyboard.press("d")  # drawer open: the Name Call text field is reachable
            page.wait_for_function("!document.getElementById('namecall').hidden")
            page.click("#namecall-name")
            page.wait_for_function("!document.getElementById('keys-off').hidden")
            page.click("#pill")  # click the page: the field loses focus, keys are back
            page.wait_for_function("document.getElementById('keys-off').hidden")
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
