"""T-158: /judge never looks live while the mic is not streaming. Chromium with fake media; no real device."""

from __future__ import annotations

import json

import pytest

from tests.test_device_picker import _serve

PILL = "document.getElementById('pill-text').textContent"
BANNER = "document.getElementById('mic-banner')"


def _page(p, base):
    browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
    page = browser.new_context(permissions=["microphone", "camera"]).new_page()
    page.set_default_timeout(15000)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(base + "/judge")
    page.wait_for_function("window.hearingJudge.connected && window.hearingJudge.telemetry")
    return browser, page, errors


@pytest.mark.slow  # starts a server and Chromium (and waits out the 8 s Start timeout); no weights, no network
def test_judge_pill_and_banner_follow_the_mic():
    sync_api = pytest.importorskip("playwright.sync_api")
    base, proc = _serve()
    try:
        with sync_api.sync_playwright() as p:
            browser, page, errors = _page(p, base)
            # (1) connected, telemetry flowing, Start not pressed: Press Start, the Start button stays
            page.wait_for_timeout(1200)
            assert page.evaluate(PILL) == "Press Start"
            assert page.locator("#start").is_visible()

            # (2a) getUserMedia rejected: red banner outside the drawer, Start enabled again
            page.evaluate("""() => { navigator.mediaDevices.getUserMedia = () =>
                Promise.reject(Object.assign(new Error('Permission denied'), {name: 'NotAllowedError'})); }""")
            page.click("#start")
            page.wait_for_function(f"!{BANNER}.hidden")
            assert "Microphone blocked" in page.evaluate(f"{BANNER}.textContent")
            assert page.evaluate(f"{BANNER}.closest('details') === null")
            assert page.locator("#start").is_enabled()
            assert page.evaluate(PILL) == "Press Start"

            # (2b) getUserMedia never settles: after 8 s the banner names it and Start is back
            page.evaluate("() => { navigator.mediaDevices.getUserMedia = () => new Promise(() => {}); }")
            page.click("#start")
            assert page.evaluate(f"{BANNER}.hidden")  # a new Start clears the old line
            assert page.locator("#start").is_disabled()
            page.wait_for_function(f"!{BANNER}.hidden", timeout=12000)
            assert "did not start within 8 s" in page.evaluate(f"{BANNER}.textContent")
            assert page.locator("#start").is_enabled()
            assert not errors, errors
            browser.close()

            # (3, 4) a working Start: the hello names the (non-Bluetooth) mic, then a stall reads Mic not streaming
            browser, page, errors = _page(p, base)
            page.evaluate("""(() => { const ws = window.hearingJudge.ws, send = ws.send.bind(ws);
                window.texts = []; ws.send = (d) => { if (typeof d === 'string') window.texts.push(d); send(d); }; })()""")
            page.click("#start")
            page.wait_for_function("window.hearingJudge.audio && window.hearingJudge.seq > 20")
            assert page.evaluate(f"{BANNER}.hidden")
            page.wait_for_function(f"{PILL} !== 'Press Start'")
            hellos = [m for m in map(json.loads, page.evaluate("window.texts")) if m.get("type") == "hello"]
            assert hellos and hellos[-1].get("input_device"), f"hello after Start carries no input_device: {hellos}"
            assert not page.evaluate("document.querySelector('[data-dev-banner]')?.textContent || ''").startswith("Mic is")
            page.evaluate("window.hearingJudge.audio.ctx.suspend()")  # the worklet stops: no frame goes out
            page.wait_for_function(f"{PILL}.startsWith('Mic not streaming')", timeout=6000)
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
