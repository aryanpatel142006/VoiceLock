"""T-133: the saved mic is opened exactly; when it is gone the page says so loudly (banner + /health).

Chromium's fake media devices are the real list here (no ``enumerateDevices`` stub), so "present" means a
fake device id the browser knows and "missing" an id it does not. No real audio device is opened.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from tests.test_device_picker import _serve

GUM = """
window.__gum = []; window.__streams = [];
const realGum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia = async (c) => {
  window.__gum.push(JSON.parse(JSON.stringify(c)));
  const s = await realGum(c); if (s.getAudioTracks().length) window.__streams.push(s); return s;
};
"""
BANNER = "[data-dev-banner]"


def _health(base: str) -> dict:
    with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
        return json.load(r)


def _start(page, base: str, path: str, mic_id: str, label: str) -> None:
    page.goto(base + path)
    page.evaluate(f"localStorage.setItem('hearing.mic', {json.dumps(mic_id)});"
                  f"localStorage.setItem('hearing.mic.label', {json.dumps(label)})")
    page.reload()
    page.locator("#start").click()
    page.wait_for_function("window.__streams.length > 0")


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
@pytest.mark.parametrize("path", ["/judge", "/simple"])
def test_saved_mic_exact_or_loud_banner(path):
    sync_api = pytest.importorskip("playwright.sync_api")
    base, proc = _serve()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(args=[
                "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required"])
            ctx = browser.new_context(permissions=["microphone", "camera"])
            ctx.set_default_timeout(15000)
            page = ctx.new_page()
            page.add_init_script(GUM)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            # missing: the exact id is asked for, the default opens, the banner names both, /health says so
            _start(page, base, path, "gone-rode-id", "RØDE VideoMic GO II")
            assert page.evaluate("window.__gum.find(c => c.audio).audio.deviceId") == {"exact": "gone-rode-id"}
            page.wait_for_selector(BANNER, state="visible")
            text = page.locator(BANNER).inner_text()
            assert "Demo mic not found (RØDE VideoMic GO II): using" in text, text
            used = page.evaluate("window.__streams.at(-1).getAudioTracks()[0].label")
            assert used in text
            for _ in range(50):
                devs = _health(base)["client_devices"]
                if devs and devs.get("missing"):
                    break
                page.wait_for_timeout(100)
            assert devs["missing"] == ["mic"] and devs["input"] == used, devs

            # present: a real (fake) non-default device is opened exactly and no banner shows
            page.locator('[data-dev-banner] button').click()  # Retry: same missing device -> banner again
            page.wait_for_selector(BANNER, state="visible")
            # a present device: Chromium's "default" input (its fake non-default audio ids open exactly from an
            # already-capturing page but fail after a reload, so the stable id is the one the test can rely on)
            dev_id = "default"
            page.evaluate("window.__gum = []; window.__streams = []")
            _start(page, base, path, dev_id, "Fake Audio Input")
            assert page.evaluate("window.__gum.find(c => c.audio).audio.deviceId") == {"exact": dev_id}
            # one audio request, exact, no fallback request (Chromium's fake audio reports every track as
            # the default device in getSettings(), so the request is what the test can pin)
            assert page.evaluate("window.__gum.filter(c => c.audio).length") == 1
            label = page.evaluate("window.__streams.at(-1).getAudioTracks()[0].label")
            page.wait_for_timeout(500)
            assert page.locator(BANNER).is_hidden()
            for _ in range(50):
                devs = _health(base)["client_devices"]
                if devs and devs.get("input") == label:
                    break
                page.wait_for_timeout(100)
            assert devs["input"] == label and devs["missing"] == [], devs
            assert not errors, errors
            browser.close()
    finally:
        proc.kill()
        proc.wait(timeout=10)
