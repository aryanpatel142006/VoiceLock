"""T-287: the iPhone mic on /judge, end to end in Chromium with a stubbed mediaDevices (no real device).

Under the team preset with ``HEARING_INPUT_DEVICE`` naming the iPhone, Start opens the iPhone by exact deviceId
even though another mic was saved; when its track ends mid-session the page reopens the MacBook mic in the same
session (same socket, a hello naming it), shows "iPhone mic lost: using MacBook mic" and the drawer select shows
the MacBook mic; once the iPhone is listed again the banner's Retry switches back and the banner clears.
"""

from __future__ import annotations

import json
import urllib.request

import pytest

from tests.test_device_picker import _serve

IPHONE = "Aryan patel's iPhone Microphone"
MACBOOK = "MacBook Pro Microphone"
BANNER = "[data-dev-banner]"

# getUserMedia returns a real track (an oscillator through MediaStreamDestination) carrying the fake label/id
FAKE_MEDIA = """(() => {
  const all = [{ deviceId: "iphone-id", label: IPHONE_LABEL }, { deviceId: "macbook-id", label: MACBOOK_LABEL }, { deviceId: "rode-id", label: "RØDE VideoMic GO II" }];
  const fake = window.__fake = { devs: [...all], all, opened: [], streams: [] };
  const md = navigator.mediaDevices;
  md.enumerateDevices = async () => [
    { kind: "audioinput", deviceId: "default", label: "Default - MacBook Pro Microphone", groupId: "" },
    ...fake.devs.map((d) => ({ kind: "audioinput", groupId: "", ...d })),
    { kind: "audiooutput", deviceId: "default", label: "Default - AirPods Pro", groupId: "" }];
  let ac = null;
  md.getUserMedia = async (c) => {
    if (c.video) throw new DOMException("no camera here", "NotFoundError");
    const want = c.audio?.deviceId?.exact;
    const dev = fake.devs.find((d) => d.deviceId === (want || "macbook-id"));
    if (!dev) throw new DOMException("device gone", "OverconstrainedError");
    ac = ac || new AudioContext();
    const osc = ac.createOscillator(), dst = ac.createMediaStreamDestination();
    osc.connect(dst); osc.start();
    const track = dst.stream.getAudioTracks()[0];
    Object.defineProperty(track, "label", { value: dev.label });
    track.getSettings = () => ({ deviceId: dev.deviceId });
    fake.opened.push(want || "default"); fake.streams.push(dst.stream);
    return dst.stream;
  };
})();""".replace("IPHONE_LABEL", json.dumps(IPHONE)).replace("MACBOOK_LABEL", json.dumps(MACBOOK))

SPY = """(() => {
  const ws = window.hearingJudge.ws, send = ws.send.bind(ws);
  window.hellos = [];
  ws.send = (d) => { if (typeof d === 'string' && d.includes('"hello"')) window.hellos.push(JSON.parse(d)); send(d); };
})()"""


def _health_input(base: str, page, want: str) -> dict:
    devs = {}
    for _ in range(50):
        with urllib.request.urlopen(f"{base}/health", timeout=3) as r:
            devs = json.load(r)["client_devices"] or {}
        if devs.get("input") == want:
            break
        page.wait_for_timeout(100)
    return devs


def _banner_lines(page) -> list[str]:
    if page.locator(BANNER).is_hidden():
        return []
    return page.locator(f"{BANNER} span").inner_text().split(" · ")


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
def test_iphone_mic_picked_dropout_falls_back_and_retry_returns(monkeypatch):
    sync_api = pytest.importorskip("playwright.sync_api")
    monkeypatch.setenv("HEARING_DEMO_PRESET", "team")
    monkeypatch.setenv("HEARING_INPUT_DEVICE", IPHONE)
    base, proc = _serve()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required"])
            ctx = browser.new_context(permissions=["microphone"])
            ctx.set_default_timeout(15000)
            page = ctx.new_page()
            page.add_init_script(FAKE_MEDIA)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base + "/judge")
            page.evaluate("localStorage.setItem('hearing.mic', 'rode-id'); localStorage.setItem('hearing.mic.label', 'RØDE VideoMic GO II')")
            page.reload()
            page.wait_for_function("window.hearingJudge.connected && window.hearingJudge.telemetry?.preset?.pins?.input_device")
            assert page.evaluate("window.hearingJudge.telemetry.preset.pins.input_device") == IPHONE
            page.evaluate(SPY)

            # (a) Start opens the iPhone (exact id) over the saved RØDE and saves it
            page.locator("#start").click()
            page.wait_for_function("window.__fake.streams.length > 0 && window.hearingJudge.audio")
            assert page.evaluate("window.__fake.opened") == ["iphone-id"]
            assert page.evaluate("localStorage.getItem('hearing.mic')") == "iphone-id"
            assert page.evaluate("localStorage.getItem('hearing.mic.label')") == IPHONE
            assert _health_input(base, page, IPHONE).get("input") == IPHONE
            assert not any("mic" in line.lower() for line in _banner_lines(page)), _banner_lines(page)

            # (c) the iPhone is unplugged mid-session: the MacBook mic within 1 s, same socket, banner, select, hello
            page.evaluate("window.__ws0 = window.hearingJudge.ws")
            elapsed = page.evaluate("""async (mac) => {
              window.__fake.devs = window.__fake.all.filter((d) => d.deviceId !== 'iphone-id');
              const t = window.hearingJudge.audio.stream.getAudioTracks()[0];
              const t0 = performance.now();
              t.stop(); t.dispatchEvent(new Event('ended'));
              while (window.__fake.opened.at(-1) !== 'macbook-id' || window.hearingJudge.audio?.stream.getAudioTracks()[0].label !== mac) {
                if (performance.now() - t0 > 3000) return -1;
                await new Promise((r) => setTimeout(r, 10));
              }
              return performance.now() - t0;
            }""", MACBOOK)
            assert 0 <= elapsed < 1000, elapsed
            assert page.evaluate("window.hearingJudge.ws === window.__ws0 && window.hearingJudge.connected")
            page.wait_for_selector(BANNER, state="visible")
            assert "iPhone mic lost: using MacBook mic" in _banner_lines(page), _banner_lines(page)
            assert page.evaluate("document.querySelector('[data-dev=in]').value") == "macbook-id"
            page.wait_for_function("(mac) => window.hellos.some((h) => h.input_device === mac)", arg=MACBOOK)
            hello = page.evaluate("(mac) => window.hellos.find((h) => h.input_device === mac)", MACBOOK)
            assert hello["mic_switch"] == f"lost (ended): {IPHONE} -> {MACBOOK}", hello
            assert hello["devices_missing"] == ["mic"], hello
            devs = _health_input(base, page, MACBOOK)
            assert devs.get("input") == MACBOOK and devs.get("missing") == ["mic"], devs
            assert page.evaluate("localStorage.getItem('hearing.mic')") == "iphone-id"

            # (d) the iPhone is back: Retry switches back and the banner clears
            page.evaluate("window.__fake.devs = [...window.__fake.all]")
            page.locator(f"{BANNER} button").click()
            page.wait_for_function("(l) => window.hearingJudge.audio?.stream.getAudioTracks()[0].label === l", arg=IPHONE)
            page.wait_for_function("!document.querySelector('[data-dev-banner] span').textContent.includes('lost')")
            assert not any("mic" in line.lower() for line in _banner_lines(page)), _banner_lines(page)
            assert page.evaluate("document.querySelector('[data-dev=in]').value") == "iphone-id"
            devs = _health_input(base, page, IPHONE)
            assert devs.get("input") == IPHONE and devs.get("missing") == [], devs
            assert page.evaluate("window.hellos.at(-1).mic_switch") == f"retry: {MACBOOK} -> {IPHONE}"
            assert not errors, errors
            browser.close()
    finally:
        proc.kill()
        proc.wait(timeout=10)
