"""T-168: under the team preset (audio only) /judge never asks getUserMedia for video, says video:false in
hello and sends no video frame. Chromium with fake media; no real device."""

from __future__ import annotations

import json

import pytest

from tests.test_device_picker import _serve

SPY = """(() => {
  window.gum = [];
  const gum = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = (c) => { window.gum.push(JSON.stringify(c)); return gum(c); };
  const ws = window.hearingJudge.ws, send = ws.send.bind(ws);
  window.texts = []; window.kinds = [];
  ws.send = (d) => {
    if (typeof d === 'string') window.texts.push(d); else window.kinds.push(new Uint8Array(d)[0]);
    send(d);
  };
})()"""


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
def test_team_preset_judge_opens_no_camera(monkeypatch):
    sync_api = pytest.importorskip("playwright.sync_api")
    monkeypatch.setenv("HEARING_DEMO_PRESET", "team")
    base, proc = _serve()
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
            page = browser.new_context(permissions=["microphone", "camera"]).new_page()
            page.set_default_timeout(15000)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(base + "/judge")
            page.wait_for_function("window.hearingJudge.connected && window.hearingJudge.telemetry?.preset?.pins")
            assert page.evaluate("window.hearingJudge.telemetry.preset.pins.audio_only") is True
            page.evaluate(SPY)
            page.click("#start")
            page.wait_for_function("window.hearingJudge.audio && window.hearingJudge.seq > 50")  # ~1 s of audio
            page.wait_for_timeout(500)
            gum = [json.loads(c) for c in page.evaluate("window.gum")]
            assert gum and not any(c.get("video") for c in gum), gum  # the mic only
            hellos = [m for m in map(json.loads, page.evaluate("window.texts")) if m.get("type") == "hello"]
            assert hellos and all(h["video"] is False for h in hellos), hellos
            kinds = set(page.evaluate("window.kinds"))
            assert kinds == {1}, kinds  # audio in only, never a video frame (kind 2)
            assert page.evaluate("window.hearingJudge.video") is None
            assert page.evaluate("document.body.classList.contains('audio-only')")
            assert not page.locator("#video").is_visible()
            assert page.evaluate("window.hearingJudge.telemetry.preset.video_dropped") == 0
            assert not page.evaluate("window.hearingJudge.cameraLost")  # no camera is not "camera lost"
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
