"""T-121: device pickers, level meter, Bluetooth badge and the hello's mic label on /judge and /simple.

Chromium runs with fake media devices (a beep as the mic); ``enumerateDevices`` is replaced by a fixed
list (a RØDE mic, the laptop mic, wired headphones, AirPods) and ``AudioContext.setSinkId`` records
its calls, so no real audio device is opened.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

INIT = """
window.__sink = [];
if (!AudioContext.prototype.setSinkId) AudioContext.prototype.setSinkId = async function () {};
const realSink = AudioContext.prototype.setSinkId;
AudioContext.prototype.setSinkId = async function (id) { window.__sink.push(id); return realSink.call(this, id).catch(() => {}); };
const dev = (kind, deviceId, label) => ({kind, deviceId, label, groupId: deviceId, toJSON() { return this; }});
navigator.mediaDevices.enumerateDevices = async () => [
  dev('audioinput', 'default', 'Default - MacBook Pro Microphone'),
  dev('audioinput', 'mic-rode', 'RODE VideoMic GO II'),
  dev('audioinput', 'mic-mac', 'MacBook Pro Microphone'),
  dev('audiooutput', 'default', 'Default - MacBook Pro Speakers'),
  dev('audiooutput', 'out-wired', 'USB-C Headphones'),
  dev('audiooutput', 'out-airpods', "Aryan's AirPods Pro"),
];
"""


def _serve() -> tuple[str, subprocess.Popen]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "HEARING_DEVICE": "cpu", "HEARING_ASR": "mock", "HEARING_TTS": "mock",
           "HEARING_VOICE_MATCH": "mock", "HEARING_FACES": "mock", "HEARING_ASD": "lipmotion",
           "HEARING_HQ_BACKEND": "local", "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    env.pop("HEARING_REMOTE_URL", None)
    env.pop("HEARING_PREFS_LOG", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "hearing", "serve", "--host", "127.0.0.1", "--port", str(port), "--no-warm",
         "--log-level", "warning", "--pipelines", "natural=passthrough; focus_hq=gain(db=-6)"],
        cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(160):
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=1) as r:
                if json.load(r)["ok"]:
                    return url, proc
        except (OSError, ValueError):
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    proc.kill()
    raise RuntimeError(f"hearing serve did not come up on {url}")


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
@pytest.mark.parametrize("path", ["/judge", "/simple"])
def test_pickers_meter_bluetooth_badge_and_hello_label(path):
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
            page.add_init_script(INIT)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            hellos: list[dict] = []

            def on_ws(ws) -> None:
                def sent(payload) -> None:
                    if isinstance(payload, str) and '"hello"' in payload:
                        hellos.append(json.loads(payload))
                ws.on("framesent", sent)

            page.on("websocket", on_ws)
            page.goto(base + path)
            page.evaluate("const d = document.getElementById('drawer'); if (d) d.open = true")
            mic, out = page.locator('[data-dev="in"]'), page.locator('[data-dev="out"]')
            page.wait_for_function("document.querySelectorAll('[data-dev=in] option').length > 1")
            assert mic.locator("option").all_inner_texts() == [
                "System default", "RODE VideoMic GO II", "MacBook Pro Microphone"]
            assert out.locator("option").all_inner_texts()[1:] == ["USB-C Headphones", "Aryan's AirPods Pro"]

            # Bluetooth badge follows the chosen output; default output is not Bluetooth here
            badge = page.locator('[data-dev="bt"]')
            assert badge.is_hidden()
            # T-140: the default output here is the built-in speakers: loud banner (<= 1 s) with an inline picker
            warn = page.locator("[data-dev-banner]")
            warn.wait_for(state="visible", timeout=1000)
            assert "laptop speakers" in warn.inner_text()
            assert page.locator("[data-dev-pick] option").all_inner_texts()[1:] == [
                "USB-C Headphones", "Aryan's AirPods Pro"]
            out.select_option("out-airpods")
            assert badge.is_visible()
            assert warn.is_hidden()
            out.select_option("out-wired")
            assert badge.is_hidden()
            out.select_option("out-airpods")
            mic.select_option("mic-rode")
            assert page.evaluate("[localStorage.getItem('hearing.mic'), localStorage.getItem('hearing.sink')]") == [
                "mic-rode", "out-airpods"]

            # Start: meter moves into a band, output is routed, hello carries the mic label
            page.locator("#start").click()
            page.wait_for_function(
                "['quiet','good','clipping'].includes(document.querySelector('[data-dev=meter]').dataset.band)")
            assert page.locator('[data-dev="db"]').inner_text().endswith("dBFS")
            assert page.locator('[data-dev="band"]').inner_text() in ("too quiet", "good", "clipping")
            page.wait_for_function("window.__sink.length > 0")
            assert page.evaluate("window.__sink")[-1] == "out-airpods"
            deadline = time.time() + 10
            while time.time() < deadline and not any(h.get("input_device") for h in hellos):
                page.wait_for_timeout(100)
            labelled = [h for h in hellos if h.get("input_device")]
            assert labelled, hellos
            assert "audio" in labelled[-1]["input_device"].lower()  # Chromium's "Fake Default Audio InputDevice"
            assert labelled[-1]["output_device"] == "Aryan's AirPods Pro"

            # the choice survives a reload
            page.reload()
            page.evaluate("const d = document.getElementById('drawer'); if (d) d.open = true")
            page.wait_for_function("document.querySelectorAll('[data-dev=in] option').length > 1")
            assert page.locator('[data-dev="in"]').input_value() == "mic-rode"
            assert page.locator('[data-dev="out"]').input_value() == "out-airpods"
            assert page.locator('[data-dev="bt"]').is_visible()
            assert not errors, errors
            browser.close()
    finally:
        proc.kill()
        proc.wait(timeout=10)
