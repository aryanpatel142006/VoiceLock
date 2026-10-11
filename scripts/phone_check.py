"""Phone check over https: ``/simple`` under iPhone 14 and Pixel 7 emulation against ``hearing serve --tls``.

    .venv/bin/python scripts/phone_check.py [--url https://127.0.0.1:PORT] [--webkit]
    .venv/bin/python scripts/phone_check.py --url https://127.0.0.1:PORT/app/   # the app's phone screen (T-351)
    .venv/bin/python scripts/phone_check.py --app                                 # same, on its own server

Without ``--url`` it starts ``hearing serve --tls`` (self-signed certificate in a temp
``HEARING_TLS_DIR``, weight-free pipelines, mock ASR/TTS/faces) on a free port and stops it after.
Chromium's fake mic/camera stand in for the devices; the browser accepts the self-signed
certificate (``ignore_https_errors``, what "trust this certificate" does on a real phone).

Per device: the page loads over https and is a secure context; the socket is ``wss://``; Start
creates the AudioContext inside the tap (user activation) and requests the mic with echo
cancellation / noise suppression / AGC off; the AudioWorklet runs (audio-in frames reach the
server) and processed audio comes back; ``hello`` carries the context's rate (48 kHz on the iPhone
profile, which uses the native rate; 16 kHz on Android). ``--webkit`` also loads the iPhone profile in
Playwright's WebKit (page + wss + secure context; WebKit has no fake mic). Exit 1 on any failure.

With a ``--url`` ending in ``/app/`` (or ``--app``) it checks the app's phone screen instead (T-351): a
desktop "stage" opens ``/app/#listen``, presses Start and shows the QR (``/connect?page=/app/?watch=<id>``);
an iPhone 14 context opens that link over https and must show the mirrored state word, captions and the
four controls (Focus, Quiet, person, Release), and its Quiet tap must reach the stage session (the
mirrored telemetry says ``quiet``). Then Solo: refused while the stage session is live; once the stage
closes it starts through the SDK (AudioContext inside the tap, own ``/ws/session`` with
``client: "solo"``, mic frames sent) and requests a screen wake lock.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Every wait scales by HEARING_TEST_SLOWDOWN (default 1) so a loaded machine gets more time (Q-245).
SLOWDOWN = max(1.0, float(os.environ.get("HEARING_TEST_SLOWDOWN") or 1.0))

# Records what Start did: getUserMedia constraints + resulting track settings, AudioContext rate
# and whether it was created during the tap's user activation (iOS needs that).
INIT = """
window.__phone = {gum: [], ctx: []};
const gum = navigator.mediaDevices && navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
if (gum) navigator.mediaDevices.getUserMedia = async (c) => {
  const s = await gum(c);
  const a = s.getAudioTracks()[0];
  window.__phone.gum.push({constraints: c, audio: a ? a.getSettings() : null});
  return s;
};
const AC = window.AudioContext;
window.AudioContext = function (opts) {
  const ctx = new AC(opts);
  window.__phone.ctx.push({rate: ctx.sampleRate, opts: opts || null,
                           activation: navigator.userActivation ? navigator.userActivation.isActive : null});
  return ctx;
};
window.AudioContext.prototype = AC.prototype;
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(tls_dir: str) -> tuple[str, subprocess.Popen]:
    port = free_port()
    env = {**os.environ, "HEARING_DEVICE": "cpu", "HEARING_ASR": "mock", "HEARING_TTS": "mock",
           "HEARING_VOICE_MATCH": "mock", "HEARING_FACES": "mock", "HEARING_ASD": "lipmotion",
           "HEARING_HQ_BACKEND": "local", "HEARING_TLS_DIR": tls_dir,
           "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    env.pop("HEARING_REMOTE_URL", None)
    env.pop("HEARING_PREFS_LOG", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "hearing", "serve", "--tls", "--host", "127.0.0.1", "--port", str(port),
         "--no-warm", "--log-level", "warning", "--pipelines",
         ("natural=passthrough; focus_hq=gain(db=-6); synthetic=passthrough>synth(speak=true); "
          "text=passthrough>synth(speak=false)")],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    url = f"https://127.0.0.1:{port}"
    for _ in range(round(160 * SLOWDOWN)):
        try:  # the server writes the certificate at start; trust exactly that one
            ctx = ssl.create_default_context(cafile=str(Path(tls_dir) / "cert.pem"))
            with urllib.request.urlopen(f"{url}/health", context=ctx, timeout=1) as r:
                if json.load(r)["ok"]:
                    return url, proc
        except (OSError, ssl.SSLError, ValueError):
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    proc.kill()
    raise SystemExit(f"hearing serve --tls did not come up on {url}:\n{proc.stdout.read()}")


def check_device(browser, devices, name: str, base: str, results: list, full: bool = True) -> None:
    def check(what: str, ok: bool, detail: object = "") -> None:
        results.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} [{name}] {what}{f': {detail}' if detail != '' else ''}")

    desc = {k: v for k, v in devices[name].items() if k != "default_browser_type"}
    ctx = browser.new_context(**desc, ignore_https_errors=True,
                              **({"permissions": ["microphone", "camera"]} if full else {}))
    ctx.set_default_timeout(30000 * SLOWDOWN)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    wire = {"urls": [], "sent": [], "audio_in": 0, "audio_out": 0}

    def on_ws(ws) -> None:
        wire["urls"].append(ws.url)

        def sent(payload) -> None:
            if isinstance(payload, str):
                wire["sent"].append(json.loads(payload))
            elif payload[:1] == b"\x01":
                wire["audio_in"] += 1

        def received(payload) -> None:
            if isinstance(payload, bytes) and payload[:1] == b"\x03":
                wire["audio_out"] += 1

        ws.on("framesent", sent)
        ws.on("framereceived", received)

    page.on("websocket", on_ws)
    page.add_init_script(INIT)
    resp = page.goto(f"{base}/simple", wait_until="load")
    check("page loads over https", resp is not None and resp.ok and page.url.startswith("https://"),
          resp.status if resp else None)
    check("secure context (getUserMedia allowed)", page.evaluate("window.isSecureContext") is True)
    page.wait_for_function("window.hearingSimple && window.hearingSimple.ws && window.hearingSimple.ws.readyState === 1",
                           timeout=5000 * SLOWDOWN)
    check("socket is wss:// on the page's host", bool(wire["urls"]) and wire["urls"][0].startswith("wss://"),
          wire["urls"][:1])
    if not full:
        check("no page errors", not errors, errors[:3])
        ctx.close()
        return
    page.tap("#start") if desc.get("has_touch") else page.click("#start")
    page.wait_for_function("window.hearingSimple.audio !== null", timeout=8000 * SLOWDOWN)
    page.wait_for_timeout(1500 * SLOWDOWN)
    phone = page.evaluate("window.__phone")
    gum = phone["gum"][0] if phone["gum"] else {}
    c = (gum.get("constraints") or {}).get("audio") or {}
    check("Start requested the mic (and camera)", len(phone["gum"]) >= 1,
          [list((g["constraints"] or {}).keys()) for g in phone["gum"]])
    check("mic permission granted", page.evaluate(
        "navigator.permissions.query({name: 'microphone'}).then(p => p.state)") == "granted")
    check("echo cancellation / noise suppression / AGC off in the constraints",
          c.get("echoCancellation") is False and c.get("noiseSuppression") is False
          and c.get("autoGainControl") is False, c)
    settings = gum.get("audio") or {}
    check("the mic track honours echoCancellation=false", settings.get("echoCancellation") in (False, None),
          {k: settings.get(k) for k in ("echoCancellation", "sampleRate")})
    actx = phone["ctx"][0] if phone["ctx"] else {}
    check("AudioContext created inside the Start tap", actx.get("activation") is True, actx)
    ios = "iPhone" in name
    want = None if ios else 16000
    check(f"AudioContext rate option {'native (iOS)' if ios else '16 kHz'}",
          (actx.get("opts") or {}).get("sampleRate") == want, actx.get("opts"))
    hellos = [m for m in wire["sent"] if m.get("type") == "hello"]
    rate = page.evaluate("window.hearingSimple.audio.rate")
    check("hello carries the context's rate", bool(hellos) and hellos[-1]["sample_rate"] == rate,
          f"hello {hellos[-1]['sample_rate'] if hellos else None}, context {rate}")
    check("worklet runs: audio-in frames sent", wire["audio_in"] >= 20, wire["audio_in"])
    check("processed audio comes back (server resampled)", wire["audio_out"] >= 10, wire["audio_out"])
    check("no page errors", not errors, errors[:3])
    ctx.close()


def check_app(browser, devices, base: str, results: list) -> None:
    """The app's phone screen: laptop QR -> iPhone companion mirror + controls -> Solo gate and SDK path."""
    def check(what: str, ok: bool, detail: object = "") -> None:
        results.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} [app] {what}{f': {detail}' if detail != '' else ''}")

    t = 30000 * SLOWDOWN
    stage_ctx = browser.new_context(viewport={"width": 1440, "height": 900}, ignore_https_errors=True,
                                    permissions=["microphone"])
    stage_ctx.set_default_timeout(t)
    stage = stage_ctx.new_page()
    errors: list[str] = []
    stage.on("pageerror", lambda e: errors.append(f"stage: {e}"))
    stage.goto(f"{base}/app/#listen", wait_until="load")
    stage.wait_for_function("window.hearingApp && window.hearingApp.session && window.hearingApp.session.sessionId")
    sid = stage.evaluate("window.hearingApp.session.sessionId")
    stage.click(".ls-start")
    stage.click(".pl-toggle")
    stage.wait_for_selector(".pl-code svg")
    href = stage.get_attribute(".pl-url", "href") or ""
    check("laptop Listen shows a QR to /app/?watch=<id> (from /connect?page=)",
          href.endswith(f"/app/?watch={sid}") and stage.locator(".pl-code svg").count() == 1, href)

    phone_desc = {k: v for k, v in devices["iPhone 14"].items() if k != "default_browser_type"}
    phone_ctx = browser.new_context(**phone_desc, ignore_https_errors=True, permissions=["microphone"])
    phone_ctx.set_default_timeout(t)
    phone_ctx.add_init_script(INIT + """
window.__wake = [];
if (navigator.wakeLock) { const req = navigator.wakeLock.request.bind(navigator.wakeLock);
  navigator.wakeLock.request = (k) => { window.__wake.push(k); return req(k); }; }""")
    phone = phone_ctx.new_page()
    phone.on("pageerror", lambda e: errors.append(f"phone: {e}"))
    wire = {"urls": [], "sent": [], "audio_in": 0}

    def on_ws(ws) -> None:
        wire["urls"].append(ws.url)

        def sent(payload) -> None:
            if isinstance(payload, str):
                wire["sent"].append({"url": ws.url, **json.loads(payload)})
            elif payload[:1] == b"\x01":
                wire["audio_in"] += 1
        ws.on("framesent", sent)

    phone.on("websocket", on_ws)
    # the QR encodes the server's LAN address; this check reaches the same server on the --url host
    path = href[href.index("/app/"):]
    resp = phone.goto(f"{base}{path}", wait_until="load")
    check("companion loads over https in a secure context", resp is not None and resp.ok
          and phone.url.startswith("https://") and phone.evaluate("window.isSecureContext") is True, phone.url)
    phone.wait_for_selector('.phone[data-role="companion"][data-watch="open"]')
    check("the mirror socket is wss:///ws/watch for that session, no engine session opened",
          any(u.startswith("wss://") and f"/ws/watch?session={sid}" in u for u in wire["urls"])
          and not any("/ws/session" in u for u in wire["urls"]), wire["urls"])
    phone.wait_for_function("window.hearingWatch && window.hearingWatch.store.state.telemetry")
    word = phone.text_content(".ph-word") or ""
    check("mirrored state word shown", word.strip() not in ("", "Connecting", "Reconnecting", "Ended"), word)
    check("captions region shown", phone.locator(".ph-captions .ph-cap-lines").count() == 1)
    acts = sorted(set(phone.eval_on_selector_all("button[data-act]", "bs => bs.map(b => b.dataset.act)")))
    check("the four controls: Focus, Quiet, person, Release", acts == ["focus", "person", "quiet", "release"], acts)
    phone.tap('button[data-act="quiet"]')
    phone.wait_for_function("window.hearingWatch.store.state.telemetry.mode === 'quiet'")
    stage.wait_for_function("window.hearingApp.store && window.hearingApp.store.store.state.telemetry"
                            " && window.hearingApp.store.store.state.telemetry.mode === 'quiet'")
    check("Quiet tapped on the phone reaches the stage session (stage telemetry mode quiet)", True)
    phone.tap('button[data-act="focus"]')
    phone.wait_for_function("window.hearingWatch.store.state.telemetry.mode === 'focus_hq'")
    phone.wait_for_selector('button[data-act="focus"][aria-pressed="true"]')
    check("Focus tapped returns the stage to focus_hq, mirrored as pressed", True)
    sent = [m for m in wire["sent"] if "/ws/watch" in m["url"]]
    check("companion sends only control JSON", bool(sent) and all(m.get("type") == "control" for m in sent),
          [m.get("type") for m in sent])

    # Solo: refused while the stage session is live, allowed once it ends
    phone.goto(f"{base}/app/#phone", wait_until="load")
    phone.wait_for_selector('.phone[data-role="lobby"][data-solo="refused"]')
    why = phone.text_content(".ph-solo-why") or ""
    check("Solo refused while a stage session is live, and says why",
          phone.is_disabled(".ph-solo-start") and "stage session" in why, why[:90])
    stage_ctx.close()
    phone.wait_for_selector('.phone[data-role="lobby"][data-solo="ready"]', timeout=15000 * SLOWDOWN)
    phone.wait_for_timeout(300)
    phone.tap(".ph-solo-start")
    phone.wait_for_function("window.__wake && window.__wake.length > 0 || (window.__soloWakeLock || {}).requested")
    check("Solo requests a screen wake lock", True, phone.evaluate("window.__soloWakeLock"))
    phone.wait_for_timeout(2000 * SLOWDOWN)
    info = phone.evaluate("window.__phone")
    actx = info["ctx"][0] if info["ctx"] else {}
    check("Solo creates the AudioContext inside the tap (SDK path)", actx.get("activation") is True, actx)
    hellos = [m for m in wire["sent"] if "/ws/session" in m["url"] and m.get("type") == "hello"]
    check("Solo opens its own session as client 'solo'", any(m.get("client") == "solo" for m in hellos),
          [m.get("client") for m in hellos])
    check("Solo streams the phone mic (audio-in frames)", wire["audio_in"] >= 20, wire["audio_in"])
    phone.tap('button[data-solo="stop"]')
    check("no page errors", not errors, errors[:3])
    phone_ctx.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", help="a running hearing serve --tls (https://host:port)")
    ap.add_argument("--webkit", action="store_true", help="also load the iPhone profile in WebKit")
    ap.add_argument("--app", action="store_true", help="check the app's phone screen (/app/, T-351)")
    args = ap.parse_args(argv)
    from playwright.sync_api import sync_playwright

    proc = None
    with tempfile.TemporaryDirectory(prefix="hearing-phone-") as tmp:
        app_mode = args.app
        if args.url:
            base = args.url.rstrip("/")
            if base.endswith("/app"):
                app_mode, base = True, base[:-len("/app")]
        else:
            base, proc = start_server(tmp)
        results: list[bool] = []
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(args=[
                    "--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
                    "--autoplay-policy=user-gesture-required",
                ])
                if app_mode:
                    check_app(browser, p.devices, base, results)
                for name in () if app_mode else ("iPhone 14", "Pixel 7"):
                    check_device(browser, p.devices, name, base, results)
                browser.close()
                if args.webkit:
                    wk = p.webkit.launch()
                    check_device(wk, p.devices, "iPhone 14", base, results, full=False)
                    wk.close()
        finally:
            if proc:
                proc.terminate()
                proc.wait(10)
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
