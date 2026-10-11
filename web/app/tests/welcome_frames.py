"""T-429 Welcome cold open: frame timings in headless Chromium, no engine, no microphone.

Serves web/app with hearing.server.static on an ephemeral port, then per profile (desktop 1440x900 DPR 2; phone
390x844 DPR 3 touch with the CPU throttled x4; the phone again with prefers-reduced-motion) records rAF intervals and
Long Animation Frames over windows: the cold open (0-6.2 s after the shell is ready), a 0.7 s hold let go early (the
spring back), a hold to the full ring (the mic cannot open here, so the disc reports the failure and the field springs
back), and 3 s of "mic live" where the store receives level events shaped like the mic's (store.emit("level"), 50 Hz;
a measurement input only, labelled as such in docs/ux-review-shots/WELCOME_WOW.md).

Run: swarm-lock --exclusive -- .venv/bin/python web/app/tests/welcome_frames.py [out.json]
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from playwright.sync_api import sync_playwright

from hearing.server.static import mount_app

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("TMPDIR", "/tmp")) / "T-429-frames.json"

DESKTOP = {"viewport": {"width": 1440, "height": 900}, "device_scale_factor": 2}
PHONE = {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 3, "is_mobile": True, "has_touch": True}
PROFILES = {
    "desktop": {"ctx": DESKTOP, "cpu": 1, "reduced": False},
    "phone-x4": {"ctx": PHONE, "cpu": 4, "reduced": False},
    "phone-x4-reduced": {"ctx": PHONE, "cpu": 4, "reduced": True},
}

COLLECT = """() => {
  const m = window.__m = { frames: [], loaf: [] };
  let last = performance.now();
  const f = (t) => { m.frames.push([t, t - last]); last = t; requestAnimationFrame(f); };
  requestAnimationFrame(f);
  try {
    new PerformanceObserver((l) => { for (const e of l.getEntries()) m.loaf.push([e.startTime, e.duration, e.blockingDuration]); })
      .observe({ type: "long-animation-frame" });
  } catch {}
}"""

WINDOW = """([t0, t1]) => {
  const m = window.__m;
  const fr = m.frames.filter(([t]) => t > t0 && t <= t1).map(([, d]) => d).sort((a, b) => a - b);
  const lo = m.loaf.filter(([t]) => t >= t0 && t <= t1);
  const p = (q) => fr.length ? +fr[Math.min(fr.length - 1, Math.floor(q * fr.length))].toFixed(1) : null;
  return { frames: fr.length, p50_ms: p(0.5), p95_ms: p(0.95), max_ms: fr.length ? +fr.at(-1).toFixed(1) : null,
           over_25ms: fr.filter((d) => d > 25).length, over_50ms: fr.filter((d) => d > 50).length,
           loaf: lo.length, loaf_ms: Math.round(lo.reduce((a, [, d]) => a + d, 0)),
           loaf_max_ms: lo.length ? Math.round(Math.max(...lo.map(([, d]) => d))) : 0 };
}"""


def serve() -> tuple[uvicorn.Server, int]:
    app = FastAPI()
    mount_app(app, ROOT / "web" / "app")
    app.mount("/console", StaticFiles(directory=ROOT / "web" / "console"), name="console-files")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return server, port


def run_profile(browser, base: str, prof: dict) -> dict:
    ctx = browser.new_context(reduced_motion="reduce" if prof["reduced"] else "no-preference", **prof["ctx"])
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)[:200]))
    if prof["cpu"] > 1:
        cdp = ctx.new_cdp_session(page)
        cdp.send("Emulation.setCPUThrottlingRate", {"rate": prof["cpu"]})
    page.add_init_script("(" + COLLECT + ")()")
    page.goto(base + "#welcome")
    page.wait_for_function("window.hearingApp?.ready === true")

    def now() -> float:
        return page.evaluate("performance.now()")

    out: dict[str, dict] = {}
    t_ready = now()

    def window(name: str, act, ms: int) -> None:
        t0 = now()
        act()
        page.wait_for_timeout(ms)
        out[name] = page.evaluate(WINDOW, [t0, now()])

    # 1. the cold open from the shell's ready flag (reduced: the still version, same window)
    page.wait_for_timeout(6200)
    out["cold open 0-6.2 s"] = page.evaluate(WINDOW, [t_ready, now()])
    page.wait_for_timeout(600)
    out["still after the cold open 1 s"] = None
    window("still after the cold open 1 s", lambda: None, 1000)

    box = page.locator("#start").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    def hold(ms: int) -> None:
        page.mouse.move(cx, cy)
        page.mouse.down()
        page.wait_for_timeout(ms)
        page.mouse.up()

    # 2. a 0.7 s hold let go early: the collapse, then the spring back
    window("hold 0.7 s + release spring (1.6 s)", lambda: hold(700), 1600)
    # 3. a hold to the full ring: the collapse completes, the mic cannot open here, the field springs back
    window("hold to full + failed (2 s)", lambda: hold(1100), 2000)
    phase = page.evaluate("document.querySelector('#start')?.dataset.phase")

    # 4. mic live: level events shaped like the mic's (the store's "level"), 50 Hz for 3 s
    page.evaluate("""() => {
      const st = window.hearingApp.store.store;
      st.set({ started: true });
      let i = 0;
      window.__lvl = setInterval(() => { i++; st.emit("level", { db: -34 + 16 * Math.sin(i / 9) + (i % 7 === 0 ? 8 : 0), band: "good" }); }, 20);
    }""")
    window("mic live 3 s (injected level events)", lambda: None, 3000)
    page.evaluate("clearInterval(window.__lvl)")
    label = page.evaluate("document.querySelector('.wl-field-label')?.textContent")
    ctx.close()
    return {"windows": out, "errors": errors, "phase_after_full_hold": phase, "label_live": label}


def main() -> None:
    server, port = serve()
    base = f"http://127.0.0.1:{port}/app/"
    results: dict = {"host": os.uname().nodename, "load_avg": os.getloadavg(), "profiles": {}}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            for name, prof in PROFILES.items():
                results["profiles"][name] = run_profile(browser, base, prof)
                print(name, json.dumps(results["profiles"][name]["windows"], indent=1), flush=True)
        finally:
            browser.close()
    server.should_exit = True
    OUT.write_text(json.dumps(results, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
