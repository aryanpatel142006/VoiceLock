"""T-366 attempt 2: portable browser evidence for the Welcome screen.

For every input path it records, at the moment the AudioContext is constructed (inside session.start()):
navigator.userActivation.isActive, and 1.5 s later the context's state ("running" means audio can play), plus the
sequence of disc phases (holding -> full -> starting, or holding -> full -> armed -> starting on a release).
Runs in Chromium (CDP touch) and WebKit (mouse + keyboard; Playwright has no touch-hold in WebKit).
"""

import json
import os
import sys

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

# usage: hearing serve --host 127.0.0.1 --port <free> --no-warm --pipelines "natural=passthrough; focus_hq=gain(db=-6)"
#        PORT=<that port> python welcome_walk.py <out dir>
port = os.environ["PORT"]
URL = f"http://127.0.0.1:{port}/app/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
IPHONE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Mobile/15E148 Safari/604.1"
)

INIT = r"""
(() => {
  window.__ev = { ctx: [], phases: [] };
  const Orig = window.AudioContext || window.webkitAudioContext;
  if (Orig) {
    const Wrapped = function (...a) {
      const c = new Orig(...a);
      const rec = { t: performance.now(), active: navigator.userActivation ? navigator.userActivation.isActive : null,
                    phase: document.querySelector('.wl-disc')?.dataset.phase ?? null, state0: c.state };
      window.__ev.ctx.push(rec);
      setTimeout(() => { rec.state1500 = c.state; }, 1500);
      return c;
    };
    Wrapped.prototype = Orig.prototype;
    window.AudioContext = Wrapped;
  }
  document.addEventListener('DOMContentLoaded', () => {
    new MutationObserver(() => {
      const d = document.querySelector('.wl-disc');
      if (!d) return;
      const p = d.dataset.phase, a = window.__ev.phases;
      if (a[a.length - 1] !== p) a.push(p);
    }).observe(document.body, { subtree: true, attributes: true, attributeFilter: ['data-phase'], childList: true });
  });
})();
"""


def ctx_page(b, engine, **kw):
    kw.setdefault("permissions", ["microphone"])
    c = b.new_context(**kw)
    c.add_init_script(INIT)
    p = c.new_page()
    errs = []
    p.on(
        "console",
        lambda m: errs.append(f"{m.type}: {m.text}") if m.type == "error" else None,
    )
    p.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    p.on("requestfailed", lambda r: errs.append(f"reqfail: {r.url} {r.failure}"))
    return c, p, errs


def shots(b, engine, res):
    for name, vp, extra in [
        ("1920", {"width": 1920, "height": 1080}, {}),
        (
            "390",
            {"width": 390, "height": 844},
            {
                "is_mobile": True,
                "has_touch": True,
                "device_scale_factor": 2,
                "user_agent": IPHONE_UA,
            },
        ),
    ]:
        if engine == "webkit" and extra.get("is_mobile") is None:
            pass
        c, p, errs = ctx_page(b, engine, viewport=vp, **extra)
        p.goto(URL)
        p.wait_for_selector(".wl-disc")
        p.wait_for_timeout(1500)
        p.screenshot(path=f"{OUT}/welcome-{name}-{engine}.jpg", type="jpeg", quality=72)
        info = p.evaluate("""() => ({screen: document.body.dataset.screen,
          hasScene: document.documentElement.classList.contains('has-scene'),
          placeholderShown: getComputedStyle(document.querySelector('.wl-field')).display !== 'none',
          sceneLabel: document.querySelector('.scene-label')?.textContent ?? null,
          hero: getComputedStyle(document.querySelector('.wl-hero')).fontSize,
          overflowX: document.documentElement.scrollWidth > innerWidth,
          install: document.querySelector('.wl-install')?.textContent || null})""")
        box = p.locator(".wl-disc").bounding_box()
        p.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        p.mouse.down()
        p.wait_for_timeout(400)
        info["p_at_400ms"] = p.evaluate(
            "getComputedStyle(document.querySelector('.wl-disc')).getPropertyValue('--wl-p')"
        )
        p.screenshot(
            path=f"{OUT}/welcome-{name}-{engine}-holding.jpg", type="jpeg", quality=72
        )
        p.mouse.up()
        p.wait_for_timeout(500)
        info["after_early_release"] = p.evaluate(
            "[document.querySelector('.wl-disc').dataset.phase, document.body.dataset.screen]"
        )
        info["errors"] = errs
        res[f"shot_{name}"] = info
        c.close()


def full(b, engine, kind, focus_disc=True, hold_ms=1300, **kw):
    c, p, errs = ctx_page(b, engine, viewport={"width": 1280, "height": 800}, **kw)
    p.goto(URL)
    p.wait_for_selector(".wl-disc")
    p.wait_for_timeout(600)
    box = p.locator(".wl-disc").bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    if kind == "mouse":
        p.mouse.move(x, y)
        p.mouse.down()
        p.wait_for_timeout(hold_ms)
        p.mouse.up()
    elif kind == "touch":
        cdp = c.new_cdp_session(p)
        cdp.send(
            "Input.dispatchTouchEvent",
            {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]},
        )
        p.wait_for_timeout(hold_ms)
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    elif kind in ("Space", "Enter"):
        key = " " if kind == "Space" else "Enter"
        if focus_disc:
            p.focus(".wl-disc")
        p.keyboard.down(key)
        p.wait_for_timeout(hold_ms)
        p.keyboard.up(key)
    elif kind == "short-space":
        p.keyboard.down(" ")
        p.wait_for_timeout(300)
        p.keyboard.up(" ")
    p.wait_for_timeout(2600)
    r = p.evaluate("""() => ({screen: document.body.dataset.screen, hash: location.hash,
        started: window.hearingApp?.session?.store?.state?.started ?? null,
        audioContext: window.__ev.ctx, phases: window.__ev.phases,
        problem: document.querySelector('.wl-problem')?.textContent ?? null})""")
    r["errors"] = errs
    r["hold_ms"] = hold_ms if kind != "short-space" else 300
    c.close()
    return r


res = {}
with sync_playwright() as pw:
    for engine in ("chromium", "webkit"):
        if engine == "chromium":
            b = pw.chromium.launch(
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                ]
            )
        else:
            b = pw.webkit.launch()
        e = res[engine] = {"version": b.version}
        try:
            shots(b, engine, e)
        except PlaywrightError as ex:  # record, keep going
            e["shots_error"] = repr(ex)
        runs = [
            ("mouse", {}),
            ("Space", {}),
            ("Enter", {}),
            ("Space_page", {}),
            ("short-space", {}),
            ("mouse_reduced", {"reduced_motion": "reduce"}),
            ("mouse_denied", {"permissions": []}),
        ]
        if engine == "chromium":
            runs.insert(1, ("touch", {"has_touch": True}))
            runs.pop()  # fake-ui grants every request; the denied path is the WebKit run (and attempt 1's check2.py)
        for name, kw in runs:
            kind = {
                "Space_page": "Space",
                "mouse_reduced": "mouse",
                "mouse_denied": "mouse",
            }.get(name, name)
            try:
                e[name] = full(b, engine, kind, focus_disc=(name != "Space_page"), **kw)
            except PlaywrightError as ex:
                e[name] = {"error": repr(ex)}
        b.close()
with open(f"{OUT}/welcome-walk.json", "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
