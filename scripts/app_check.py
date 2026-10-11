"""Walk the app (/app/) in headless Chromium like a judge would, and fail on what a judge would see.

    .venv/bin/python scripts/app_check.py [--output DIR]
    .venv/bin/python scripts/app_check.py https://localhost:8000 [--live] [--shots DIR] [--wav FILE]

With no origin (the journey, T-354): start `hearing serve --tls --port 0` on loopback with weight-free mock
pipelines, then at 390x844, 1440x900 and 1920x1080 walk Welcome -> Setup -> People -> Listen with Chromium's fake
microphone, flip Selective/Raw and Focus/Quiet, and check the PWA (secure context, manifest fields, every icon 200 at
its declared size, the service worker controlling /app/ before and after reload). Any console error fails. Six
screenshots per viewport, results.json, server.log and browser-WxH.json go to --output. No real device is opened.

With an origin (T-356): idle walk, at 390x844 (touch), 768x1024 (touch), 1440x900 and 1920x1080: every screen in the
nav and the Diagnostics sheet. Each step fails on a page error, a console error, a sideways scroll
(scrollWidth > innerWidth), a "Not in this build" placeholder in the nav, or (Listen) a green pressed control while
nothing plays.

--live (1440x900 and 390x844): headless Chromium with a fake microphone that loops `--wav` (default: the team
fixture's held-out Aryan clip under $HEARING_FIXTURES_DIR) presses Start listening and walks every Listen state the
engine reaches on its own telemetry: Starting, Loading voice / Searching, Isolated (waits up to 40 s), Raw, Quiet,
Release, Person 1 again, Stop. On the first live size, while the stage is isolated and again under Raw, an
iPhone-sized context opens the phone companion (`/app/?watch=<session>`, T-351) and checks the same rules on it. It records the rAF frame intervals for 5 s while isolated (headless: compositor
timing only, not a display's vsync). No real microphone or speaker is opened.

Exit 0 when every check passed, 1 when one failed. The origin walk's JSON report goes to stdout (and
DIR/app_check.json with --shots); the journey writes DIR/results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urljoin

REPO = Path(__file__).resolve().parents[1]
SIZES = ((390, 844), (1440, 900), (1920, 1080))
SLOWDOWN = max(1.0, float(os.environ.get('HEARING_TEST_SLOWDOWN') or 1))


def validate_manifest(manifest: dict) -> None:
    for key in ('name', 'short_name', 'theme_color', 'background_color'):
        assert isinstance(manifest.get(key), str) and manifest[key].strip(), f'manifest: missing {key}'
    for key in ('id', 'start_url', 'scope'):
        assert manifest.get(key) == '/app/', f'manifest: {key} must be /app/'
    assert manifest.get('display') == 'standalone', 'manifest: display must be standalone'
    icons = manifest.get('icons', [])
    for size, purpose in (('192x192', 'any'), ('512x512', 'any'), ('512x512', 'maskable')):
        assert any(size in i.get('sizes', '').split()
                   and purpose in i.get('purpose', 'any').split()
                   and i.get('type') == 'image/png' and i.get('src')
                   for i in icons), f'manifest: missing {size} {purpose} PNG'


@contextmanager
def server(output: Path):
    """Let the CLI bind port zero; retain diagnostics and always reap the child."""
    with tempfile.TemporaryDirectory(prefix='T-354-app-check-') as scratch:
        env = {**os.environ, 'HEARING_DEVICE': 'cpu', 'HEARING_ASR': 'mock',
               'HEARING_TTS': 'mock', 'HEARING_VOICE_MATCH': 'mock', 'HEARING_FACES': 'mock',
               'HEARING_ASD': 'lipmotion', 'HEARING_HQ_BACKEND': 'local',
               'HEARING_TLS_DIR': scratch, 'HEARING_REMOTE_URL': '', 'HEARING_DEMO_PRESET': '',
               'HEARING_PROFILES_DIR': '', 'HEARING_VOICE_PROFILE': '', 'HEARING_PREFS_LOG': '',
               'HEARING_SESSION_LOG': '', 'HEARING_REPLAY_DIR': scratch,
               'PYTHONPATH': f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
        command = [sys.executable, '-m', 'hearing', 'serve', '--tls', '--host', '127.0.0.1',
                   '--port', '0', '--no-warm', '--log-level', 'warning', '--pipelines',
                   ('natural=passthrough;focus_hq=gain(db=-6);synthetic=passthrough;'
                   'text=passthrough;name_call=passthrough')]
        log_path = output / 'server.log'
        with log_path.open('w') as log:
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 60 * SLOWDOWN
                while time.monotonic() < deadline:
                    match = re.search(r'health: (https://127\.0\.0\.1:\d+)/health', log_path.read_text())
                    if match:
                        base = match[1]
                        try:
                            tls = ssl.create_default_context(cafile=str(Path(scratch) / 'cert.pem'))
                            with urllib.request.urlopen(base + '/health', context=tls, timeout=1) as response:
                                if json.load(response)['ok']:
                                    break
                        except (OSError, ValueError):
                            pass
                    if proc.poll() is not None:
                        raise RuntimeError(f'Server exited: {log_path.read_text()}')
                    time.sleep(0.1)
                else:
                    raise RuntimeError(f'Server startup timed out: {log_path.read_text()}')
                yield base
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)


def check_pwa(page, base):
    assert page.evaluate('isSecureContext'), 'not a secure context'
    manifest_url = page.locator('link[rel="manifest"]').evaluate('(el) => el.href')
    response = page.request.get(manifest_url)
    assert response.status == 200, 'manifest did not load'
    manifest = response.json()
    validate_manifest(manifest)
    for icon in manifest['icons']:
        url = urljoin(manifest_url, icon['src'])
        assert url.startswith(base + '/app/'), f'nonlocal icon: {url}'
        response = page.request.get(url)
        assert response.status == 200, f'icon HTTP {response.status}: {url}'
        dimensions = page.evaluate('''url => new Promise((resolve, reject) => {
          const img = new Image(); img.onload = () => resolve(`${img.naturalWidth}x${img.naturalHeight}`);
          img.onerror = () => reject(new Error(`Invalid icon: ${url}`)); img.src = url;
        })''', url)
        assert dimensions in icon['sizes'].split(), f'icon dimensions: {url}: {dimensions}'
    page.wait_for_function('navigator.serviceWorker.controller !== null')
    page.reload(wait_until='load')
    page.wait_for_function('window.hearingApp?.ready && navigator.serviceWorker.controller !== null')
    worker = page.evaluate('''async () => {
      const r = await navigator.serviceWorker.getRegistration();
      return {scope: r.scope, script: navigator.serviceWorker.controller.scriptURL};
    }''')
    assert worker == {'scope': base + '/app/', 'script': base + '/app/sw.js'}, worker
    return manifest


def check_viewport(browser, base, output, width, height):
    from playwright.sync_api import expect

    context = browser.new_context(viewport={'width': width, 'height': height},
                                  device_scale_factor=1, ignore_https_errors=True,
                                  permissions=['microphone'])
    context.set_default_timeout(20000 * SLOWDOWN)
    page = context.new_page()
    errors, sent, received, sockets = [], [], [], []
    frames = {'in': 0, 'out': 0}
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)

    def websocket(ws):
        sockets.append(ws.url)

        def frame(payload, incoming):
            if isinstance(payload, str):
                event = json.loads(payload)
                (received if incoming else sent).append(event)
                if incoming and event.get('type') == 'error':
                    errors.append(f'engine: {event}')
            elif payload[:1] == (b'\x03' if incoming else b'\x01'):
                frames['out' if incoming else 'in'] += 1

        ws.on('framesent', lambda payload: frame(payload, False))
        ws.on('framereceived', lambda payload: frame(payload, True))

    page.on('websocket', websocket)

    def wait_for(predicate, description):
        deadline = time.monotonic() + 20 * SLOWDOWN
        while not predicate() and time.monotonic() < deadline:
            page.wait_for_timeout(50)
        assert predicate(), description

    def shot(name):
        page.evaluate('document.fonts.ready')
        page.evaluate('window.scrollTo(0, 0)')
        # Capture the completed cut/control transition, not its previous-state overlay.
        # The live audio scene keeps running; never freeze or fabricate its levels.
        page.evaluate('''() => Promise.all(document.getAnimations()
          .filter(a => a.effect?.getComputedTiming().endTime !== Infinity)
          .map(a => a.finished.catch(() => {})))''')
        page.screenshot(path=str(output / f'{name}-{width}x{height}.png'))

    def screen(name):
        expect(page.locator(f'#stage > [data-screen="{name}"]')).to_be_visible()
        expect(page.locator('[data-placeholder]')).to_have_count(0)

    def control(selector, key, value):
        before = len(sent)
        page.locator(selector).click()
        expect(page.locator(selector)).to_have_attribute('aria-pressed', 'true')
        wait_for(lambda: any(m.get('type') == 'control' and m.get(key) == value
                             for m in sent[before:]), f'no {key}={value} control on WebSocket')

    try:
        response = page.goto(base + '/app/', wait_until='load')
        assert response.status == 200
        page.wait_for_function('window.hearingApp?.ready')
        manifest = check_pwa(page, base)
        screen('welcome')
        shot('welcome')
        page.locator('#start').focus()
        page.keyboard.down('Enter')
        try:
            screen('setup')  # hold until the app advances; no fixed-duration sleep
        finally:
            page.keyboard.up('Enter')
        # Holding Start on Welcome opens the mic (T-366); press Start mic only when it did not.
        expect(page.locator('.su-mic-btn')).to_be_enabled()
        if page.locator('.su-mic-btn').inner_text().strip() == 'Start mic':
            page.locator('.su-mic-btn').click()
        expect(page.locator('.su-mic-btn')).to_have_text('Stop mic')
        wait_for(lambda: frames['in'] > 0 and frames['out'] > 0,
                 'fake mic must send audio and receive processed audio')
        assert sockets and all(url.startswith(base.replace('https:', 'wss:') + '/') for url in sockets)
        shot('setup')
        page.locator('.su-next').click()
        screen('people')
        shot('people')
        page.locator('.pp-next').click()
        screen('listen')
        expect(page.locator('.listen')).to_be_visible()
        shot('listen-selective')
        control('button[data-ab="raw"]', 'ab', 'raw')
        shot('listen-raw')
        control('button[data-ab="processed"]', 'ab', 'processed')
        control('button[data-mode="quiet"]', 'mode', 'quiet')
        wait_for(lambda: any(m.get('type') == 'telemetry' and m.get('mode') == 'quiet'
                             for m in received), 'server must acknowledge Quiet in telemetry')
        expect(page.locator('.ls-captions')).to_be_visible()
        shot('listen-quiet')
        control('button[data-mode="focus_hq"]', 'mode', 'focus_hq')
        assert not errors, '\n'.join(errors)
        return {'viewport': [width, height], 'passed': True, 'audio_frames': frames,
                'console_errors': errors, 'manifest': manifest, 'service_worker_controlling': True}
    finally:
        (output / f'browser-{width}x{height}.json').write_text(json.dumps(
            {'errors': errors, 'sent': sent, 'audio_frames': frames}, indent=2) + '\n')
        context.close()


VIEWPORTS = {
    "390": {"viewport": {"width": 390, "height": 844}, "device_scale_factor": 2, "is_mobile": True, "has_touch": True},
    "768": {"viewport": {"width": 768, "height": 1024}, "has_touch": True, "is_mobile": True},
    "1440": {"viewport": {"width": 1440, "height": 900}},
    "1920": {"viewport": {"width": 1920, "height": 1080}},
}
IPHONE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
             "Version/18.0 Mobile/15E148 Safari/604.1")
# Console lines that are not the app's: Chromium logs the untrusted self-signed cert on the SW fetch itself.
BENIGN = ("An SSL certificate error occurred when fetching the script", "Failed to register a ServiceWorker")


def fetch_json(url: str) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx, timeout=10) as r:
        return json.loads(r.read())


def default_wav() -> str | None:
    roots = [os.environ.get("HEARING_FIXTURES_DIR"), str(Path(__file__).resolve().parents[1] / "eval/fixtures/generated")]
    for root in filter(None, roots):
        p = Path(root) / "team/aryan/holdout-01.wav"
        if p.exists():
            return str(p)
    return None


class Walk:
    def __init__(self, base: str, shots: Path | None):
        self.base = base.rstrip("/")
        self.shots = shots
        self.failures: list[str] = []
        self.steps: list[dict] = []

    def fail(self, where: str, why: str) -> None:
        self.failures.append(f"{where}: {why}")

    def shot(self, page, name: str) -> str | None:
        if not self.shots:
            return None
        png = self.shots / f"{name}.png"
        page.screenshot(path=str(png))
        return str(png)

    def page(self, browser, size: str):
        opts = dict(VIEWPORTS[size], ignore_https_errors=True)
        if size == "390":
            opts["user_agent"] = IPHONE_UA
        ctx = browser.new_context(**opts, permissions=["microphone"])
        page = ctx.new_page()
        page.console_errors = []
        page.on("console", lambda m: m.type == "error" and not any(b in m.text for b in BENIGN)
                and page.console_errors.append(m.text))
        page.on("pageerror", lambda e: page.console_errors.append(f"pageerror: {e}"))
        return ctx, page

    def layout(self, page, where: str) -> dict:
        info = page.evaluate("""() => ({
          sw: document.documentElement.scrollWidth, iw: innerWidth,
          placeholder: !!document.querySelector('#stage [data-placeholder]'),
          nav: [...document.querySelectorAll('#nav a')].map(a => a.dataset.screen),
          files: (window.hearingApp && hearingApp.files.screens) || [],
          greenIdle: (() => {
            const l = document.querySelector('.listen');
            if (!l || l.dataset.live === 'true') return false;
            return [...l.querySelectorAll('.ls-seg[aria-pressed="true"]')].some(b => {
              const c = getComputedStyle(b).backgroundColor.match(/[\\d.]+/g).map(Number);
              return c[1] > 150 && c[0] < 140 && (c[3] ?? 1) > .5;  // a green fill
            });
          })(),
          text: document.querySelector('#stage')?.innerText.slice(0, 400) || '',
        })""")
        if info["sw"] > info["iw"]:
            self.fail(where, f"sideways scroll: scrollWidth {info['sw']} > innerWidth {info['iw']}")
        if info["placeholder"]:
            self.fail(where, "a 'Not in this build' placeholder is on screen")
        missing = [s for s in info["nav"] if s != "welcome" and s not in info["files"]]
        if missing:
            self.fail(where, f"nav links to screens the server lacks: {missing}")
        if info["greenIdle"]:
            self.fail(where, "a pressed control is green while nothing plays")
        for line in page.console_errors:
            self.fail(where, f"console: {line[:200]}")
        page.console_errors.clear()
        return info

    def idle(self, browser, size: str) -> None:
        ctx, page = self.page(browser, size)
        page.goto(f"{self.base}/app/#welcome", wait_until="networkidle")
        page.wait_for_function("window.hearingApp && hearingApp.ready")
        screens = page.evaluate("[...document.querySelectorAll('#nav a')].map(a => a.dataset.screen)")
        for sid in screens:
            page.evaluate(f"location.hash = '{sid}'")
            page.wait_for_function(f"document.body.dataset.screen === '{sid}'")
            page.wait_for_timeout(900)
            where = f"{size}/{sid}"
            info = self.layout(page, where)
            self.steps.append({"step": where, "shot": self.shot(page, f"{size}-{sid}"), "scrollWidth": info["sw"]})
        if "diag" in page.evaluate("hearingApp.files.screens"):
            page.evaluate("location.hash = 'listen'")
            page.wait_for_timeout(600)
            page.keyboard.press("d")
            page.wait_for_timeout(700)
            where = f"{size}/diag"
            if page.evaluate("document.querySelector('#sheet').hidden"):
                self.fail(where, "D did not open the Diagnostics sheet")
            close = page.evaluate("""() => { const b = document.querySelector('.sheet-close');
              if (!b) return null; const r = b.getBoundingClientRect(); return [r.left, r.right, innerWidth]; }""")
            if not close or close[1] > close[2] or close[0] < 0:
                self.fail(where, f"the sheet's Close is outside the viewport: {close}")
            self.layout(page, where)
            self.steps.append({"step": where, "shot": self.shot(page, f"{size}-diag")})
        ctx.close()

    def companion(self, browser, page, out: dict, name: str, phone=None):
        """Open (or re-read) the phone companion mirroring `page`'s stage session; returns (ctx, page) to reuse."""
        if phone is None:
            sid = page.evaluate("hearingApp.store?.store?.state?.sessionId || null")
            if not sid or "phone" not in page.evaluate("hearingApp.files.screens"):
                return None
            _, phone = self.page(browser, "390")
            phone.goto(f"{self.base}/app/?watch={sid}", wait_until="networkidle")
            phone.wait_for_function("document.querySelector('.phone[data-live=\"true\"]')", timeout=15000)
        phone.wait_for_timeout(1500)
        where = f"390/companion-{name}"
        info = self.layout(phone, where)
        m = phone.evaluate("""() => { const f = document.querySelector('.phone'); const q = s => f.querySelector(s)?.innerText || '';
          const green = [...f.querySelectorAll('.ph-btn[aria-pressed="true"]')].some(b => {
            const c = getComputedStyle(b).backgroundColor.match(/[\\d.]+/g).map(Number);
            return c[1] > 150 && c[0] < 140 && (c[3] ?? 1) > .5; });
          return { tone: f.dataset.tone, ab: f.dataset.ab, live: f.dataset.live, word: q('.ph-word'), stage: q('.ph-stage'),
                   who: q('.ph-who'), green }; }""")
        if m["green"] and not (m["tone"] == "focus" and m["ab"] == "processed"):
            self.fail(where, f"a pressed phone control is green while the stage is not isolated ({m['tone']}, {m['ab']})")
        out["states"][f"companion-{name}"] = m
        self.steps.append({"step": where, "shot": self.shot(phone, f"390-companion-{name}"), "scrollWidth": info["sw"], **m})
        return phone

    def live(self, browser, size: str, settle_s: float = 40.0, companion: bool = False) -> dict:
        ctx, page = self.page(browser, size)
        out: dict = {"size": size, "states": {}}
        page.goto(f"{self.base}/app/#listen", wait_until="networkidle")
        page.wait_for_function("window.hearingApp && hearingApp.ready && document.querySelector('.listen')")
        page.wait_for_function("document.querySelector('.listen').dataset.state !== 'lost'", timeout=15000)
        page.wait_for_timeout(800)
        self.record(page, out, size, "idle")
        page.click(".ls-start")
        t0 = time.monotonic()
        seen: list[str] = []
        while time.monotonic() - t0 < settle_s:
            st = page.evaluate("document.querySelector('.listen').dataset.state")
            if not seen or seen[-1] != st:
                seen.append(st)
                self.record(page, out, size, f"live-{st}")
            if st == "isolated":
                break
            page.wait_for_timeout(250)
        out["sequence"] = seen
        out["to_isolated_s"] = round(time.monotonic() - t0, 1) if seen and seen[-1] == "isolated" else None
        if out["to_isolated_s"] is None:
            self.fail(f"{size}/live", f"never isolated in {settle_s:.0f} s (states {seen})")
        page.wait_for_timeout(1500)
        self.record(page, out, size, "isolated")
        phone = self.companion(browser, page, out, "isolated") if companion else None
        out["frames"] = page.evaluate("""() => new Promise(done => {
          const ts = []; const end = performance.now() + 5000;
          const f = (t) => { ts.push(t); if (t < end) requestAnimationFrame(f); else {
            const d = ts.slice(1).map((v, i) => v - ts[i]).sort((a, b) => a - b);
            done({ frames: d.length, fps: +(1000 * d.length / (ts.at(-1) - ts[0])).toFixed(1),
                   p50_ms: +d[Math.floor(d.length / 2)].toFixed(1), p99_ms: +d[Math.floor(d.length * .99)].toFixed(1),
                   over_25ms: d.filter(x => x > 25).length }); } };
          requestAnimationFrame(f); })""")
        page.keyboard.press("Space")
        page.wait_for_timeout(1200)
        self.record(page, out, size, "raw")
        if phone:
            self.companion(browser, page, out, "raw", phone)
            phone.context.close()
        page.keyboard.press("Space")
        page.wait_for_timeout(600)
        page.keyboard.press("Shift+Q")
        page.wait_for_timeout(6000)
        self.record(page, out, size, "quiet")
        page.keyboard.press("Shift+F")
        page.wait_for_timeout(800)
        page.keyboard.press("Shift+0")
        page.wait_for_timeout(2500)
        self.record(page, out, size, "release")
        page.keyboard.press("1")
        page.wait_for_timeout(1500)
        page.click(".ls-start")
        page.wait_for_timeout(1500)
        self.record(page, out, size, "stopped")
        ctx.close()
        return out

    def record(self, page, out: dict, size: str, name: str) -> None:
        where = f"{size}/listen-{name}"
        info = self.layout(page, where)
        m = page.evaluate("""() => { const l = document.querySelector('.listen'); const q = s => l.querySelector(s)?.innerText || '';
          return { state: l.dataset.state, tone: l.dataset.tone, ab: l.dataset.ab, live: l.dataset.live, word: q('.ls-word'),
                   who: q('.ls-who'), sentence: q('.ls-sentence'), hear: q('.ls-hear'),
                   truth: [...l.querySelectorAll('.ls-fact')].map(f => f.innerText.replace(/\\s+/g, ' ')).join(' | ') }; }""")
        for bad in ("overrun", "known voice", " — "):
            if bad in (m["who"] + m["sentence"]).lower():
                self.fail(where, f"engine words on stage: {m['who']!r} / {m['sentence']!r}")
        out["states"][name] = m
        self.steps.append({"step": where, "shot": self.shot(page, f"{size}-listen-{name}"), "scrollWidth": info["sw"], **m})


def journey_main(args) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    results = {'passed': False, 'fixture': 'Chromium fake mic; local mock/weight-free pipelines', 'checks': []}
    try:
        from playwright.sync_api import sync_playwright

        with server(args.output) as base, sync_playwright() as pw:
            browser = pw.chromium.launch(args=[
                '--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream',
                '--mute-audio', '--ignore-certificate-errors',
            ])
            try:
                results['browser'] = browser.version
                for width, height in SIZES:
                    results['checks'].append(check_viewport(browser, base, args.output, width, height))
                    print(f'PASS {width}x{height}: journey, audio, A/B, Quiet, PWA, zero console errors')
            finally:
                browser.close()
        results['passed'] = True
    except Exception as error:  # noqa: BLE001 — CLI boundary records any failure and exits 1
        results['error'] = f'{type(error).__name__}: {error}'
        print(f"FAIL {results['error']}", file=sys.stderr)
    finally:
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return 0 if results['passed'] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base", nargs="?", default=None,
                    help="origin of a running server, e.g. https://localhost:8000; omit it to start "
                         "`hearing serve --tls` on a free port and run the journey")
    ap.add_argument("--output", type=Path, default=REPO / ".swarm-run/app-check",
                    help="journey only: screenshots, server/browser logs and results.json directory")
    ap.add_argument("--live", action="store_true", help="also walk the live Listen states with a fake mic")
    ap.add_argument("--wav", default=None, help="fake mic clip for --live (default: team/aryan/holdout-01.wav)")
    ap.add_argument("--shots", type=Path, default=None, help="save a PNG per step here")
    ap.add_argument("--sizes", default="390,768,1440,1920")
    ap.add_argument("--live-sizes", default="1440,390")
    args = ap.parse_args(argv)
    if args.base is None:
        if args.live or args.shots or args.wav:
            ap.error("--live, --shots and --wav need a server origin; the journey writes to --output")
        return journey_main(args)
    from playwright.sync_api import sync_playwright

    health = fetch_json(f"{args.base.rstrip('/')}/health")
    files = fetch_json(f"{args.base.rstrip('/')}/app/_files.json")
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
    walk = Walk(args.base, args.shots)
    report: dict = {"health": {k: health.get(k) for k in ("ok", "device", "warm", "profiles", "version")},
                    "screens": files.get("screens"), "live": []}
    chromium_args = []
    wav = args.wav or default_wav()
    if args.live:
        if not wav:
            print("--live needs --wav (no team/aryan/holdout-01.wav under $HEARING_FIXTURES_DIR)", file=sys.stderr)
            return 2
        chromium_args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
                         f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required"]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(args=chromium_args)
        for size in args.sizes.split(","):
            walk.idle(browser, size.strip())
        if args.live:
            for i, size in enumerate(args.live_sizes.split(",")):
                report["live"].append(walk.live(browser, size.strip(), companion=i == 0))
        browser.close()
    report["steps"] = walk.steps
    report["failures"] = walk.failures
    report["ok"] = not walk.failures
    text = json.dumps(report, indent=1)
    if args.shots:
        (args.shots / "app_check.json").write_text(text)
    print(text)
    print(("app_check: OK" if report["ok"] else f"app_check: {len(walk.failures)} failure(s)"), file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
