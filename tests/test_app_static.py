"""T-345: the M14 app shell at ``/app/`` (PWA scaffold) next to the untouched ``/judge`` and ``/simple``."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from hearing.server.app import create_app
from hearing.server.session import ServerConfig, default_console_dir
from hearing.server.static import APP_DIR, app_files, mount_app

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setenv("HEARING_DEVICE", "cpu")
    monkeypatch.delenv("HEARING_APP_DIR", raising=False)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(ServerConfig()))


def test_app_shell_is_served_no_cache_and_app_redirects(client):
    r = client.get("/app/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert r.headers["cache-control"] == "no-cache"
    assert 'src="app.js"' in r.text and 'id="stage"' in r.text
    r = client.get("/app", follow_redirects=False)
    assert r.status_code in (307, 308) and r.headers["location"] == "/app/"
    for path in ("/app/app.js", "/app/app.css", "/app/sw.js"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", path
    assert client.get("/app/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/app/sw.js").headers["service-worker-allowed"] == "/app/"


def test_judge_simple_and_console_unchanged(client):
    console = default_console_dir()
    for path, name in (("/judge", "judge.html"), ("/simple", "simple.html"), ("/", "index.html")):
        r = client.get(path)
        assert r.status_code == 200 and r.text == (console / name).read_text(), path
    assert client.get("/protocol.js").status_code == 200  # the console's relative imports still resolve
    assert client.get("/health").json()["ok"] is True


def test_ios_and_pwa_meta_in_the_shell():
    html = (APP_DIR / "index.html").read_text()
    for needle in ('name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover"',
                   'name="apple-mobile-web-app-capable" content="yes"',
                   'name="apple-mobile-web-app-status-bar-style" content="black-translucent"',
                   'rel="apple-touch-icon" sizes="180x180" href="icons/apple-touch-icon.png"',
                   'rel="manifest" href="manifest.webmanifest"', 'name="theme-color"'):
        assert needle in html, needle
    css = (APP_DIR / "app.css").read_text()
    assert "env(safe-area-inset-top" in css and "prefers-reduced-motion" in css
    assert "http://" not in html and "https://" not in html  # no runtime CDN


def test_manifest_fields_and_icons(client):
    r = client.get("/app/manifest.webmanifest")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/manifest+json")
    m = r.json()
    assert m["start_url"] == "/app/" and m["scope"] == "/app/" and m["display"] == "standalone"
    assert m["name"] and m["short_name"] and m["theme_color"] and m["background_color"]
    sizes = {(i["sizes"], i.get("purpose", "any")) for i in m["icons"]}
    assert {("192x192", "any"), ("512x512", "any"), ("512x512", "maskable")} <= sizes
    for icon in m["icons"] + [{"src": "icons/apple-touch-icon.png", "sizes": "180x180"}]:
        r = client.get(f"/app/{icon['src']}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png", icon
        w, h = (int(x) for x in icon["sizes"].split("x"))
        assert Image.open(APP_DIR / icon["src"]).size == (w, h), icon
    assert Image.open(APP_DIR / "icons/apple-touch-icon.png").mode == "RGB"  # iOS draws alpha as black


def test_files_listing_drives_the_lazy_registry(tmp_path):
    (tmp_path / "index.html").write_text("<title>app</title>")
    (tmp_path / "app.js").write_text("")
    (tmp_path / "store.js").write_text("")
    (tmp_path / "tokens.css").write_text("")
    (tmp_path / "screens").mkdir()
    (tmp_path / "screens" / "listen.js").write_text("export function mount() {}")
    (tmp_path / "screens" / "listen.css").write_text("")
    assert app_files(tmp_path) == {"screens": ["listen"], "styles": ["listen"], "modules": ["store.js"],
                                   "css": ["tokens.css"]}
    app = FastAPI()
    mount_app(app, tmp_path)
    r = TestClient(app).get("/app/_files.json")
    assert r.status_code == 200 and r.json()["screens"] == ["listen"] and r.headers["cache-control"] == "no-cache"


def test_registry_and_worker_contract_in_source():
    js = (APP_DIR / "app.js").read_text()
    assert "import(`./screens/${id}.js`)" in js and 'fetch("_files.json"' in js
    for screen in ("welcome", "setup", "people", "listen", "about", "phone", "diag"):
        assert f'id: "{screen}"' in js, screen
    sw = (APP_DIR / "sw.js").read_text()
    assert "skipWaiting" in sw and "clients.claim" in sw and "const VERSION" in sw
    assert "startsWith(SCOPE)" in sw  # /ws, /health, /console/*, /sdk/* pass through untouched


# ---------- Chromium (slow) ----------
def _serve(app_dir: Path) -> tuple[str, subprocess.Popen]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "HEARING_DEVICE": "cpu", "HEARING_ASR": "mock", "HEARING_TTS": "mock",
           "HEARING_VOICE_MATCH": "mock", "HEARING_FACES": "mock", "HEARING_HQ_BACKEND": "local",
           "HEARING_APP_DIR": str(app_dir), "PYTHONPATH": f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
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


FAKE_SCREEN = """export const title = "People";
export function mount(root, ctx) {
  root.innerHTML = '<h1 id="fake-people">people mounted</h1>';
  window.fakeMounted = (window.fakeMounted || 0) + 1;
  ctx.signal.addEventListener("abort", () => { window.fakeAborted = true; });
  return () => { window.fakeCleaned = true; };
}
"""


@pytest.mark.slow  # starts a server and Chromium; no weights, no network
def test_shell_in_chromium_registry_placeholder_manifest_and_sw(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    app_dir = tmp_path / "app"
    shutil.copytree(APP_DIR, app_dir)
    shutil.rmtree(app_dir / "screens", ignore_errors=True)
    (app_dir / "screens").mkdir()
    (app_dir / "screens" / "people.js").write_text(FAKE_SCREEN)
    (app_dir / "screens" / "people.css").write_text("#fake-people { color: rgb(1, 2, 3); }")
    (app_dir / "screens" / "about.js").write_text("export function mount() { throw new Error('boom'); }")
    base, proc = _serve(app_dir)
    try:
        with sync_api.sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_context().new_page()
            page.set_default_timeout(15000)
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            failed: list[str] = []
            page.on("response", lambda r: failed.append(r.url) if r.status >= 400 else None)

            page.goto(base + "/app/")
            page.wait_for_function("window.hearingApp && window.hearingApp.ready")
            assert page.locator("#welcome-title").is_visible()  # built-in Welcome
            assert page.locator("#start").is_visible()

            page.evaluate("location.hash = 'listen'")  # no module: placeholder, no 404 fetched
            page.wait_for_selector('[data-placeholder="listen"]')
            assert "screens/listen.js" in page.locator('[data-placeholder="listen"]').inner_text()

            page.evaluate("location.hash = 'people'")  # a module: lazy-imported, its css linked
            page.wait_for_selector("#fake-people")
            assert page.evaluate("getComputedStyle(document.getElementById('fake-people')).color") == "rgb(1, 2, 3)"
            assert page.evaluate("document.title").startswith("People")

            page.evaluate("location.hash = 'about'")  # a module that throws: placeholder with the reason
            page.wait_for_selector('[data-placeholder="about"]')
            assert "boom" in page.locator('[data-placeholder="about"]').inner_text()
            assert page.evaluate("window.fakeCleaned && window.fakeAborted") is True

            page.evaluate("location.hash = 'diag'")  # the sheet opens over the current screen
            page.wait_for_selector('#sheet:not([hidden]) [data-placeholder="diag"]')
            page.locator(".sheet-close").click()
            page.wait_for_selector("#sheet[hidden]", state="attached")

            # hold-to-proceed Start: a short tap does nothing, a hold goes to Setup
            page.goto(base + "/app/#welcome")
            page.wait_for_function("window.hearingApp.ready")
            page.locator("#start").click()
            assert page.evaluate("location.hash") == "#welcome"
            box = page.locator("#start").bounding_box()
            page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            page.mouse.down()
            page.wait_for_function("location.hash === '#setup'")
            page.mouse.up()
            page.wait_for_selector('[data-placeholder="setup"]')

            # PWA: SW controls the page after a reload; Chromium parses the manifest without errors
            page.evaluate("navigator.serviceWorker.ready")
            page.reload()
            page.wait_for_function("window.hearingApp && window.hearingApp.ready")
            page.wait_for_function("navigator.serviceWorker.controller !== null")
            assert page.evaluate("navigator.serviceWorker.controller.scriptURL").endswith("/app/sw.js")
            cdp = page.context.new_cdp_session(page)
            manifest = cdp.send("Page.getAppManifest")
            assert manifest["errors"] == [], manifest["errors"]
            parsed = manifest.get("manifest")  # Chromium's own parse (newer CDP)
            if parsed:
                assert parsed["display"] == "kStandalone" and parsed["scope"] == base + "/app/"
                assert parsed["startUrl"] == base + "/app/" and len(parsed["icons"]) == 3
            installability = cdp.send("Page.getInstallabilityErrors")["installabilityErrors"]
            assert installability == [], installability
            for src in ("icons/icon-192.png", "icons/icon-512.png", "icons/maskable-512.png", "icons/apple-touch-icon.png"):
                assert page.evaluate(f"fetch('{src}').then(r => r.status)") == 200, src
            assert page.evaluate("fetch('/health').then(r => r.json()).then(j => j.ok)") is True  # passes through the SW

            assert not failed, failed
            assert not errors, errors
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)
