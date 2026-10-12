"""Serving the debug console (``web/console/``) at ``/``, the simple test page at ``/simple`` and the app at ``/app/``.

Console and app assets go out with ``Cache-Control: no-cache``: the browser revalidates every
load (cheap 304s via ETag) instead of heuristically caching ``app.js`` and running a
stale console after an edit. If the directory has no ``index.html`` the root explains
that instead of returning 404.

The app (``web/app/``, M14) is a PWA scoped to ``/app/``. ``GET /app/_files.json`` lists the
optional modules present on disk (``screens/*.js``, their ``.css``, ``tokens.css``, ``store.js`` ...)
so the shell imports only what exists: a screen that has not landed shows a placeholder without a
404 in the browser console. ``HEARING_APP_DIR`` points the mount at another directory (tests).

Build stamp (T-452): ``sw.js`` and ``index.html`` are served with :func:`build_stamp` injected, so a
new build (git rev, or any file under ``web/app/`` changing) gives the service worker a new
``VERSION`` (old caches are dropped on activate) and the shell's entry module a new ``?v=`` URL.
Safari otherwise kept serving modules from the old cache after a merge.

Companion: ``/app/?watch=...`` (the phone page) links ``companion.webmanifest`` instead of the app's manifest, so
Add to Home Screen installs the companion (start_url ``/app/?watch=latest``) even before its JS swaps the link
(``web/app/install.js``).
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import Headers
from starlette.types import Scope

APP_DIR = Path(__file__).resolve().parents[2] / "web" / "app"

_NO_CONSOLE = """<!doctype html><title>hearing</title>
<p>The debug console is not built yet: expected <code>{path}/index.html</code>.
The session server is up: <a href="/health">/health</a>, WebSocket <code>/ws/session</code>.</p>
"""


class ConsoleFiles(StaticFiles):
    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


@lru_cache(maxsize=8)
def _git_rev(directory: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(directory), "rev-parse", "--short=10", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    rev = out.stdout.strip()
    return rev if re.fullmatch(r"[0-9a-f]{4,40}", rev) else None


def build_stamp(app_dir: Path) -> str:
    """``<git rev>-<hash>`` of the app as it is on disk now (``<hash>`` alone outside git).

    The hash covers every file's path, size and mtime (``tests/`` and dot-files aside), so an
    uncommitted edit or a merge under a running server changes the stamp too.
    """
    app_dir = Path(app_dir)
    digest = hashlib.sha1()
    for p in sorted(app_dir.rglob("*")):
        rel = p.relative_to(app_dir)
        if rel.parts[0] == "tests" or any(part.startswith(".") for part in rel.parts) or not p.is_file():
            continue
        st = p.stat()
        digest.update(f"{rel.as_posix()}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    rev = _git_rev(app_dir.resolve())
    return f"{rev}-{digest.hexdigest()[:8]}" if rev else digest.hexdigest()[:8]


_SW_BUILD = re.compile(r'const BUILD = "[^"]*";')
_ENTRY = re.compile(r'(<script type="module" src="app\.js)(?:\?v=[^"]*)?(")')


def stamp_sw(text: str, stamp: str) -> str:
    return _SW_BUILD.sub(f'const BUILD = "{stamp}";', text, count=1)


def stamp_index(text: str, stamp: str) -> str:
    return _ENTRY.sub(rf"\g<1>?v={stamp}\g<2>", text, count=1)


_MANIFEST_LINK = re.compile(r'(<link rel="manifest" href=")manifest\.webmanifest(")')


def companion_index(text: str) -> str:
    """The shell for a ``?watch=`` load: its manifest is the companion's."""
    return _MANIFEST_LINK.sub(r"\g<1>companion.webmanifest\g<2>", text, count=1)


def _is_companion(scope: Scope) -> bool:
    return "watch" in parse_qs(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)


# the StaticFiles path (``.`` is ``/app/``) -> (file, how to stamp it, media type)
_STAMPED = {"sw.js": ("sw.js", stamp_sw, "text/javascript; charset=utf-8"),
            ".": ("index.html", stamp_index, "text/html; charset=utf-8"),
            "index.html": ("index.html", stamp_index, "text/html; charset=utf-8")}


class AppFiles(ConsoleFiles):
    """``web/app/``: no-cache like the console, the build stamp in ``sw.js`` / ``index.html``, and the
    service worker may control all of ``/app/``."""

    def _stamped(self, path: str, scope: Scope) -> Response | None:
        name, stamp, media_type = _STAMPED[path]
        file = Path(str(self.directory)) / name
        if not file.is_file():
            return None
        text = stamp(file.read_text(encoding="utf-8"), build_stamp(file.parent))
        if name == "index.html" and _is_companion(scope):
            text = companion_index(text)
        body = text.encode()
        # the ETag follows the stamped body, not the file: sw.js must change when another file does
        etag = f'"{hashlib.sha1(body).hexdigest()[:16]}"'
        headers = {"ETag": etag, "Cache-Control": "no-cache"}
        if path == "sw.js":
            headers["Service-Worker-Allowed"] = "/app/"
        if etag in Headers(scope=scope).get("if-none-match", ""):
            return Response(status_code=304, headers=headers)
        return Response(body, media_type=media_type, headers=headers)

    async def get_response(self, path: str, scope: Scope):
        if path in _STAMPED and scope["method"] in ("GET", "HEAD"):
            response = self._stamped(path, scope)
            if response is not None:
                return response
        response = await super().get_response(path, scope)
        if path == "sw.js":
            response.headers["Service-Worker-Allowed"] = "/app/"
        return response


def _page(path: Path):
    def page() -> FileResponse:
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    return page


def app_files(app_dir: Path) -> dict[str, list[str]]:
    """The optional modules of the app shell that exist on disk, by stem (``screens/listen.js`` -> ``listen``)."""
    screens = app_dir / "screens"
    return {
        "screens": sorted(p.stem for p in screens.glob("*.js")) if screens.is_dir() else [],
        "styles": sorted(p.stem for p in screens.glob("*.css")) if screens.is_dir() else [],
        "modules": sorted(p.name for p in app_dir.glob("*.js") if p.name not in ("app.js", "sw.js")),
        "css": sorted(p.name for p in app_dir.glob("*.css") if p.name != "app.css"),
    }


def mount_app(app: FastAPI, app_dir: Path) -> None:
    """Mount ``web/app/`` at ``/app/`` (``/app`` redirects there); call before the ``/`` console mount."""
    if not (app_dir / "index.html").is_file():
        return

    @app.get("/app", include_in_schema=False)
    def app_redirect() -> RedirectResponse:
        return RedirectResponse("/app/", status_code=308)

    @app.get("/app/_files.json", include_in_schema=False)
    def files() -> JSONResponse:
        return JSONResponse(app_files(app_dir), headers={"Cache-Control": "no-cache"})

    app.mount("/app", AppFiles(directory=app_dir, html=True), name="app")


def mount_console(app: FastAPI, console_dir: Path, app_dir: Path | None = None) -> None:
    """Mount the console at ``/``; call after every other route so they take precedence.

    ``/simple``, ``/judge`` and ``/who`` serve ``simple.html`` / ``judge.html`` / ``who.html`` from the same directory, so their
    relative imports (``simple.js``, ``judge.js``, ``protocol.js``, ``audio-worklet.js``) resolve to the
    console's files. The app (``web/app/``, or ``HEARING_APP_DIR``) goes at ``/app/`` first.
    """
    mount_app(app, Path(app_dir or os.environ.get("HEARING_APP_DIR") or APP_DIR))
    for name in ("simple", "judge", "who"):  # /judge: the judge demo page (T-088); /who: Who's speaking (hearing.who)
        page = console_dir / f"{name}.html"
        if page.is_file():
            app.add_api_route(f"/{name}", _page(page), include_in_schema=False)

    if (console_dir / "index.html").is_file():
        app.mount("/", ConsoleFiles(directory=console_dir, html=True), name="console")
        return

    @app.get("/", response_class=HTMLResponse)
    def no_console() -> str:
        return _NO_CONSOLE.format(path=console_dir)
