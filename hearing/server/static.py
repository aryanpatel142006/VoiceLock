"""Serving the debug console (``web/console/``) at ``/``, the simple test page at ``/simple`` and the app at ``/app/``.

Console and app assets go out with ``Cache-Control: no-cache``: the browser revalidates every
load (cheap 304s via ETag) instead of heuristically caching ``app.js`` and running a
stale console after an edit. If the directory has no ``index.html`` the root explains
that instead of returning 404.

The app (``web/app/``, M14) is a PWA scoped to ``/app/``. ``GET /app/_files.json`` lists the
optional modules present on disk (``screens/*.js``, their ``.css``, ``tokens.css``, ``store.js`` ...)
so the shell imports only what exists: a screen that has not landed shows a placeholder without a
404 in the browser console. ``HEARING_APP_DIR`` points the mount at another directory (tests).
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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


class AppFiles(ConsoleFiles):
    """``web/app/``: no-cache like the console, and the service worker may control all of ``/app/``."""

    async def get_response(self, path: str, scope: Scope):
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

    ``/simple`` and ``/judge`` serve ``simple.html`` / ``judge.html`` from the same directory, so their
    relative imports (``simple.js``, ``judge.js``, ``protocol.js``, ``audio-worklet.js``) resolve to the
    console's files. The app (``web/app/``, or ``HEARING_APP_DIR``) goes at ``/app/`` first.
    """
    mount_app(app, Path(app_dir or os.environ.get("HEARING_APP_DIR") or APP_DIR))
    for name in ("simple", "judge"):  # /judge: the judge demo page (T-088)
        page = console_dir / f"{name}.html"
        if page.is_file():
            app.add_api_route(f"/{name}", _page(page), include_in_schema=False)

    if (console_dir / "index.html").is_file():
        app.mount("/", ConsoleFiles(directory=console_dir, html=True), name="console")
        return

    @app.get("/", response_class=HTMLResponse)
    def no_console() -> str:
        return _NO_CONSOLE.format(path=console_dir)
