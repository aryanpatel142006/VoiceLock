"""``/connect`` (QR code + URLs for phones), ``/hearing-ca.crt`` (the certificate to trust), ``/sdk/``.

``/sdk/hearing-client.js`` imports ``../console/protocol.js``, so the console directory is also
mounted at ``/console/``: the same relative import works from the repo and over http.

``hearing serve`` puts a :class:`hearing.server.tls.ConnectInfo` in ``app.state.connect`` and the
served certificate's path in ``app.state.cert_path``; without them (tests, ``uvicorn`` by hand)
``/connect`` shows the URL the page itself was loaded from.
"""

from __future__ import annotations

import html
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from hearing.server.tls import ConnectInfo, qr_svg

SDK_DIR = Path(__file__).resolve().parents[2] / "web" / "sdk"

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Connect a phone · hearing</title>
<style>
body{{font:16px/1.45 system-ui,sans-serif;max-width:34rem;margin:1.5rem auto;padding:0 1rem;color:#111}}
.qr svg{{width:min(80vw,320px);height:auto;background:#fff}}
a{{word-break:break-all}} li{{margin:.3rem 0}} .warn{{background:#fff4d6;padding:.6rem;border-radius:6px}}
</style></head><body>
<h1>Connect a phone</h1>
<p>Same Wi-Fi as this laptop. Scan, then press <b>Start</b> on the page.</p>
<div class="qr">{qr}</div>
<ul>{links}</ul>
{trust}
<p>Wire contract: <code>docs/CONTRACTS.md#events</code>. SDK: <a href="/sdk/hearing-client.js">/sdk/hearing-client.js</a>.</p>
</body></html>
"""

_TRUST = """<div class="warn"><p><b>One-time certificate trust</b> (self-signed):</p>
<ol><li><a href="/hearing-ca.crt">Download the certificate</a>.</li>
<li><b>iPhone</b>: Settings → Profile Downloaded → Install; then Settings → General → About →
Certificate Trust Settings → turn on <i>hearing.local</i>.</li>
<li><b>Android</b>: Settings → Security → Encryption &amp; credentials → Install a certificate →
CA certificate → pick the file. (Or tap <i>Advanced → Proceed</i> on Chrome's warning.)</li></ol></div>"""

_PLAIN = """<p class="warn">Served over plain http: phones cannot open the microphone.
Restart with <code>hearing serve --tls --host 0.0.0.0</code>.</p>"""


def mount_connect(app: FastAPI, console_dir: Path) -> None:
    """Register the routes; call before ``mount_console`` (it owns ``/``)."""

    @app.get("/connect", response_class=HTMLResponse, include_in_schema=False)
    def connect(request: Request) -> HTMLResponse:
        info: ConnectInfo | None = getattr(request.app.state, "connect", None)
        if info is None:
            tls = request.url.scheme == "https"
            info = ConnectInfo([f"{request.url.scheme}://{request.url.netloc}/simple"], tls)
        target = request.query_params.get("page")
        if target is not None:
            if (not target.startswith("/") or target.startswith("//") or "\\" in target
                    or any(ord(c) < 32 for c in target)):
                raise HTTPException(400, "page must be an origin-relative path")
            page_url = urlsplit(target)
            info = ConnectInfo([
                urlunsplit((*urlsplit(u)[:2], page_url.path, page_url.query, page_url.fragment))
                for u in info.urls
            ], info.tls)
        links = "".join(f'<li><a href="{html.escape(u)}">{html.escape(u)}</a></li>' for u in info.urls)
        page = _PAGE.format(qr=qr_svg(info.primary), links=links, trust=_TRUST if info.tls else _PLAIN)
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/hearing-ca.crt", include_in_schema=False)
    def certificate(request: Request) -> FileResponse:
        path = getattr(request.app.state, "cert_path", None)
        if not path or not Path(path).is_file():
            raise HTTPException(404, "not serving TLS")
        # DER-less PEM is fine for both iOS (profile install) and Android (CA certificate picker).
        return FileResponse(path, media_type="application/x-x509-ca-cert", filename="hearing-ca.crt")

    if SDK_DIR.is_dir():
        app.mount("/sdk", StaticFiles(directory=SDK_DIR), name="sdk")
    if Path(console_dir).is_dir():
        app.mount("/console", StaticFiles(directory=console_dir), name="console-files")
