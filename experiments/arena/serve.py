"""Stdlib server for one Arena batch: the UI, ``batch.json``, the run's audio, and ``POST /verdict``.

Bind to port 0 when testing (several worktrees share the machine). ``key.json`` is never served.
"""
from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from experiments.arena.arena import record_verdict

UI = Path(__file__).parent / "ui" / "index.html"


def make_server(run_dir: Path, rung: int, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    run_dir = run_dir.resolve()
    batch_dir = run_dir / "arena" / f"rung{rung}"

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            if path == "/batch.json":
                return self._send(200, (batch_dir / "batch.json").read_bytes(), "application/json")
            target = (run_dir / path.lstrip("/")).resolve()
            if (path.startswith("/audio/") and target.is_relative_to(run_dir / "audio")
                    and target.suffix == ".wav" and target.is_file()):
                return self._send(200, target.read_bytes(), "audio/wav")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/verdict":
                return self._send(404, b"not found", "text/plain")
            size = min(int(self.headers.get("Content-Length", 0)), 4096)
            try:
                saved = record_verdict(batch_dir, json.loads(self.rfile.read(size) or b"{}"))
            except (ValueError, json.JSONDecodeError) as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            return self._send(200, json.dumps(saved).encode(), "application/json")

    return ThreadingHTTPServer((host, port), Handler)
