"""Serve web/app (and web/console, which screens import) on an ephemeral loopback port, no engine, no microphone.

For browser walks of a screen's look and motion (T-429 used it for the Welcome cold open with the Playwright MCP).
Prints the port on the first line and keeps serving until killed.

Run from the repo root: .venv/bin/python web/app/tests/serve_app.py [--port 0]
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from hearing.server.static import mount_app

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=0, help="0 (default) picks a free port; it is printed first")
    args = ap.parse_args()
    app = FastAPI()
    mount_app(app, ROOT / "web" / "app")
    app.mount("/console", StaticFiles(directory=ROOT / "web" / "console"), name="console-files")
    sock = socket.socket()
    sock.bind(("127.0.0.1", args.port))
    print(sock.getsockname()[1], flush=True)
    sys.stdout.flush()
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])


if __name__ == "__main__":
    main()
