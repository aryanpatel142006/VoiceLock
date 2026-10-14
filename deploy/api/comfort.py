"""Vercel function for the web demo: the real Comfort routes (hearing/server/comfort.py) with Gemini and ElevenLabs.

The demo page (deploy/demo/shim.js) sends every /comfort/* call here as /api/comfort?path=/comfort/... with the
visitor's saved profiles in the `x-voicelock-profiles` header (base64 JSON, kept in their own browser). Each request
starts from the fictional demo seeds plus those profiles, so visitors never see each other's data and nothing is
kept on the server. Only comfort.py and the two small modules it imports are loaded: no PyTorch, no audio engine.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
import os
import re
import sys
import time
import types
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HEARING_COMFORT", "1")
os.environ.setdefault("HEARING_COMFORT_DIR", "/tmp/voicelock-comfort")

# The package __init__ files import the whole engine; register bare packages so only the leaf modules load.
for _name in ("hearing", "hearing.server", "hearing.synth", "hearing.audio"):
    _pkg = types.ModuleType(_name)
    _pkg.__path__ = [str(ROOT / _name.replace(".", "/"))]
    sys.modules.setdefault(_name, _pkg)
comfort = importlib.import_module("hearing.server.comfort")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
comfort.mount_comfort(app)

STORE = Path(os.environ["HEARING_COMFORT_DIR"]) / "profiles.json"
MAX_HEADER = 64_000
MAX_BODY = 6_000_000  # a few seconds of hold-to-talk audio as base64
LIMIT_PER_MIN = int(os.environ.get("DEMO_CHAT_PER_MIN", "12"))  # chat turns per visitor IP per minute, per instance
_lock = asyncio.Lock()
_recent: dict[str, deque] = defaultdict(deque)


def _visitor_profiles(header: str | None) -> dict:
    if not header or len(header) > MAX_HEADER:
        return {}
    try:
        data = json.loads(base64.b64decode(header))
    except (ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if isinstance(k, str) and re.fullmatch(comfort.PROFILE_ID, k) and isinstance(v, dict)}


def _seed_store(visitor: dict) -> None:
    profiles = {s["id"]: comfort.normalize_profile(s["id"], s) for s in comfort.DEMO_PROFILES}
    for pid, p in visitor.items():
        try:
            profiles[pid] = comfort.normalize_profile(pid, p)
        except Exception:  # a malformed stored profile falls back to the seed (or is dropped)
            continue
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(profiles))


def _rate_limited(ip: str) -> bool:
    now = time.time()
    q = _recent[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= LIMIT_PER_MIN:
        return True
    q.append(now)
    return False


@app.middleware("http")
async def per_visitor(request: Request, call_next):
    path = request.query_params.get("path", request.url.path)
    if not path.startswith("/comfort/"):
        return JSONResponse({"detail": "not found"}, status_code=404)
    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        return JSONResponse({"detail": "request too large"}, status_code=413)
    if path == "/comfort/chat" and request.method == "POST":
        ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0].strip()
        if _rate_limited(ip):
            return JSONResponse({"detail": "Too many messages. Wait a minute and try again."}, status_code=429)
    request.scope["path"] = path
    request.scope["raw_path"] = path.encode()
    request.scope["query_string"] = b""
    request.scope["root_path"] = ""
    async with _lock:  # one visitor's profiles in the store at a time
        _seed_store(_visitor_profiles(request.headers.get("x-voicelock-profiles")))
        return await call_next(request)
