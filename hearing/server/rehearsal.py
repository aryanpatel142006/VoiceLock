"""Rehearsal testbed API (T-285): live A/B matrix, dialogue scripts and the take recorder.

``mount(app)`` adds ``/api/rehearsal/*``; the page is ``web/console/rehearsal.html`` (served by the
console mount). Audio is captured by the browser from the phone microphone and uploaded here as raw
PCM16 mono chunks, so a take folder is written even when the engine is busy:

    results/rehearsal/<UTC stamp>_<mode>_<scenario>/
        mic.wav  out.wav  stem_<n>.wav   (mono PCM16 at the rate the page declared, 16 kHz except on iOS)
        events.jsonl                      (one telemetry / control / marker event per line)
        take.json                         (matrix selection, start/stop, per-stream sample counts)

The matrix is applied live, with no restart: the ambiguity policy sets ``HEARING_GATED_PLAY`` (read
on every enrolled hop), the ladder and suppression go to the session as ordinary ``control``
messages from the page. The representation choice is recorded only (the engine has no switch yet).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from hearing.tse.enrolled import GATED_PLAY_ENV

RESULTS_ENV = "HEARING_REHEARSAL_DIR"
SAMPLE_RATE = 16000
MAX_CHUNK_BYTES = 4_000_000  # ~2 min of PCM16 at 16 kHz; one upload is a few hundred ms

REPRESENTATIONS = {
    "ecapa": {"label": "Stock ECAPA", "applied": True},
    "projection": {"label": "Team projection head", "applied": False, "weights": "models/team_projection_head.pt"},
    "mahalanobis": {"label": "Mahalanobis metric", "applied": False},
}
POLICIES = {
    "stream": {"label": "Best-guess stream", "gated_play": "stream", "ab": "processed"},
    "duck": {"label": "Duck to room baseline", "gated_play": "", "ab": "processed"},
    "passthrough": {"label": "Pass-through mix", "gated_play": "", "ab": "raw"},
}
# Window/hop in seconds are the engine's presets (hearing.latency_preset); the requested 0.5/0.05,
# 1.0/0.1, 2.0/0.25 ladder is not a preset the engine has, so the page shows what really runs.
LADDER = {
    "ultra": {"label": "Ultra-low latency", "preset": "speed", "window_s": 0.5, "hop_s": 0.25},
    "balanced": {"label": "Balanced", "preset": "balanced", "window_s": 1.0, "hop_s": 0.5},
    "accurate": {"label": "High accuracy", "preset": "accuracy", "window_s": 1.5, "hop_s": 0.75},
}
DEFAULT_MATRIX = {"representation": "ecapa", "policy": "stream", "ladder": "balanced", "suppression": 0.8}

SCRIPTS = [
    {"id": "alt-1", "kind": "alternating", "title": "Alternating turns", "cards": [
        {"who": "A", "line": "Did you manage to book the table for tonight?", "seconds": 4},
        {"who": "B", "line": "Yes, seven thirty at the place by the fountain.", "seconds": 4},
        {"who": "A", "line": "Perfect. Should we bring the umbrella just in case?", "seconds": 4},
        {"who": "B", "line": "Definitely. The forecast says rain after eight.", "seconds": 4}]},
    {"id": "overlap-1", "kind": "overlap", "title": "Simultaneous overlap", "cards": [
        {"who": "A", "line": "Read steadily: the quick brown fox jumps over the lazy dog, again and again.",
         "seconds": 8, "overlap_with": "B"},
        {"who": "B", "line": "Read steadily: pack my box with five dozen liquor jugs, again and again.",
         "seconds": 8, "overlap_with": "A"}]},
    {"id": "interrupt-1", "kind": "interruption", "title": "Rapid interruption", "cards": [
        {"who": "A", "line": "So what I wanted to say about the schedule is that we really need to", "seconds": 4},
        {"who": "B", "line": "Sorry to cut in: the venue just called about the time.", "seconds": 3,
         "interrupts": True},
        {"who": "A", "line": "Right, go on, what did they say?", "seconds": 3},
        {"who": "B", "line": "They moved us to six, so we should leave now.", "seconds": 3}]},
]
FREEFORM = {"id": "freeform", "kind": "freeform", "title": "Freeform: say whatever",
            "cards": [{"who": "ANY", "line": "Talk naturally. Mark moments with the Marker button.", "seconds": 0}]}


def safe_slug(text: str, default: str = "x") -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", str(text or "")).strip("-")[:40]
    return slug or default


def validate_matrix(raw: dict | None) -> dict:
    """The default matrix with the keys in ``raw`` applied; ``ValueError`` on an unknown choice."""
    out = dict(DEFAULT_MATRIX)
    for key, table in (("representation", REPRESENTATIONS), ("policy", POLICIES), ("ladder", LADDER)):
        if raw and key in raw:
            if raw[key] not in table:
                raise ValueError(f"unknown {key} {raw[key]!r}; expected one of {sorted(table)}")
            out[key] = raw[key]
    if raw and "suppression" in raw:
        value = raw["suppression"]
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
            raise ValueError("suppression must be a number in [0, 1]")
        out["suppression"] = float(value)
    return out


def matrix_controls(matrix: dict) -> dict:
    """The ``control`` fields (docs/CONTRACTS.md) the page sends to the session for this matrix."""
    return {"latency_preset": LADDER[matrix["ladder"]]["preset"], "ab": POLICIES[matrix["policy"]]["ab"],
            "suppression": round(matrix["suppression"] * 100)}


def apply_policy(policy: str) -> str:
    """Set the process-wide gated-play switch for ``policy``; returns the value now in force."""
    value = POLICIES[policy]["gated_play"]
    if value:
        os.environ[GATED_PLAY_ENV] = value
    else:
        os.environ.pop(GATED_PLAY_ENV, None)
    return value


class Take:
    """One recording folder. Streams are appended to ``<name>.pcm`` and wrapped as WAV on stop."""

    def __init__(self, root: Path, mode: str, scenario: str, matrix: dict, clock=time.time) -> None:
        stamp = datetime.fromtimestamp(clock(), UTC).strftime("%Y%m%dT%H%M%SZ")
        base = f"{stamp}_{safe_slug(mode, 'mode')}_{safe_slug(scenario, 'scenario')}"
        self.dir = root / base
        n = 1
        while self.dir.exists():  # two takes in one second
            n += 1
            self.dir = root / f"{base}-{n}"
        self.id = self.dir.name
        self.dir.mkdir(parents=True)
        self.matrix, self.mode, self.scenario = matrix, mode, scenario
        self.started = clock()
        self.stopped: float | None = None
        self.samples: dict[str, int] = {}
        self.rates: dict[str, int] = {}
        self.events = 0
        self.closed = False
        self.lock = threading.Lock()
        (self.dir / "events.jsonl").touch()
        self._save_metadata()

    def _save_metadata(self) -> None:
        temporary = self.dir / "take.json.tmp"
        temporary.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        temporary.replace(self.dir / "take.json")

    def add_audio(self, stream: str, data: bytes, rate: int = SAMPLE_RATE) -> int:
        stream = safe_slug(stream, "stream")
        if not 8000 <= rate <= 96000:
            raise ValueError("rate must be 8000-96000 Hz")
        data = data[: len(data) - len(data) % 2]
        with self.lock:
            if self.closed:
                raise ValueError("take is closed")
            with (self.dir / f"{stream}.pcm").open("ab") as f:
                f.write(data)
            self.samples[stream] = self.samples.get(stream, 0) + len(data) // 2
            self.rates.setdefault(stream, rate)
            self._save_metadata()
            return self.samples[stream]

    def add_events(self, events: list) -> int:
        with self.lock:
            if self.closed:
                raise ValueError("take is closed")
            with (self.dir / "events.jsonl").open("a", encoding="utf-8") as output:
                for event in events:
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")
            self.events += len(events)
            self._save_metadata()
            return self.events

    def stop(self, clock=time.time) -> dict:
        with self.lock:
            if self.closed:
                return self.summary()
            self.closed = True
            for pcm in sorted(self.dir.glob("*.pcm")):
                with wave.open(str(pcm.with_suffix(".wav")), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self.rates.get(pcm.stem, SAMPLE_RATE))
                    w.writeframes(pcm.read_bytes())
                pcm.unlink()
            self.stopped = clock()
            summary = self.summary()
            self._save_metadata()
            return summary

    def summary(self) -> dict:
        return {"id": self.id, "dir": str(self.dir), "mode": self.mode, "scenario": self.scenario,
                "matrix": self.matrix, "started": self.started, "stopped": self.stopped,
                "sample_rates": dict(self.rates), "samples": dict(self.samples), "events": self.events}


def mount(app: FastAPI, results_root: Path | None = None) -> None:
    """Add ``/api/rehearsal/*``. State is per app: the live matrix and the open takes."""
    root = Path(results_root or os.environ.get(RESULTS_ENV) or "results/rehearsal")
    state = {"matrix": dict(DEFAULT_MATRIX)}
    takes: dict[str, Take] = {}

    def take_or_404(take_id: str) -> Take:
        take = takes.get(take_id)
        if take is None:
            raise HTTPException(404, "unknown take")
        return take

    async def json_body(request: Request):
        try:
            return await request.json()
        except ValueError as exc:
            raise HTTPException(400, "body must be JSON") from exc

    @app.get("/api/rehearsal/options")
    def options() -> dict:
        return {"representations": REPRESENTATIONS, "policies": POLICIES, "ladder": LADDER,
                "default": DEFAULT_MATRIX, "scripts": [*SCRIPTS, FREEFORM],
                "projection_head_present": Path(REPRESENTATIONS["projection"]["weights"]).is_file()}

    @app.get("/api/rehearsal/config")
    def get_config() -> dict:
        # Activate the process-wide rehearsal policy only when a rehearsal client reads it.
        apply_policy(state["matrix"]["policy"])
        return {"matrix": state["matrix"], "controls": matrix_controls(state["matrix"]),
                "gated_play": os.environ.get(GATED_PLAY_ENV, "")}

    @app.post("/api/rehearsal/config")
    async def set_config(request: Request) -> dict:
        body = await json_body(request)
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        try:
            matrix = validate_matrix({**state["matrix"], **body})
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        state["matrix"] = matrix
        gated = apply_policy(matrix["policy"])
        return {"matrix": matrix, "controls": matrix_controls(matrix), "gated_play": gated,
                "representation_applied": REPRESENTATIONS[matrix["representation"]]["applied"]}

    @app.post("/api/rehearsal/take")
    async def start_take(request: Request) -> dict:
        body = await json_body(request)
        if not isinstance(body, dict):
            raise HTTPException(400, "body must be a JSON object")
        take = Take(root, str(body.get("mode") or "script"), str(body.get("scenario") or "freeform"),
                    dict(state["matrix"]))
        takes[take.id] = take
        return take.summary()

    @app.post("/api/rehearsal/take/{take_id}/audio")
    async def audio(take_id: str, request: Request, stream: str = "mic", rate: int = SAMPLE_RATE) -> dict:
        take = take_or_404(take_id)
        data = await request.body()
        if len(data) > MAX_CHUNK_BYTES:
            raise HTTPException(413, "chunk too large")
        try:
            return {"samples": take.add_audio(stream, data, rate)}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/rehearsal/take/{take_id}/events")
    async def events(take_id: str, request: Request) -> dict:
        take = take_or_404(take_id)
        body = await json_body(request)
        if not isinstance(body, list) or not all(isinstance(e, dict) for e in body):
            raise HTTPException(400, "body must be a JSON array of objects")
        try:
            return {"events": take.add_events(body)}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/rehearsal/take/{take_id}/stop")
    def stop_take(take_id: str) -> dict:
        take = take_or_404(take_id)
        summary = take.stop()
        takes.pop(take_id, None)
        return summary
