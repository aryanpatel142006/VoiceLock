"""Live-replay check of the Calibrate button (T-085).

Starts ``hearing serve`` in-process on an ephemeral port (default ``tiers`` ladder, real models),
streams the demo clip over ``/ws/session`` in real time (``hearing.replay.feed``, looped), sends
``control.enroll {action:"start", seconds:3}`` at ``--press`` s, and records:

* ``ready_s``: press -> first telemetry with ``enrollment.state == "calibrated"`` (wall clock);
* ``isolated_at_s`` (T-092): session start -> first telemetry whose ``acquisition`` says ``locked`` with an
  isolating tier (``enrolled``/``focus_hq``/``av``) playing; with ``--press -1`` nothing is pressed and
  this is the automatic enrollment's time-to-isolated; ``hints`` counts the acquisition pill's
  ``· hint`` suffixes seen before it; ``enrolled_at_s``: first ``enrollment.state`` ready/calibrated;
* wrong-target seconds per 10 s of the output after the clip is ready, against the fixture's
  target/interferer stems (the demo clip is the fixture looped to 12 s). Per 1 s span of the
  delay-aligned output, least squares on [target, interferer]; a span is ``wrong`` when the
  interferer part carries more energy than the target part, ``muted`` when the output is < -40 dB
  of the mix.

    .venv/bin/python -m eval.calibrate_replay --isolation standard strict --seconds 30
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path

import numpy as np

from hearing.audio.io import pcm16_to_float
from hearing.replay import Clip, feed
from hearing.server.protocol import unpack_frame

SR = 16000
ROOT = Path(__file__).resolve().parents[1]
GENERATED = Path(os.environ["HEARING_FIXTURES_DIR"]) if os.environ.get("HEARING_FIXTURES_DIR") else ROOT / "eval" / "fixtures" / "generated"
DEMO = GENERATED.parent / "demo"  # eval.make_fixtures --demo writes it next to the shared fixtures


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve():
    import uvicorn

    from hearing.server.app import create_app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 60
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.05)
    return server, f"ws://127.0.0.1:{port}/ws/session"


async def _session(url: str, clip: Clip, isolation: str, press: float, seconds: float, oracle=None) -> dict:
    import websockets

    out: dict[int, np.ndarray] = {}
    states: list[tuple[float, str, str | None]] = []
    isolated: list[float] = []
    hints: dict[str, int] = {}
    reasons: list[dict] = []
    async with websockets.connect(url, max_size=4 * 1024 * 1024) as ws:
        async def send(message):
            await ws.send(json.dumps(message) if isinstance(message, dict) else message)

        t0 = time.monotonic()
        pressed: list[float] = []

        async def receive():
            async for message in ws:
                if isinstance(message, bytes):
                    frame = unpack_frame(message)
                    if frame.kind == 3:
                        out[frame.seq] = pcm16_to_float(frame.payload)
                    continue
                event = json.loads(message)
                lock = oracle.on_message(event) if oracle is not None else None
                if lock is not None:  # --lock-target: the manifest's target face, as scripts/replay.py
                    await send(lock)
                if event.get("type") == "telemetry":
                    tiers = event.get("tiers") or {}
                    acq = event.get("acquisition") or {}
                    active = tiers.get("active") if isinstance(tiers, dict) else None
                    if not isolated and acq.get("state") == "locked" and active in {"enrolled", "focus_hq", "av"}:
                        isolated.append(time.monotonic() - t0)
                    if not isolated and " · " in acq.get("detail", ""):
                        h = acq["detail"].split(" · ", 1)[1]
                        hints[h] = hints.get(h, 0) + 1
                    states.append((time.monotonic() - t0, (event.get("enrollment") or {}).get("state"),
                                   tiers.get("active") if isinstance(tiers, dict) else None))
                    if isinstance(tiers, dict):
                        reasons.append({row["name"]: row["reason"] for row in tiers.get("tiers", [])}
                                       | {"same_bucket": tiers.get("same_bucket")})

        async def press_button():
            if press < 0:  # T-092: automatic enrollment only
                return
            await asyncio.sleep(press)
            pressed.append(time.monotonic() - t0)
            await send({"type": "control", "enroll": {"action": "start", "seconds": 3}})

        receiver = asyncio.create_task(receive())
        presser = asyncio.create_task(press_button())
        feeder = asyncio.create_task(feed(clip, send, loop=True,
                                          control={"mode": "focus_hq", "suppression": 100,
                                                   "isolation": isolation}))
        await asyncio.sleep(seconds)
        feeder.cancel()
        await asyncio.sleep(0.5)
        receiver.cancel()
        presser.cancel()
    ready = next((t for t, s, _ in states if s == "calibrated"), None)
    if press < 0:
        ready = isolated[0] if isolated else None
    hop = len(next(iter(out.values()))) if out else 0  # one reply per input message, same seq
    audio = np.concatenate([out.get(k, np.zeros(hop, np.float32)) for k in range(max(out) + 1)]) if out else np.zeros(0, np.float32)
    return {"isolation": isolation, "press_s": pressed[0] if pressed else None, "ready_at_s": ready,
            "ready_s": None if ready is None or not pressed else round(ready - pressed[0], 2),
            "isolated_at_s": round(isolated[0], 2) if isolated else None, "hints": hints,
            "enrolled_at_s": next((round(t, 2) for t, s, _ in states if s in ("ready", "calibrated")), None),
            "states": states, "audio": audio, "reasons": reasons[-1:] + reasons[len(reasons) // 2: len(reasons) // 2 + 1]}


def _score(audio: np.ndarray, mix: np.ndarray, tgt: np.ndarray, itf: np.ndarray, start_s: float) -> dict:
    """Wrong/muted 1 s spans of ``audio`` from ``start_s`` on, after aligning it to ``mix``."""
    n = min(len(audio), len(mix))
    a, m = audio[:n], mix[:n]
    best, lag = -1.0, 0
    for d in range(0, min(4 * SR, n // 2), 160):  # output lags input by the playout delay
        x, y = a[d:], m[: n - d]
        c = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
        if c > best:
            best, lag = c, d
    a = a[lag:]
    spans = {"ok": 0, "wrong": 0, "muted": 0}
    for i in range(int(start_s * SR), len(a) - SR + 1, SR):
        y = a[i: i + SR]
        basis = np.stack([tgt[i: i + SR], itf[i: i + SR]], 1)
        if 10 * np.log10((y ** 2).mean() + 1e-12) < 10 * np.log10((m[i: i + SR] ** 2).mean() + 1e-12) - 40:
            spans["muted"] += 1
            continue
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        e_t, e_i = (np.linalg.norm(basis * coef, axis=0) ** 2)
        spans["wrong" if e_i > e_t else "ok"] += 1
    total = sum(spans.values())
    return {"lag_s": lag / SR, "align_corr": round(best, 3), **spans,
            "wrong_per_10s": round(10 * spans["wrong"] / max(total, 1), 2),
            "acc": round(spans["ok"] / max(total, 1), 3)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", type=Path, default=None)
    p.add_argument("--isolation", nargs="+", default=["standard", "strict"])
    p.add_argument("--press", type=float, default=2.0, help="seconds; -1 = never (automatic enrollment, T-092)")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--lock-target", action="store_true",
                   help="lock the manifest's target face (v2 demo scenes) instead of letting TargetSelector choose")
    args = p.parse_args(argv)
    clip_path = args.clip or (DEMO / "grid-two-face.mp4")
    manifest = json.loads((clip_path.parent / "manifest.json").read_text())
    if "stems" in manifest:  # v2 demo scenes (T-097): stems next to the clip
        files = {"target": manifest["stems"]["target"], "interferer": manifest["stems"]["other"]}
        files = {k: clip_path.parent / f for k, f in files.items()}
    else:
        src = GENERATED / "av_full" / manifest["source_fixture"]
        files = {k: src / f"{k}.wav" for k in ("target", "interferer")}
    import soundfile as sf

    stems = {k: sf.read(f, dtype="float32")[0] for k, f in files.items()}
    clip = Clip.load(clip_path)
    reps = int(np.ceil(args.seconds * SR / len(stems["target"]))) + 2
    tgt, itf = (np.tile(stems[k], reps) for k in ("target", "interferer"))
    mix = np.tile(clip.audio[: int(clip.duration * SR)], int(np.ceil(args.seconds / clip.duration)) + 2)

    server, url = _serve()
    results = []
    for iso in args.isolation:
        oracle = None
        if args.lock_target:
            from scripts.replay import OracleLock

            oracle = OracleLock(manifest)
        r = asyncio.run(_session(url, clip, iso, args.press, args.seconds, oracle))
        start = (r["ready_at_s"] or args.seconds) + 0.0
        r["score_after_ready"] = _score(r.pop("audio"), mix, tgt, itf, start)
        r["tiers_after_ready"] = {}
        for t, _, tier in r["states"]:
            if r["ready_at_s"] is not None and t >= r["ready_at_s"] and tier:
                r["tiers_after_ready"][tier] = r["tiers_after_ready"].get(tier, 0) + 1
        r["states"] = [s for s in r["states"] if s[0] < (r["ready_at_s"] or 0) + 1][-12:]
        print(json.dumps({k: v for k, v in r.items() if k != "states"}), flush=True)
        results.append(r)
    server.should_exit = True
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
