"""Replay a recorded session's mic through :class:`hearing.tse.enroll.AutoEnroller` (T-119).

The recording keeps ``in.wav`` (every 10 ms hop the session saw) and ``events.jsonl`` (telemetry at
~10 Hz with the target id and every face's ``p_speaking``), but only every 15th camera frame, so the
vision side cannot be re-run. This script holds each telemetry tick's target and faces for the hops
until the next tick and feeds ``in.wav`` to a fresh enroller, then reports per tick the acquisition
hint the enroller would show, split by whether ASD said the target was speaking (``p_speaking >=
0.5``). ``--absolute`` runs the pre-T-119 gate (fixed -40 dBFS) for comparison.

    python -m eval.enroll_session_replay results/sessions/20261007T010121Z-1 [--absolute]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from hearing.tse import enroll
from hearing.types import FaceTrack, StageContext, TargetState

HOP = 160


def load(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, dict]]]:
    """(in.wav samples, audio-in arrival times, cumulative samples at each arrival, telemetry ticks)."""
    x, _sr = sf.read(session / "in.wav", dtype="float32")
    ts, cum, ticks, n = [], [], [], 0
    with open(session / "events.jsonl") as f:
        for line in f:
            r = json.loads(line)
            e = r["event"]
            if r["dir"] == "in" and e.get("type") == "binary" and e.get("kind") == 1:
                n += e["bytes"] // 2
                ts.append(r["t"])
                cum.append(n)
            elif r["dir"] == "out" and e.get("type") == "telemetry":
                ticks.append((r["t"], e))
    return x, np.asarray(ts), np.asarray(cum), ticks


def replay(session: Path, *, absolute: bool = False, skip: tuple[tuple[float, float], ...] = ()) -> dict:
    x, ts, cum, ticks = load(session)
    old_min = enroll.VOICED_MIN_DB
    if absolute:
        enroll.VOICED_MIN_DB = 20.0 * float(np.log10(enroll.VOICED_RMS))  # relative gate never relaxes
    try:
        en = enroll.AutoEnroller()
        ctx = StageContext.default()
        hop = 0
        counts: dict[str, Counter] = {"speaking": Counter(), "silent": Counter()}
        for t, e in ticks:
            end = int(np.interp(t, ts, cum)) // HOP if len(ts) else 0
            while hop < end and (hop + 1) * HOP <= len(x):
                en.update(x[hop * HOP:(hop + 1) * HOP], ctx)
                hop += 1
            faces = e.get("faces") or []
            tid = (e.get("target") or {}).get("track_id")
            ctx.tracks = [FaceTrack(int(f["track_id"]), tuple(f["box"]), None, float(f.get("p_speaking") or 0.0), t)
                          for f in faces]
            ctx.target = TargetState(tid, 1.0, False, 0.0)
            if tid is None or not len(ts) or t < ts[0] or any(a <= t < b for a, b in skip):
                continue
            p = next((f.p_speaking for f in ctx.tracks if f.track_id == tid), 0.0)
            counts["speaking" if p >= 0.5 else "silent"][en.hint or ("ready" if en.ready else "kept")] += 1
    finally:
        enroll.VOICED_MIN_DB = old_min
    return {k: dict(v) for k, v in counts.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("--absolute", action="store_true", help="the pre-T-119 fixed -40 dBFS gate")
    ap.add_argument("--skip", action="append", default=[], help="t0:t1 session seconds to leave out (a replay)")
    args = ap.parse_args(argv)
    skip = tuple(tuple(float(v) for v in s.split(":")) for s in args.skip)
    res = replay(args.session, absolute=args.absolute, skip=skip)
    for k, c in res.items():
        total = sum(c.values()) or 1
        quiet = c.get(enroll.HINTS["unvoiced"], 0)
        print(f"{k:9s} ticks {total:5d}  'too quiet' {quiet:4d} ({100 * quiet / total:.1f} %)  {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
