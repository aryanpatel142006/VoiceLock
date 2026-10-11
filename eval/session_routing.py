"""Wrong-speaker routing of a recorded session against a T-141 team scene (T-180).

``pipeline_eval --profile --team`` reads its own scene layout, not the T-141 ``demo_scene`` dirs
(``stems/{target,other,noise}.wav``), so ``demo_regress`` rows had no wrong-speaker number (T-178).
This scores a session dir written with ``HEARING_SESSION_LOG`` (``out_pre_agc.wav`` when present,
else ``out.wav``) with :func:`eval.pipeline_eval.routing`: 0.25 s windows, the target is intended
throughout, a target-active window is *wrong* when the other talker dominates the aligned output.
Scored from the first telemetry tick whose ladder plays ``enrolled`` (the tier the knobs change);
``--from-start`` scores the whole run.

    python -m eval.session_routing <session dir> <scene dir> [--from-start]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eval import namecall_eval as nc
from eval import pipeline_eval as pe
from hearing.audio import io as aio

SR = pe.SR


def _read(path: Path) -> np.ndarray:
    return aio.load(path, sr=SR)[0]


def enrolled_from(events: Path) -> float | None:
    """Session time (s) of the first telemetry tick that plays the enrolled tier; None if it never did."""
    if not events.exists():
        return None
    t0 = None
    for line in events.open():
        rec = json.loads(line)
        ev = rec.get("event") or {}
        if rec.get("dir") != "out" or ev.get("type") != "telemetry":
            continue
        t0 = rec["t"] if t0 is None else t0
        if (ev.get("tiers") or {}).get("active") == "enrolled":
            return float(rec["t"]) - t0
    return None


def score(session: Path, scene: Path, from_start: bool = False) -> dict:
    man = json.loads((scene / "manifest.json").read_text())
    stems = {k: _read(scene / v) for k, v in man["stems"].items() if k in ("target", "other", "noise")}
    wav = session / "out_pre_agc.wav"
    out = _read(wav if wav.exists() else session / "out.wav")
    n = len(stems["target"])
    t_iso = None if from_start else enrolled_from(session / "events.jsonl")
    i0 = 0 if t_iso is None else min(round(t_iso * SR), max(0, n - 2 * SR))
    delay = nc.best_lag(out, stems["target"], i0)
    d = round(delay * SR)
    y = np.zeros(n, np.float32)
    seg = out[d : d + n]
    y[: len(seg)] = seg
    sc = {"stems": {"A": stems["target"], "B": stems["other"]}, "noise": stems["noise"], "intended": [(0.0, "A")]}
    r = pe.routing(sc, y, 0.0 if t_iso is None else t_iso)
    return {"session": str(session), "scene": man.get("name"), "enrolled_from_s": t_iso, "delay_s": delay,
            "scored": wav.name if wav.exists() else "out.wav", **r}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("scene", type=Path)
    ap.add_argument("--from-start", action="store_true")
    a = ap.parse_args(argv)
    print(json.dumps(score(a.session, a.scene, a.from_start)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
