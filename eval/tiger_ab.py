"""Offline A/B of the blind separator behind the enrolled pick: MossFormer2 vs TIGER-speech (T-372).

    .venv/bin/python -m eval.tiger_ab --profiles $F/profiles --backends mossformer2 tiger tiger-large \\
        --scene joseph:aryan:$F/t141/joseph/demo-team-easy [--scene ...] [--device cpu]

Each backend replays ``eval.wrong_speaker_audit.run`` (real weights, saved profiles, 1.0 s windows every 0.5 s,
same ECAPA/LDA pick) with ``HEARING_SEPARATOR`` set. Per scene: wrong-speaker % (other-dominant seconds over
clip seconds, session_score rule), SI-SDRi of the played output over the mix, leak dB (other stem's
least-squares energy, output vs mix) and p50/p95 separator ms per 1.0 s window. Wall-clock numbers need
``swarm-lock --exclusive``; the accuracy columns do not. ``--device cpu`` hides MPS/CUDA from ``hearing.device``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from eval import metrics
from eval import wrong_speaker_audit as audit

SR = audit.SR


def leak_db(y, x, t, o, nz) -> float:
    """Other stem's least-squares energy in the output relative to the mix, over frames where it is present."""
    gy, gx = audit._ls_gains(y, t, o, nz), audit._ls_gains(x, t, o, nz)
    eo = np.sum(audit._frames(o) ** 2, axis=1)
    act = eo > 0.1 * np.median(eo[eo > 1e-9]) if (eo > 1e-9).any() else np.zeros(len(eo), bool)
    ey, ex = (gy[:, 1] ** 2 * eo)[act].sum(), (gx[:, 1] ** 2 * eo)[act].sum()
    return float(10 * np.log10(max(ey, 1e-12) / max(ex, 1e-12)))


def run_scene(backend: str, target: str, other: str, scene: Path, profiles: Path) -> dict:
    from eval.contrast_sweep import load_profile
    from hearing.tse import blind

    os.environ["HEARING_SEPARATOR"] = "" if backend == "mossformer2" else backend
    times, captured = [], []
    orig_sep, orig_dom = blind.BlindSeparator.separate, audit.other_dominant

    def timed(self, mix, sr=SR):
        t0 = time.perf_counter()
        out = orig_sep(self, mix, sr)
        if len(mix) == SR:
            times.append((time.perf_counter() - t0) * 1000)
        return out

    def grab(y, x, t, o, nz):
        captured.append((y, x, t, o, nz))
        return orig_dom(y, x, t, o, nz)

    blind.BlindSeparator.separate, audit.other_dominant = timed, grab
    try:
        res = audit.run(scene, load_profile(profiles, target), load_profile(profiles, other))
    finally:
        blind.BlindSeparator.separate, audit.other_dominant = orig_sep, orig_dom
    played, mix, t, o, nz = captured[0]
    n = min(len(played), len(mix))
    ms = np.asarray(times[1:] or [0.0])  # first call per process is kernel build / cold
    return {
        "backend": backend, "target": target, "scene": scene.name,
        "wrong_speaker_pct": 100 * res["other_dominant_s"] / res["seconds"],
        "other_dominant_s": res["other_dominant_s"], "seconds": res["seconds"],
        "stream_alone_wrong_s": res["other_dominant_stream_s"], "true_dropout_s": res["true_dropout_s"],
        "sisdri_db": float(metrics.si_sdr(played[:n], t[:n]) - metrics.si_sdr(mix[:n], t[:n])),
        "leak_db": leak_db(played, mix, t, o, nz),
        "p50_ms": float(np.percentile(ms, 50)), "p95_ms": float(np.percentile(ms, 95)), "n_windows": len(ms),
        "deadline_ms": 500.0, "late_windows": int((ms > 500.0).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--scene", action="append", required=True, help="TARGET:OTHER:SCENE_DIR")
    ap.add_argument("--backends", nargs="+", default=["mossformer2", "tiger"])
    ap.add_argument("--device", choices=("auto", "cpu"), default="auto")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.device == "cpu":
        os.environ["HEARING_DEVICE"] = "cpu"
    from hearing import device

    rows = []
    for backend in args.backends:
        for spec in args.scene:
            target, other, path = spec.split(":", 2)
            r = run_scene(backend, target, other, Path(path), args.profiles)
            r["device"] = device.pick()
            rows.append(r)
            print(json.dumps(r, default=float), flush=True)
    out = args.out or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / f"tiger_ab_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1, default=float))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
