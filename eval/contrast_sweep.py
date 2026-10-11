"""Sweep the enrolled tier's ``contrast`` floor on the team scenes with the saved profiles (T-178).

    .venv/bin/python -m eval.contrast_sweep --profiles DIR --scene SCENE_DIR [--scene ...] [--contrast 0,0.1,0.2,0.3]

Runs :class:`hearing.tse.enrolled.EnrolledCandidate` offline (real weights, closed-set LDA as on the
Profile path) over each scene in 1.0 s windows every 0.5 s, the target and the other voice loaded from
``DIR/<name>/`` exactly as ``Session.load_voice_profile`` does (``load_profile`` -> ``profile_clip``).
A scene is a T-141 ``demo_scene`` dir (``stems/{mix,target,other}.wav``). Per contrast value it prints:

* ``pair_cos``: cos(target profile, other profile) and whether the negative is in use (``<= SAME_VOICE``);
* ``wrong_pct`` / ``longest_wrong_s``: target-active windows whose (ungated) output correlates more with
  the other talker's stem than with the target's, and the longest run of them (hop units);
* ``gated_pct``: target-active windows returned as silence (the cost of the floor);
* ``absent_db``: output minus input level over the hops where the target is silent and the other talks;
* ``fit_margin_95`` / ``_97``: ``eval.confidence_fit.fit_margin`` over the target-active windows (the
  ``CONF_MARGIN*`` refit on the real voices, in the units of ``scoring``);
* ``si_sdri_db`` / ``leak_db``: on the streamed output (each window's last hop), the TSE stream alone
  (no mixer residual): SI-SDR gain on the target stem, and the other stem's gain in the output.

Results go to ``$HEARING_RESULTS_DIR/contrast_sweep_<stamp>.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval import namecall_eval as nc
from eval import pipeline_eval as pe
from eval.metrics import si_sdr
from hearing.audio import io as aio

SR = 16000
WIN_S, HOP_S = 1.0, 0.5


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(np.dot(a, b)) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def _db(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12))


def load_profile(root: Path, name: str) -> np.ndarray:
    from hearing.tse.enroll import load_profile as pieces
    from hearing.tse.enroll import profile_clip

    return profile_clip(pieces(name, root, SR), name)  # T-402: named, so HEARING_PICK=fixed finds the slot


def load_scene(path: Path) -> dict:
    st = path / "stems"
    mix = aio.load(st / "mix.wav", SR)[0]
    a = aio.load(st / "target.wav", SR)[0][: len(mix)]
    b = aio.load(st / "other.wav", SR)[0][: len(mix)]
    return {"id": path.name, "mix": mix, "A": a, "B": b}


def run(scene: dict, enroll: np.ndarray, other: np.ndarray, contrast: float) -> dict:
    from hearing.tse.enrolled import EnrolledCandidate

    cand = EnrolledCandidate(contrast=contrast)
    cand.prepare(enroll, other, SR)
    a, b, mix = scene["A"], scene["B"], scene["mix"]
    thr_a = 10 ** (pe.ACTIVE_DB / 20) * nc._speech_level(a)
    thr_b = 10 ** (pe.ACTIVE_DB / 20) * nc._speech_level(b)
    win, hop = round(WIN_S * SR), round(HOP_S * SR)
    out = np.zeros_like(mix)
    rows = []
    for i in range(0, len(mix) - win + 1, hop):
        sl = slice(i, i + win)
        y = cand.extract(mix[sl], SR, enroll=enroll, other=other)
        if i == 0:
            out[:hop] = y[:hop]
        out[i + hop : i + win] = y[hop:]
        tail = slice(i + hop, i + win)  # the hop this window contributes when streaming
        active = nc._rms(a[sl]) > thr_a
        right = None if (cand.gated or not active) else _corr(y, a[sl]) >= _corr(y, b[sl])
        rows.append({"t": round(i / SR, 2), "active": bool(active), "gated": bool(cand.gated), "right": right,
                     "score": round(float(cand.scores[cand.selected_stream]), 3), "scoring": cand.scoring,
                     "margin": cand.margin_score,
                     "absent": bool(nc._rms(a[tail]) <= thr_a and nc._rms(b[tail]) > thr_b),
                     "in_db": round(_db(mix[tail]), 2), "out_db": round(_db(out[tail]), 2)})
    act = [r for r in rows if r["active"]]
    wrong, longest, run_n = 0, 0, 0
    for r in act:
        bad = r["right"] is False
        wrong += bad
        run_n = run_n + 1 if bad else 0
        longest = max(longest, run_n)
    absent = [r for r in rows if r["absent"]]
    n = len(out)
    gb = float(np.dot(out[:n], b[:n]) / (np.dot(b[:n], b[:n]) + 1e-12))
    from eval.confidence_fit import fit_margin

    fit = {p: fit_margin([r for r in act if r["right"] is not None], p) for p in (0.95, 0.97)}
    return {"scene": scene["id"], "contrast": contrast, "fit_margin_95": fit[0.95], "fit_margin_97": fit[0.97], "pair_cos": None if cand.pair_cos is None else round(cand.pair_cos, 3),
            "negative_active": cand.negative_active, "scoring": sorted({r["scoring"] for r in rows}),
            "windows": len(rows), "active_windows": len(act),
            "wrong_pct": round(100.0 * wrong / max(len(act), 1), 1), "longest_wrong_s": round(longest * HOP_S, 2),
            "gated_pct": round(100.0 * sum(r["gated"] for r in act) / max(len(act), 1), 1),
            "absent_hops": len(absent),
            "absent_db": round(float(np.mean([r["out_db"] - r["in_db"] for r in absent])), 1) if absent else None,
            "si_sdri_db": round(si_sdr(out, a) - si_sdr(mix, a), 2),
            "leak_db": round(20 * np.log10(abs(gb) + 1e-9), 1), "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", type=Path, required=True, help="dir with <target>/ and <other>/ profile wavs")
    ap.add_argument("--target", default="aryan")
    ap.add_argument("--other", default="joseph")
    ap.add_argument("--scene", type=Path, action="append", required=True)
    ap.add_argument("--contrast", default="0,0.1,0.2,0.3")
    args = ap.parse_args(argv)
    enroll, other = load_profile(args.profiles, args.target), load_profile(args.profiles, args.other)
    out = []
    for p in args.scene:
        sc = load_scene(p)
        for c in (float(v) for v in args.contrast.split(",")):
            r = run(sc, enroll, other, c)
            print(json.dumps({k: v for k, v in r.items() if k != "rows"}), flush=True)
            out.append(r)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = nc.results_dir() / f"contrast_sweep_{stamp}.json"
    path.write_text(json.dumps({"profiles": str(args.profiles), "target": args.target, "other": args.other,
                                "results": out}, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
