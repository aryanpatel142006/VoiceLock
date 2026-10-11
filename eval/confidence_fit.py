"""Fit the enrolled tier's pick confidence (T-134) on a two-voice scene with saved profiles.

    .venv/bin/python -m eval.confidence_fit                  # pp-mm stand-in (GRID s1 vs s2, male-male)
    .venv/bin/python -m eval.confidence_fit --team DIR       # a teammate recording with profile takes

Runs :class:`hearing.tse.enrolled.EnrolledCandidate` offline (real weights) over the scene in 1.0 s
windows every 0.5 s, the target and the other voice loaded as saved profiles exactly as
``Session.load_voice_profile`` does (``split_pieces`` -> ``profile_clip``), and logs per window the
scoring (``lda`` / ``cosine``), the score margin, the confidence and whether the pick was right (the
output correlates more with the target's stem than with the other's; windows where the target is
silent are skipped). It prints, per scoring, the smallest margin above which the picks are at least
``--precision`` right: that is the full-confidence margin (``CONF_MARGIN`` / ``CONF_MARGIN_LDA``).
Results go to ``$HEARING_RESULTS_DIR/confidence_fit_<stamp>.json``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from eval import namecall_eval as nc
from eval import pipeline_eval as pe

SR = 16000
WIN_S, HOP_S = 1.0, 0.5


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(abs(np.dot(a, b)) / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0


def windows(scene: dict, cand) -> list[dict]:
    from hearing.tse.enroll import profile_clip, split_pieces

    enroll = profile_clip(split_pieces(np.asarray(scene["profiles"]["target"], np.float32)))
    other = profile_clip(split_pieces(np.asarray(scene["profiles"]["other"], np.float32)))
    cand.prepare(enroll, other, SR)
    a, b, mix = scene["stems"]["A"], scene["stems"]["B"], scene["mix"]
    lvl_a = nc._speech_level(a)
    win, hop = round(WIN_S * SR), round(HOP_S * SR)
    rows = []
    for i in range(0, len(mix) - win + 1, hop):
        sl = slice(i, i + win)
        if nc._rms(a[sl]) <= 10 ** (pe.ACTIVE_DB / 20) * lvl_a:
            continue
        y = cand.extract(mix[sl], SR, enroll=enroll, other=other)
        if cand.gated:
            right = None
        else:
            right = _corr(y, a[sl]) >= _corr(y, b[sl])
        rows.append({"t": round(i / SR, 2), "scoring": cand.scoring, "margin": cand.margin_score,
                     "confidence": round(float(cand.confidence), 3), "vetoed": cand.vetoed, "held": cand.held,
                     "gated": bool(cand.gated), "right": right,
                     "both": bool(nc._rms(b[sl]) > 10 ** (pe.ACTIVE_DB / 20) * nc._speech_level(b))})
    return rows


def fit_margin(rows: list[dict], precision: float) -> float | None:
    """Smallest margin m such that every window with margin >= m is right at least ``precision``."""
    pts = sorted((r["margin"], r["right"]) for r in rows if r["right"] is not None and r["margin"] is not None)
    if not pts:
        return None
    for k, (m, _) in enumerate(pts):
        tail = [ok for _, ok in pts[k:]]
        if sum(tail) / len(tail) >= precision:
            return round(float(m), 3)
    return None


def main(argv: list[str] | None = None) -> int:
    from hearing.tse.enrolled import EnrolledCandidate

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team", type=Path, action="append", default=[], help="teammate dir with profile takes")
    ap.add_argument("--precision", type=float, default=0.97)
    ap.add_argument("--no-lda", action="store_true", help="cosine scoring only (closed_set=False)")
    args = ap.parse_args(argv)
    scenes = [pe.build_profile_scene()] + [pe.load_team(p) for p in args.team]
    out = []
    for sc in scenes:
        if "profiles" not in sc:
            print(f"{sc['id']}: no profile takes, skipped")
            continue
        cand = EnrolledCandidate(closed_set=not args.no_lda)
        rows = windows(sc, cand)
        right = [r for r in rows if r["right"] is not None]
        fit = fit_margin(rows, args.precision)
        acc = sum(r["right"] for r in right) / len(right) if right else None
        conf_ok = [r["confidence"] for r in right if r["right"]]
        conf_bad = [r["confidence"] for r in right if not r["right"]]
        summ = {"scene": sc["id"], "windows": len(rows), "gated": sum(r["gated"] for r in rows),
                "pick_accuracy": None if acc is None else round(acc, 3),
                "scoring": sorted({r["scoring"] for r in rows}), "fit_margin": fit,
                "conf_median_right": round(float(np.median(conf_ok)), 3) if conf_ok else None,
                "conf_median_wrong": round(float(np.median(conf_bad)), 3) if conf_bad else None,
                "margin_median": round(float(np.median([r["margin"] for r in rows if r["margin"] is not None])), 3)}
        print(json.dumps(summ))
        out.append({**summ, "rows": rows})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = nc.results_dir() / f"confidence_fit_{stamp}.json"
    path.write_text(json.dumps({"precision": args.precision, "lda": not args.no_lda, "results": out}, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
