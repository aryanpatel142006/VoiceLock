"""Offline A/B of the mixer floor for gated / unsure enrolled windows (T-180 knobs K2, K3, K5).

The session A/B (``demo_regress``) could not be used on the shared Mac: the enrolled tier overran
in every row and the ladder played ``audio-only`` (T-178, T-180). This runs the real
``EnrolledCandidate`` (aryan profile as target, joseph as the negative) over 1.0 s windows every
0.5 s, like ``eval.contrast_sweep``, keeps each window's last hop, its best stream, its confidence and
whether it was gated, and then applies what the session's ``mixer`` does per hop:
``out = s + g * (mix - s)``, ``s`` = the window's output (silence when gated, the best stream under
K5), ``g`` = ``10 ** (-floor_db / 20)`` from :func:`hearing.fusion.mixer.confidence_floor_db` with the
row's env knobs (``conf0``: ``HEARING_TSE_CONFIDENCE=0``, i.e. ``max_duck_db=20`` everywhere). Slider 100,
no gain ramp, no output AGC (pre-AGC). Metrics on the whole stitched output:

* SI-SDRi and leak (least-squares gain of the other talker's stem in the output, dB);
* wrong-speaker % / longest wrong stretch: :func:`eval.pipeline_eval.routing` (0.25 s windows);
* true dropouts: seconds of 0.5 s hops where the target talks but its gain in the output is < -10 dB;
* DNSMOS OVRL (``eval.dnsmos``, real when available).

    python -m eval.floor_ab --profiles <generated>/profiles --scene <generated>/t141/aryan/demo-team-easy ...
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from eval import namecall_eval as nc
from eval import pipeline_eval as pe
from eval.contrast_sweep import HOP_S, SR, WIN_S, load_profile
from eval.metrics import si_sdr
from hearing.audio import io as aio
from hearing.fusion.mixer import confidence_floor_db
from hearing.latency_preset import STRICT_KWARGS

DROP_DB = -10.0  # a target-active hop whose target gain is below this is a dropout
# row -> (isolation, env for the mixer floor, K5 on)
COMBOS: dict[str, tuple[str, dict, bool]] = {
    "strict": ("strict", {}, False),
    "standard": ("standard", {}, False),
    "strict+conf0": ("strict", {"HEARING_TSE_CONFIDENCE": "0"}, False),
    "standard+conf0": ("standard", {"HEARING_TSE_CONFIDENCE": "0"}, False),
    "standard+K2": ("standard", {"HEARING_GATED_FLOOR_DB": "20"}, False),
    "standard+K3": ("standard", {"HEARING_UNSURE_DB": "12"}, False),
    "standard+K5": ("standard", {}, True),
    "standard+K2+K5": ("standard", {"HEARING_GATED_FLOOR_DB": "20"}, True),
}


def load_scene(path: Path) -> dict:
    st = path / "stems"
    mix = aio.load(st / "mix.wav", SR)[0]
    n = len(mix)
    return {"id": path.name, "mix": mix, "A": aio.load(st / "target.wav", SR)[0][:n],
            "B": aio.load(st / "other.wav", SR)[0][:n], "noise": aio.load(st / "noise.wav", SR)[0][:n]}


def windows(scene: dict, enroll: np.ndarray, other: np.ndarray, isolation: str) -> list[dict]:
    """Per hop: [i0, i1), best stream, gated, confidence (the candidate's own, 0 when gated)."""
    from hearing.tse.enrolled import EnrolledCandidate

    cand = EnrolledCandidate(**(STRICT_KWARGS if isolation == "strict" else {}))
    cand.prepare(enroll, other, SR)
    mix = scene["mix"]
    win, hop = round(WIN_S * SR), round(HOP_S * SR)
    hops = []
    old = os.environ.pop("HEARING_GATED_PLAY", None)
    os.environ["HEARING_GATED_PLAY"] = "stream"  # always get the best stream; gating is applied below
    try:
        for i in range(0, len(mix) - win + 1, hop):
            y = cand.extract(mix[i : i + win], SR, enroll=enroll, other=other)
            duck = bool(cand.paused and cand.gated and not np.any(y))
            if i == 0:
                hops.append({"i0": 0, "i1": hop, "y": y[:hop], "gated": cand.gated, "duck": duck,
                             "conf": cand.confidence})
            hops.append({"i0": i + hop, "i1": i + win, "y": y[hop:],
                         "gated": cand.gated, "duck": duck, "conf": cand.confidence})
    finally:
        os.environ.pop("HEARING_GATED_PLAY", None)
        if old is not None:
            os.environ["HEARING_GATED_PLAY"] = old
    return hops


def render(scene: dict, hops: list[dict], env: dict, k5: bool) -> np.ndarray:
    mix = scene["mix"]
    out = mix.copy()
    saved = {k: os.environ.get(k) for k in ("HEARING_GATED_FLOOR_DB", "HEARING_UNSURE_DB")}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in env.items() if k != "HEARING_TSE_CONFIDENCE"})
    try:
        for h in hops:
            m = mix[h["i0"] : h["i1"]]
            play_stream = not h["gated"] or (k5 and not h["duck"])
            s = h["y"] if play_stream else np.zeros_like(m)
            if env.get("HEARING_TSE_CONFIDENCE") == "0":
                floor_db = 20.0
            else:
                floor_db = confidence_floor_db(h["conf"] or 0.0)
            out[h["i0"] : h["i1"]] = s + 10 ** (-floor_db / 20) * (m - s)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
    return out


def score(scene: dict, out: np.ndarray, dnsmos) -> dict:
    a, b, mix = scene["A"], scene["B"], scene["mix"]
    g = nc._lsq_gains(out, [a, b, scene["noise"]])
    thr = 10 ** (pe.ACTIVE_DB / 20) * nc._speech_level(a)
    hop = round(HOP_S * SR)
    drop = 0
    for i in range(0, len(out) - hop + 1, hop):
        sl = slice(i, i + hop)
        if nc._rms(a[sl]) > thr:
            ga = nc._lsq_gains(out[sl], [a[sl], b[sl], scene["noise"][sl]])[0]
            drop += 20 * np.log10(max(abs(ga), 1e-4)) < DROP_DB
    sc = {"stems": {"A": a, "B": b}, "noise": scene["noise"], "intended": [(0.0, "A")]}
    r = pe.routing(sc, out, 0.0)
    return {"si_sdri_db": round(si_sdr(out, a) - si_sdr(mix, a), 2),
            "leak_db": round(20 * np.log10(abs(g[1]) + 1e-9), 1),
            "true_dropout_s": round(drop * HOP_S, 2),
            "dnsmos_ovrl": round(float(dnsmos.score(out, SR).ovrl), 2),
            "wrong_pct": r["wrong_pct"], "longest_wrong_s": r["longest_wrong_s"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", type=Path, required=True)
    ap.add_argument("--target", default="aryan")
    ap.add_argument("--other", default="joseph")
    ap.add_argument("--scene", type=Path, action="append", required=True)
    ap.add_argument("--dnsmos", default="auto", choices=["auto", "dnsmos", "mock"])
    args = ap.parse_args(argv)
    from eval.dnsmos import get_dnsmos

    dn = get_dnsmos(args.dnsmos)
    enroll, other = load_profile(args.profiles, args.target), load_profile(args.profiles, args.other)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    results = []
    for p in args.scene:
        sc = load_scene(p)
        cache = {iso: windows(sc, enroll, other, iso) for iso in ("strict", "standard")}
        for name, (iso, env, k5) in COMBOS.items():
            hops = cache[iso]
            r = {"scene": sc["id"], "combo": name, "isolation": iso, "env": env, "k5": k5,
                 "gated_hops": sum(h["gated"] for h in hops), "hops": len(hops),
                 **score(sc, render(sc, hops, env, k5), dn), "run": f"floor_ab_{stamp}"}
            print(json.dumps(r), flush=True)
            results.append(r)
    path = nc.results_dir() / f"floor_ab_{stamp}.json"
    path.write_text(json.dumps({"profiles": str(args.profiles), "target": args.target, "other": args.other,
                                "results": results}, indent=1))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
