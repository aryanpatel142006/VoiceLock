"""Which two teammates make the best demo pair? Pairwise comparison on one shared passage (T-369).

    .venv/bin/python -m eval.pair_compare [--names aryan joseph ahmad] [--out FILE]

Data: ``$HEARING_FIXTURES_DIR/team/<name>/same-01.wav`` (enrollment) and ``same-02.wav`` (test): one 40 s passage
read by every teammate twice on the MacBook Pro microphone (Oct 10), so content is identical across people and the
comparison is fair. Nothing else of theirs is used.

1. **Embeddings.** ECAPA cosine between every pair of ``same-01`` enrollment embeddings, as the enrolled tier
   derives them (``EnrolledCandidate.target_embedding`` of the profile clip). Short-crop confusion: 1.0 s crops of
   each ``same-02`` every 0.5 s (active speech only), raw ECAPA, nearest enrollment among all three (3-way matrix)
   and within each pair (2-way error, mean cosine margin own - other).
2. **Pipeline.** For each pair and both target roles: enroll both from ``same-01`` (target + the other as the
   closed-set negative, as two saved profiles do live), build scenes from ``same-02`` in the shape target alone
   0-5 s / overlap 5-15 s / target alone 15-20 s, overlap SIR 0 and -6 dB (the other that much LOUDER), with no
   noise or DEMAND cafeteria noise at SNR 10 dB (``eval.knob_sweep``'s levels). The other's pieces start half-way
   through its pool, so the two never read the same words at once. The real ``EnrolledCandidate`` (MossFormer2_SS +
   ECAPA pick, demo defaults: every T-180 knob env unset) replays 1.0 s windows every 0.5 s and the played output is
   ``stream + floor * (window mix - stream)`` (``eval.knob_sweep.replay`` / ``compose_output``). Scores
   (``eval.knob_sweep.score``): wrong-speaker % of scene time (the session_score other-dominant rule), leak dB,
   SI-SDRi (whole scene and overlap section), target >= 10 dB down s, gated-window share.

Ranking: mean wrong-speaker % over both roles and the four conditions (lower first), ties by mean leak. Results:
``$HEARING_RESULTS_DIR/pair_compare_<stamp>.json``. Caveats: one 40 s take per person per role, digital mixtures of
separately recorded voices (no shared room, no crosstalk), one session, one microphone.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

from eval import knob_sweep as ks

SR = 16000
NAMES = ("aryan", "joseph", "ahmad")
SIRS = (0.0, -6.0)
NOISES = ("none", "cafe")
CROP_S, CROP_HOP_S = 1.0, 0.5
# (section, start s, end s, who): target alone / overlap / target alone
SECTIONS = (("target_alone", 0.0, 5.0, "target"), ("overlap", 5.0, 15.0, "both"),
            ("target_alone_2", 15.0, 20.0, "target"))
DUR_S = 20.0
OVERLAP = (5.0, 15.0)


def team_dir() -> Path:
    return ks.fixtures_dir() / "team"


def load_take(name: str, take: str, root: Path | None = None) -> np.ndarray:
    from hearing.audio import io as aio

    return aio.load((root or team_dir()) / name / f"{take}.wav", SR)[0]


def compose(target_pool: list[np.ndarray], other_pool: list[np.ndarray], sir_db: float,
            noise: np.ndarray | None, sections=SECTIONS, dur_s: float = DUR_S) -> dict:
    """Stems ``{mix, target, other, noise}``; levels as ``knob_sweep.compose`` (target at its speech level, the
    other ``-sir_db`` dB above it, noise at SNR 10 dB under the target, peak-limited)."""
    n = round(dur_s * SR)
    t, o = np.zeros(n, np.float32), np.zeros(n, np.float32)
    ct = co = 0
    for _, a, b, who in sections:
        s, e = round(a * SR), round(b * SR)
        ct = ks.lay(target_pool, s, e, ct, t)
        if who == "both":
            co = ks.lay(other_pool, s, e, co, o)
    ref = 10 ** (ks.SPEECH_LEVEL_DB / 20)
    t *= ref / ks._active_rms(t)
    o *= ref * 10 ** (-sir_db / 20) / ks._active_rms(o)
    nz = np.zeros(n, np.float32) if noise is None else np.resize(noise, n).astype(np.float32)
    if noise is not None:
        nz *= ref / (float(np.sqrt(np.mean(nz.astype(np.float64) ** 2))) + 1e-12) / 10 ** (ks.CAFE_SNR_DB / 20)
    mix = t + o + nz
    g = min(1.0, ks.PEAK / float(np.max(np.abs(mix))))
    return {"mix": mix * g, "target": t * g, "other": o * g, "noise": nz * g}


def half_shift(pool: list) -> list:
    """The pool started half-way through, so the other never reads the target's words at the same time."""
    h = len(pool) // 2
    return pool[h:] + pool[:h]


def scene(target: str, other: str, sir: float, noise: str, pools: dict, cafe: np.ndarray | None) -> dict:
    st = compose(pools[target], half_shift(pools[other]), sir, cafe if noise == "cafe" else None)
    sid = f"{target}-vs-{other}-sir{int(sir):+d}-{noise}".replace("+0", "0")
    return {"id": sid, "mix": st["mix"], "A": st["target"], "B": st["other"], "noise": st["noise"]}


def crops(x: np.ndarray, crop_s: float = CROP_S, hop_s: float = CROP_HOP_S) -> list[np.ndarray]:
    """``crop_s`` crops every ``hop_s`` whose RMS is above 0.3 x the take's 95th-percentile crop RMS."""
    n, h = round(crop_s * SR), round(hop_s * SR)
    cs = [x[i: i + n] for i in range(0, len(x) - n + 1, h)]
    r = np.array([np.sqrt(np.mean(np.asarray(c, np.float64) ** 2)) for c in cs])
    thr = 0.3 * np.percentile(r, 95) if len(r) else 0.0
    return [c for c, v in zip(cs, r, strict=True) if v > thr]


def confusion(crop_emb: dict[str, np.ndarray], enroll_emb: dict[str, np.ndarray]) -> dict:
    """3-way nearest-enrollment counts and, per pair, the 2-way error and mean margin (own - other cosine)."""
    names = list(enroll_emb)
    e = np.stack([enroll_emb[k] for k in names])
    matrix = {}
    for a in names:
        cos = crop_emb[a] @ e.T
        best = np.argmax(cos, axis=1)
        matrix[a] = {b: int(np.sum(best == j)) for j, b in enumerate(names)}
    pairs = {}
    for a, b in itertools.combinations(names, 2):
        ma = crop_emb[a] @ enroll_emb[a] - crop_emb[a] @ enroll_emb[b]
        mb = crop_emb[b] @ enroll_emb[b] - crop_emb[b] @ enroll_emb[a]
        pairs[f"{a}-{b}"] = {"crops": int(len(ma) + len(mb)), "errors": int(np.sum(ma <= 0) + np.sum(mb <= 0)),
                             "error_pct": round(100.0 * float(np.sum(ma <= 0) + np.sum(mb <= 0)) / (len(ma) + len(mb)), 1),
                             "margin_" + a: round(float(np.mean(ma)), 3), "margin_" + b: round(float(np.mean(mb)), 3)}
    return {"matrix": matrix, "pairs": pairs}


def cosine_table(emb: dict[str, np.ndarray]) -> dict[str, float]:
    return {f"{a}-{b}": round(float(emb[a] @ emb[b]), 3) for a, b in itertools.combinations(emb, 2)}


def section_si_sdri(played: np.ndarray, sc: dict, span=OVERLAP) -> float:
    from eval.metrics import si_sdri

    s, e = round(span[0] * SR), round(span[1] * SR)
    return float(si_sdri(played[s:e], sc["mix"][s:e], sc["A"][s:e]))


def rank(rows: list[dict]) -> list[dict]:
    """Per pair (both roles, every condition): mean wrong %, mean leak, mean SI-SDRi, worst role; best first."""
    out = {}
    for r in rows:
        out.setdefault(r["pair"], []).append(r)
    ranked = []
    for pair, rs in out.items():
        by_role = {}
        for r in rs:
            by_role.setdefault(r["target"], []).append(r["wrong_pct"])
        worst = max(by_role, key=lambda k: np.mean(by_role[k]))
        ranked.append({"pair": pair, "n": len(rs),
                       "wrong_pct": round(float(np.mean([r["wrong_pct"] for r in rs])), 1),
                       "leak_db": round(float(np.mean([r["leak_db"] for r in rs])), 2),
                       "si_sdri": round(float(np.mean([r["si_sdri"] for r in rs])), 2),
                       "si_sdri_overlap": round(float(np.mean([r["si_sdri_overlap"] for r in rs])), 2),
                       "atten_s": round(float(np.mean([r["atten_s"] for r in rs])), 2),
                       "gated_pct": round(float(np.mean([r["gated_pct"] for r in rs])), 1),
                       "worst_role": worst, "worst_role_wrong_pct": round(float(np.mean(by_role[worst])), 1)})
    ranked.sort(key=lambda p: (p["wrong_pct"], p["leak_db"]))
    return ranked


def pair_key(a: str, b: str, names=NAMES) -> str:
    i, j = sorted((names.index(a), names.index(b)))
    return f"{names[i]}-{names[j]}"


def embeddings(names, takes) -> dict:
    """Section 1: enrollment cosines and the short-crop confusion."""
    from hearing.tse.enroll import profile_clip, split_pieces
    from hearing.tse.enrolled import EnrolledCandidate, _unit_embedding

    cand = EnrolledCandidate()
    cand.load()
    enroll = {n: cand.target_embedding(profile_clip(split_pieces(takes[n]["same-01"], SR))) for n in names}
    raw = {n: _unit_embedding(cand.embedder, takes[n]["same-01"]) for n in names}
    crop_emb = {n: np.stack([_unit_embedding(cand.embedder, c) for c in crops(takes[n]["same-02"])]) for n in names}
    return {"cosine_enrolled": cosine_table(enroll), "cosine_raw_clip": cosine_table(raw),
            "crop_s": CROP_S, "crops": {n: len(crop_emb[n]) for n in names},
            "confusion_vs_enrolled": confusion(crop_emb, enroll)}


def pipeline(names, takes) -> list[dict]:
    """Section 2: every (pair, target role, SIR, noise) scene through the replayed enrolled tier."""
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enroll import profile_clip, split_pieces

    pools = {n: ks.speech_chunks(takes[n]["same-02"]) for n in names}
    profiles = {n: profile_clip(split_pieces(takes[n]["same-01"], SR)) for n in names}
    cafe = ks.cafe_noise(ks.fixtures_dir())
    sep = ks.CachedSeparator(BlindSeparator())
    rows = []
    for target, other in itertools.permutations(names, 2):
        for sir, noise in itertools.product(SIRS, NOISES):
            t0 = time.monotonic()
            sc = scene(target, other, sir, noise, pools, cafe)
            wins = ks.replay(sc, sep, profiles[target], profiles[other], 0.0, None)
            played = ks.compose_output(sc, wins)
            m = ks.score(played, sc)
            from eval.metrics import si_sdri

            row = {"pair": pair_key(target, other, tuple(names)), "target": target, "other": other, "sir_db": sir,
                   "noise": noise, **m, "wrong_pct": round(100.0 * m["wrong_s"] / m["seconds"], 1),
                   "si_sdri": round(float(si_sdri(played, sc["mix"], sc["A"])), 2),
                   "si_sdri_overlap": round(section_si_sdri(played, sc), 2),
                   "gated_windows": sum(w["gated"] for w in wins), "windows": len(wins),
                   "gated_pct": round(100.0 * sum(w["gated"] for w in wins) / len(wins), 1)}
            rows.append(row)
            print(f"{sc['id']:32s} wrong {row['wrong_pct']:5.1f} %  leak {row['leak_db']:+6.2f} dB  SI-SDRi "
                  f"{row['si_sdri']:+6.2f} (ovl {row['si_sdri_overlap']:+6.2f})  gated {row['gated_pct']:4.1f} %  "
                  f"{time.monotonic() - t0:4.1f} s", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    from hearing import device

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", nargs="+", default=list(NAMES))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    saved = {k: os.environ.pop(k, None) for k in ks.ENVS}  # demo defaults: every knob unset
    try:
        takes = {n: {t: load_take(n, t) for t in ("same-01", "same-02")} for n in args.names}
        emb = embeddings(args.names, takes)
        print(json.dumps(emb, indent=1), flush=True)
        rows = pipeline(args.names, takes)
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    ranked = rank(rows)
    for p in ranked:
        print(json.dumps(p), flush=True)
    out = args.out or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / (
        f"pair_compare_{time.strftime('%Y%m%d-%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"task": "T-369", "device": device.pick(), "names": args.names,
                               "embeddings": emb, "rows": rows, "ranking": ranked}, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
