"""T-402: fit the fixed-slot gate (``hearing.tse.fixed_gate``) on the team train takes.

    HEARING_SEP_CKPT=$HEARING_MODELS_DIR/T-374/fixed_best.pt .venv/bin/python -m eval.fixed_gate_train \
        --segments 160 --out $HEARING_RESULTS_DIR/T-402_gate.json

Renders 3 s mixtures from Aryan's and Joseph's ``profile-02..05`` takes (the T-374 train split; holdout, same,
profile-01/06/07 and fixture takes are never read) with random on/off turns of 0.4-1.6 s per voice (solo,
overlap and silence), a -6..+6 dB voice ratio, sometimes a third voice (Ahmad ``profile-02``, a stranger:
label "none" for both slots) and white noise at 5-20 dB SNR. Each 1.0 s / 0.5 s window is separated by the
fine-tuned net; for each role (target = aryan -> slot 0, joseph -> slot 1) the label of the window's last half
is target (target stem active), else partner (partner stem active), else none. Fits a multinomial logistic
(sklearn, C=1) and prints the weights to paste into ``fixed_gate.WEIGHTS``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hearing.tse.fixed_gate import CLASSES, FEATURES, features

SR = 16000
WIN, HOP = SR, SR // 2
SEG = 3 * SR
TRAIN_TAKES = ("profile-02", "profile-03", "profile-04", "profile-05")
SLOTS = ("aryan", "joseph")


def team_dir() -> Path:
    from eval.knob_sweep import fixtures_dir

    return fixtures_dir() / "team"


def turns(rng, n: int) -> np.ndarray:
    """0/1 envelope of random on/off runs of 0.4-1.6 s (20 ms ramps)."""
    env, i, on = np.zeros(n, np.float32), 0, bool(rng.random() < 0.5)
    while i < n:
        k = int(rng.uniform(0.4, 1.6) * SR)
        if on:
            env[i : i + k] = 1.0
        i, on = i + k, not on
    ramp = np.ones(320, np.float32) / 320
    return np.convolve(env, ramp, mode="same").astype(np.float32)


def active(x: np.ndarray, level: float) -> bool:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) > 0.1 * level


def render(rng, takes: dict, stranger: np.ndarray) -> tuple[np.ndarray, dict]:
    stems = {}
    for name in SLOTS:
        src = takes[name][rng.integers(len(takes[name]))]
        o = rng.integers(0, len(src) - SEG)
        stems[name] = src[o : o + SEG] * turns(rng, SEG)
    g = 10.0 ** (rng.uniform(-6, 6) / 20.0)
    stems["joseph"] = stems["joseph"] * g
    third = np.zeros(SEG, np.float32)
    if rng.random() < 0.2:
        o = rng.integers(0, len(stranger) - SEG)
        third = stranger[o : o + SEG] * turns(rng, SEG) * 10.0 ** (rng.uniform(-6, 0) / 20.0)
    speech = stems["aryan"] + stems["joseph"] + third
    p = float(np.mean(np.square(speech, dtype=np.float64))) + 1e-9
    noise = rng.standard_normal(SEG).astype(np.float32) * np.sqrt(p / 10.0 ** (rng.uniform(5, 20) / 10.0))
    mix = speech + noise
    peak = float(np.max(np.abs(mix))) + 1e-9
    scale = 10.0 ** (rng.uniform(-25, -8) / 20.0) / np.sqrt(np.mean(np.square(mix, dtype=np.float64)) + 1e-12)
    scale = min(scale, 0.95 / peak)
    return (mix * scale).astype(np.float32), {k: v * scale for k, v in stems.items()}


def collect(segments: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    from eval.namecall_eval import _speech_level
    from hearing.audio import io as aio
    from hearing.tse.enroll import load_profile, profile_clip
    from hearing.tse.enrolled import EnrolledCandidate, _unit_embedding

    rng = np.random.default_rng(seed)
    root = team_dir()
    takes = {n: [aio.load(root / n / f"{t}.wav", SR)[0] for t in TRAIN_TAKES] for n in SLOTS}
    stranger = aio.load(root / "ahmad" / "profile-02.wav", SR)[0]
    cand = EnrolledCandidate()
    cand.load()
    from eval.knob_sweep import fixtures_dir

    prof = {n: profile_clip(load_profile(n, fixtures_dir() / "profiles", SR), n) for n in SLOTS}
    emb = {n: cand.target_embedding(prof[n]) for n in SLOTS}
    xs, ys = [], []
    for s in range(segments):
        mix, stems = render(rng, takes, stranger)
        level = {n: _speech_level(stems[n]) if np.any(stems[n]) else np.inf for n in SLOTS}
        for i in range(0, SEG - WIN + 1, HOP):
            w = mix[i : i + WIN]
            streams = cand.separator.separate(w, SR)
            e = [_unit_embedding(cand.embedder, y) for y in streams]
            for k, name in enumerate(SLOTS):
                other = SLOTS[1 - k]
                cos_t, cos_p = float(np.dot(emb[name], e[k])), float(np.dot(emb[other], e[k]))
                xs.append(features(streams[k], streams[1 - k], w, cos_t, cos_p))
                tail = slice(i + WIN // 2, i + WIN)
                if active(stems[name][tail], level[name]):
                    ys.append(0)
                elif active(stems[other][tail], level[other]):
                    ys.append(1)
                else:
                    ys.append(2)
        if (s + 1) % 20 == 0:
            print(f"  {s + 1}/{segments} segments, {len(ys)} rows", flush=True)
    return np.array(xs), np.array(ys)


def fit(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, dict]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False)
    clf.fit(xs, ys)
    w = clf.coef_
    pred = clf.predict(xs)
    stats = {"rows": len(ys), "counts": {c: int(np.sum(ys == i)) for i, c in enumerate(CLASSES)},
             "train_acc": round(float(np.mean(pred == ys)), 3),
             "target_recall": round(float(np.mean(pred[ys == 0] == 0)), 3) if np.any(ys == 0) else None,
             "partner_as_target": round(float(np.mean(pred[ys == 1] == 0)), 3) if np.any(ys == 1) else None}
    return w, stats


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segments", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    xs, ys = collect(args.segments, args.seed)
    w, stats = fit(xs, ys)
    print(json.dumps(stats))
    print("features:", FEATURES)
    print("WEIGHTS = np.array([\n" + "".join(f"    [{', '.join(f'{v:.4f}' for v in row)}],\n" for row in w) + "])")
    if args.out:
        args.out.write_text(json.dumps({"weights": w.tolist(), "features": list(FEATURES), "classes": list(CLASSES),
                                        "stats": stats, "segments": args.segments, "seed": args.seed,
                                        "takes": list(TRAIN_TAKES)}, indent=1))


if __name__ == "__main__":
    main()
