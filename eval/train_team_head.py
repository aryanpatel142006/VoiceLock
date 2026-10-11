"""T-336: train and test the team head (``hearing/tse/team_head.py``) on separated 1 s windows.

Train ONLY on the enrollment cuts ``$HEARING_FIXTURES_DIR/team/{aryan,joseph}/profile-01.wav``: random
on/off talk turns of both voices mixed at aryan:joseph ratios +10/+5/0/-5/-10 dB (0/-5/-10 dB SIR for either
voice as target) plus single-talker takes, all over babble at 5 dB SNR, cut into the live 1.0 s window /
0.5 s hop and run through the real MossFormer2 separator. Each stream gets the live features (LDA score
from the saved profiles, cosines, energy) for both target directions, labelled 1 when the target's stem
dominates the stream by >= 3 dB (oracle least-squares projection), else 0 (other, mixed or silent).

Test ONLY on the T-141 scenes (disjoint takes): per window, the LDA gate (accept the argmax stream when its
LDA score >= 0) against the head (accept when p >= threshold), and a target-absent take (the scene's
other + noise stems: seconds accepted as target).

    .venv/bin/python -m eval.train_team_head            # writes $HEARING_RESULTS_DIR/T-336_team_head.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from hearing.audio.io import load
from hearing.tse.enroll import profile_pairs
from hearing.tse.enrolled import EnrolledCandidate, _unit_embedding
from hearing.tse.team_head import FEATURES, TeamHead, fit_logistic, stream_features

SR = 16000
WIN, HOP = SR, SR // 2  # the live enrolled window / hop
RATIOS_DB = (10.0, 5.0, 0.0, -5.0, -10.0)  # aryan level re joseph
SNR_DB = 5.0
DOMINANT_DB = 3.0  # target component must beat the other's by this to label a stream "target"
AUDIBLE_DBFS = -45.0  # and be at least this loud
SEG_S = 15.0
NAMES = ("aryan", "joseph")


def _root(env, default):
    return Path(os.environ.get(env) or default)


FIXTURES = _root("HEARING_FIXTURES_DIR", "eval/fixtures/generated")
RESULTS = _root("HEARING_RESULTS_DIR", "eval/results")


def _norm(x, dbfs=-25.0):
    rms = float(np.sqrt(np.mean(np.square(x[np.abs(x) > 1e-4], dtype=np.float64)))) if np.any(np.abs(x) > 1e-4) else 1.0
    return (x * (10 ** (dbfs / 20) / max(rms, 1e-9))).astype(np.float32)


def _turns(n, rng, on=(1.0, 3.0), off=(0.5, 2.0)):
    """0/1 talk-turn envelope with 20 ms ramps."""
    env = np.zeros(n, np.float32)
    t = int(rng.uniform(0, off[1]) * SR)
    while t < n:
        d = int(rng.uniform(*on) * SR)
        env[t : t + d] = 1.0
        t += d + int(rng.uniform(*off) * SR)
    k = int(0.02 * SR)
    return np.convolve(env, np.ones(k) / k, mode="same").astype(np.float32)


def _babble(n, rng):
    from eval import demo_scene as ds
    from eval.fixtures.sources import TINY_DIR

    noise, _ = ds.babble(n + SR * 7, TINY_DIR)
    off = int(rng.integers(0, SR * 7))
    return np.asarray(noise[off : off + n], np.float32)


def training_scenes(seed=336):
    """``[(name, mix, stems{aryan, joseph})]`` built from the two enrollment cuts only."""
    rng = np.random.default_rng(seed)
    j = _norm(_load(FIXTURES / "team/joseph/profile-01.wav"))
    a = _norm(_load(FIXTURES / "team/aryan/profile-01.wav"))
    n = int(SEG_S * SR)
    scenes = []
    for k in range(len(a) // n):
        seg_a = a[k * n : (k + 1) * n]
        seg_j = np.roll(np.resize(j, n), int(rng.integers(0, len(j))))
        ea, ej = _turns(n, rng), _turns(n, rng)
        for r in RATIOS_DB:
            sa, sj = seg_a * ea * 10 ** (r / 20), seg_j * ej
            speech = sa + sj
            noise = _babble(n, rng)
            noise *= np.sqrt(np.mean(speech**2) / max(np.mean(noise**2), 1e-12)) * 10 ** (-SNR_DB / 20)
            scenes.append((f"seg{k}_r{r:+.0f}", speech + noise, {"aryan": sa, "joseph": sj}))
        for who, x in (("aryan", seg_a * ea), ("joseph", seg_j * ej)):  # target-absent for the other voice
            noise = _babble(n, rng)
            noise *= np.sqrt(np.mean(x**2) / max(np.mean(noise**2), 1e-12)) * 10 ** (-SNR_DB / 20)
            stems = {"aryan": x if who == "aryan" else np.zeros(n, np.float32),
                     "joseph": x if who == "joseph" else np.zeros(n, np.float32)}
            scenes.append((f"seg{k}_{who}_only", x + noise, stems))
    return scenes


def _load(path):
    x = load(path, SR)
    x = x[0] if isinstance(x, tuple) else x
    return np.asarray(x, np.float32).reshape(-1)


def candidates():
    """Per target name an ``EnrolledCandidate`` prepared exactly as live (saved profiles, other = negative)."""
    root = FIXTURES / "profiles"
    from hearing.tse.enroll import list_profiles

    names = list_profiles(root)
    pairs = dict(zip(names, profile_pairs(root), strict=True))
    out = {}
    shared = None
    for name in NAMES:
        c = EnrolledCandidate() if shared is None else EnrolledCandidate(shared.separator, shared.embedder)
        c.team_head = None
        c.prepare(*pairs[name])
        shared = shared or c
        out[name] = (c, pairs[name])
    return out


def _component_db(y, t, o):
    """Energies (dBFS RMS) of the target and other stems' least-squares parts in stream ``y``."""
    basis = np.column_stack([t, o]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(basis, y.astype(np.float64), rcond=None)
    db = []
    for c, s in zip(coef, (t, o), strict=True):
        db.append(10 * np.log10(max(float(np.mean((c * s.astype(np.float64)) ** 2)), 1e-12)))
    return db


def window_rows(mix, stems, cands, directions=NAMES):
    """Per window and target direction: features, labels, and the LDA scores of both streams."""
    first = cands[NAMES[0]][0]
    rows = []
    for w0 in range(0, len(mix) - WIN + 1, HOP):
        x = np.ascontiguousarray(mix[w0 : w0 + WIN], np.float32)
        streams = first.separator.separate(x, SR)
        embs = [_unit_embedding(first.embedder, y) for y in streams]
        energy = [float(np.sum(np.square(y, dtype=np.float64))) for y in streams]
        for name in directions:
            other = NAMES[1 - NAMES.index(name)]
            c, (tclip, oclip) = cands[name]
            target, negative = c.target_embedding(tclip), c.other_embedding(oclip)
            w, b = c._closed_set()
            lda = [float(w @ e) - b for e in embs]
            feats = stream_features(lda, [float(target @ e) for e in embs], [float(negative @ e) for e in embs], energy)
            t, o = stems[name][w0 : w0 + WIN], stems[other][w0 : w0 + WIN]
            labels = []
            for y in streams:
                dt, do = _component_db(y, t, o) if (np.any(t) or np.any(o)) else (-120.0, -120.0)
                labels.append(int(dt - do >= DOMINANT_DB and dt >= AUDIBLE_DBFS))
            t_active = 10 * np.log10(max(float(np.mean(np.square(t, dtype=np.float64))), 1e-12)) >= AUDIBLE_DBFS
            rows.append({"t": w0 / SR, "target": name, "x": feats, "y": labels, "lda": lda, "t_active": bool(t_active)})
    return rows


def decisions(rows, head):
    """Per window: (accepted by LDA gate, LDA pick right, accepted by head, head pick right, any stream target)."""
    out = []
    for r in rows:
        lda = np.asarray(r["lda"])
        i = int(np.argmax(lda))
        p = head.prob(r["x"])
        k = int(np.argmax(p))
        out.append((bool(lda[i] >= 0.0), bool(r["y"][i]), bool(p[k] >= head.threshold), bool(r["y"][k]), bool(any(r["y"]))))
    return out


def summarise(rows, head):
    d = decisions(rows, head)
    s = {"windows": len(d), "target_windows": sum(x[4] for x in d)}
    for tag, (acc, ok) in {"lda": (0, 1), "head": (2, 3)}.items():
        s[f"{tag}_accept_right"] = sum(x[acc] and x[ok] for x in d)  # plays a target-dominant stream
        s[f"{tag}_accept_wrong"] = sum(x[acc] and not x[ok] for x in d)  # plays a non-target stream as target
        s[f"{tag}_gated_target"] = sum((not x[acc]) and x[4] for x in d)  # gated though a stream held the target
    return s


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=RESULTS / "T-336_team_head.json")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--features-cache", type=Path, default=RESULTS / "T-336_train_rows.npz")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--targets", nargs="+", default=list(NAMES), choices=NAMES, help="T-141 targets to test")
    args = ap.parse_args(argv)
    t0 = time.time()
    cands = candidates()
    if args.features_cache.is_file():
        z = np.load(args.features_cache)
        x, y = z["x"], z["y"]
    else:
        xs, ys = [], []
        for name, mix, stems in training_scenes():
            for r in window_rows(mix, stems, cands):
                xs.append(r["x"])
                ys.extend(r["y"])
            print(f"train {name}: {len(ys)} streams so far ({time.time() - t0:.0f} s)", flush=True)
        x, y = np.concatenate(xs), np.asarray(ys)
        np.savez(args.features_cache, x=x, y=y)
    mean, scale, coef, intercept = fit_logistic(x, y, l2=args.l2)
    train_src = {n: {"path": f"team/{n}/profile-01.wav", "sha256": sha256(FIXTURES / f"team/{n}/profile-01.wav")} for n in NAMES}
    profiles = {n: {"path": f"profiles/{n}/profile-01.wav", "sha256": sha256(FIXTURES / f"profiles/{n}/profile-01.wav")} for n in NAMES}
    head = TeamHead(mean, scale, coef, intercept, args.threshold, meta={
        "task": "T-336", "label": "tuned to the two teammates' voices, trained on enrollment cuts, tested on disjoint takes",
        "train_audio": train_src, "lda_profiles": profiles, "train_streams": len(y), "train_positive": int(y.sum()),
        "ratios_db": RATIOS_DB, "snr_db": SNR_DB, "dominant_db": DOMINANT_DB, "features": FEATURES,
    })
    p = head.prob(x)
    train = {"streams": len(y), "positive": int(y.sum()),
             "acc": float(np.mean((p >= head.threshold) == (y == 1))),
             "false_accept": float(np.mean(p[y == 0] >= head.threshold)),
             "lda_false_accept": float(np.mean(x[y == 0, 0] >= 0)), "lda_acc": float(np.mean((x[:, 0] >= 0) == (y == 1)))}
    head.meta["train"] = train
    args.out.parent.mkdir(parents=True, exist_ok=True)
    head.save(args.out)
    print(json.dumps({"train": train, "coef": dict(zip(FEATURES, np.round(coef, 3).tolist(), strict=True)),
                      "weights": str(args.out), "sha256": sha256(args.out)}, indent=1), flush=True)
    if args.skip_test:
        return 0
    report = {"weights_sha256": sha256(args.out), "scenes": {}}
    for target in args.targets:
        other = NAMES[1 - NAMES.index(target)]
        for scene in ("demo-team-easy", "demo-team-hard"):
            d = FIXTURES / "t141" / target / scene / "stems"
            tgt, oth, noise, mix = (_load(d / f"{k}.wav") for k in ("target", "other", "noise", "mix"))
            n = min(map(len, (tgt, oth, noise, mix)))
            stems = {target: tgt[:n], other: oth[:n]}
            s = summarise(window_rows(mix[:n], stems, cands, (target,)), head)
            absent_stems = {target: np.zeros(n, np.float32), other: oth[:n]}
            a = summarise(window_rows((oth[:n] + noise[:n]).astype(np.float32), absent_stems, cands, (target,)), head)
            absent = {"lda_accept_s": (a["lda_accept_right"] + a["lda_accept_wrong"]) * HOP / SR,
                      "head_accept_s": (a["head_accept_right"] + a["head_accept_wrong"]) * HOP / SR, "windows": a["windows"]}
            report["scenes"][f"{target}/{scene}"] = {"mix": s, "target_absent": absent}
            print(target, scene, json.dumps(s), "absent", json.dumps(absent), flush=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    out = RESULTS / f"T-336_offline_test_{'-'.join(args.targets)}_{stamp}.json"
    out.write_text(json.dumps(report, indent=1))
    print("offline test:", out, f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
