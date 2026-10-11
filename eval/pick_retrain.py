"""T-375: retrain the closed-set aryan-vs-joseph pick on separated crops of ALL 12 min, with quality features.

Offline, CPU-side after the one separator pass (MossFormer2 via ``EnrolledCandidate``; windows 1.0 s / 0.5 s hop,
as live). Nothing here changes ``hearing/``; the knob is the weights file plus ``--features quality`` and stays
default off (live uses ``HEARING_TEAM_HEAD`` = T-368's 5-feature head, or the LDA gate).

Stages (each caches under ``$HEARING_RESULTS_DIR/T-375_*``):
  ``extract``  separate + embed every window of (a) the six 2-min training takes profile-02..07, mixed as
               aryan:joseph +5/0/-5 dB, aryan-only, joseph-only 24 s chunks over babble at 5 dB SNR, and
               (b) the test scenes: the eight T-368 held-out scenes (holdout-01/02 takes) and same-passage mixes of
               same-01/02. Per window and direction it stores the stream embeddings, energies, voiced seconds
               and the oracle labels (target stem dominates the stream by >= 3 dB).
  ``eval``     split by take (fit profile-02..06, validate 07, test holdout + same only), fit heads, score.
  ``augment``  enrollment-speech augmentation (arXiv 2409.09589): noise/RIR variants of the profile pieces,
               ECAPA prototypes + LDA recomputed, margin and false accepts on the test windows.

    .venv/bin/python -m eval.pick_retrain extract|eval|augment
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np

from eval import train_team_head as th
from hearing.tse.enrolled import _unit_embedding, fit_lda
from hearing.tse.team_head import FEATURES, TeamHead, fit_logistic, stream_features

SR, WIN, HOP, NAMES = th.SR, th.WIN, th.HOP, th.NAMES
FIX, RES = th.FIXTURES, th.RESULTS
TRAIN_TAKES = tuple(f"profile-0{k}.wav" for k in range(2, 7))
VAL_TAKES = ("profile-07.wav",)
PROFILE_CUT_S = (20.0, 30.0)  # profiles-v2/v2 pieces come from here: no training window may overlap it
CHUNK_S = 24.0
QUALITY = ("dur_s", "mix_db", "coh_mean", "cos_t_z", "cos_o_z")
Q_FEATURES = tuple(FEATURES) + QUALITY
TOP_K = 10
PROFILES = FIX / "profiles-v2"
CACHE = RES / "T-375_windows.pkl"


# --------------------------------------------------------------------------- features
def voiced_s(y, floor_db=-55.0, rel_db=-30.0):
    """Seconds of the stream whose 20 ms frames sit above ``floor_db`` dBFS and within ``rel_db`` of its loudest."""
    n = len(y) // 320
    if n == 0:
        return 0.0
    f = 10 * np.log10(np.maximum(np.mean(np.square(np.asarray(y[: n * 320], np.float64).reshape(n, 320)), axis=1), 1e-12))
    return float(np.sum(f >= max(floor_db, f.max() + rel_db)) * 0.02)


def cohort_embeddings(embedder, n_spk=40):
    """Unit ECAPA embeddings of 1 s crops of ``n_spk`` LibriSpeech test-clean speakers (one utterance each)."""
    path = RES / "T-375_cohort.npz"
    if path.is_file():
        return np.load(path)["emb"]

    tgz = FIX / "public_voice/test-clean.tar.gz"
    out, seen = [], set()
    with tempfile.TemporaryDirectory(prefix="T-375-coh-") as tmp, tarfile.open(tgz) as tar:
        for m in tar:
            if not m.name.endswith(".flac"):
                continue
            spk = m.name.split("/")[2]
            if spk in seen:
                continue
            seen.add(spk)
            src, dst = Path(tmp) / "a.flac", Path(tmp) / "a.wav"
            src.write_bytes(tar.extractfile(m).read())
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)], check=True)
            x = th._load(dst)
            for s in range(0, max(len(x) - SR, 1), SR * 2)[:4]:
                out.append(_unit_embedding(embedder, x[s : s + SR]))
            if len(seen) >= n_spk:
                break
    emb = np.stack(out)
    np.savez(path, emb=emb)
    return emb


def quality_columns(embs, cos_t, cos_o, cohort, dur, mix_db):
    sims = np.asarray(embs) @ cohort.T  # (streams, cohort)
    top = np.sort(sims, axis=1)[:, -TOP_K:].mean(axis=1)
    mu, sd = sims.mean(axis=1), np.maximum(sims.std(axis=1), 1e-3)
    return np.column_stack([dur, mix_db, top, (np.asarray(cos_t) - mu) / sd, (np.asarray(cos_o) - mu) / sd])


def extract_windows(mix, stems, cands, cohort, t_lo=0.0, skip=None):
    """Per window: stream embeddings, features for both directions, labels. ``skip`` = (a, b) seconds to drop."""
    first = cands[NAMES[0]][0]
    rows = []
    mix_e = None
    for w0 in range(0, len(mix) - WIN + 1, HOP):
        if skip and w0 / SR < skip[1] and (w0 + WIN) / SR > skip[0]:
            continue
        x = np.ascontiguousarray(mix[w0 : w0 + WIN], np.float32)
        streams = first.separator.separate(x, SR)
        embs = np.stack([_unit_embedding(first.embedder, y) for y in streams])
        energy = [float(np.sum(np.square(y, dtype=np.float64))) for y in streams]
        mix_e = float(np.sum(np.square(x, dtype=np.float64))) + 1e-12
        dur = [voiced_s(y) for y in streams]
        mix_db = [10 * np.log10(max(e, 1e-12) / mix_e) for e in energy]
        per = {}
        for name in NAMES:
            other = NAMES[1 - NAMES.index(name)]
            c, (tclip, oclip) = cands[name]
            target, negative = c.target_embedding(tclip), c.other_embedding(oclip)
            w, b = c._closed_set()
            lda = [float(w @ e) - b for e in embs]
            ct, co = [float(target @ e) for e in embs], [float(negative @ e) for e in embs]
            base = stream_features(lda, ct, co, energy)
            q = quality_columns(embs, ct, co, cohort, dur, mix_db)
            t, o = stems[name][w0 : w0 + WIN], stems[other][w0 : w0 + WIN]
            labels = []
            for y in streams:
                dt, do = th._component_db(y, t, o) if (np.any(t) or np.any(o)) else (-120.0, -120.0)
                labels.append(int(dt - do >= th.DOMINANT_DB and dt >= th.AUDIBLE_DBFS))
            t_act = 10 * np.log10(max(float(np.mean(np.square(t, dtype=np.float64))), 1e-12)) >= th.AUDIBLE_DBFS
            per[name] = {"x": np.column_stack([base, q]), "y": labels, "lda": lda, "t_active": bool(t_act)}
        rows.append({"t": t_lo + w0 / SR, "embs": embs, "per": per})
    return rows


def training_take(take, seed, ratios=(5.0, 0.0, -5.0)):
    """One 120 s scene from ``team/{aryan,joseph}/<take>``: 24 s chunks at ``ratios`` dB, aryan-only, joseph-only."""
    rng = np.random.default_rng(seed)
    a, j = th._norm(th._load(FIX / "team/aryan" / take)), th._norm(th._load(FIX / "team/joseph" / take))
    n = min(len(a), len(j))
    a, j = a[:n], j[:n]
    ea, ej = th._turns(n, rng), th._turns(n, rng)
    sa, sj = a * ea, j * ej
    g = np.ones(n, np.float32)
    ja = np.ones(n, np.float32)
    chunk = int(CHUNK_S * SR)
    plan = [("r", r) for r in ratios] + [("a", 0), ("j", 0)]
    for k, (kind, r) in enumerate(plan):
        s = slice(k * chunk, min((k + 1) * chunk, n))
        if kind == "r":
            g[s] = 10 ** (r / 20)
        elif kind == "a":
            ja[s] = 0.0
        else:
            g[s] = 0.0
    # pairs of chunks: aryan-only silences joseph, joseph-only silences aryan
    sa = sa * g
    sj = sj * ja
    speech = sa + sj
    noise = th._babble(n, rng)
    noise *= np.sqrt(np.mean(speech**2) / max(np.mean(noise**2), 1e-12)) * 10 ** (-th.SNR_DB / 20)
    return (speech + noise).astype(np.float32), {"aryan": sa.astype(np.float32), "joseph": sj.astype(np.float32)}


def same_scene(target, other, take, sir_db, seed):
    """Same-passage take pair (both read the same passage) mixed with turns over babble at SNR 10 dB."""
    rng = np.random.default_rng(seed)
    t, o = th._norm(th._load(FIX / "team" / target / take)), th._norm(th._load(FIX / "team" / other / take))
    n = min(len(t), len(o))
    st = (t[:n] * th._turns(n, rng, on=(2.0, 5.0), off=(0.3, 1.0))).astype(np.float32)
    so = (o[:n] * th._turns(n, rng, on=(2.0, 5.0), off=(0.3, 1.0)) * 10 ** (-sir_db / 20)).astype(np.float32)
    speech = st + so
    noise = th._babble(n, rng)
    noise *= np.sqrt(np.mean(speech**2) / max(np.mean(noise**2), 1e-12)) * 10 ** (-10.0 / 20)
    return (speech + noise).astype(np.float32), {target: st, other: so}


def candidates(root=PROFILES):
    from eval.oct10_reenroll import candidates as c

    return c(root)


def cmd_extract(args):
    cache = pickle.loads(CACHE.read_bytes()) if CACHE.is_file() else {}
    cands = candidates()
    cohort = cohort_embeddings(cands["aryan"][0].embedder)
    t0 = time.time()

    def run(key, build, **kw):
        if key in cache:
            return
        mix, stems = build()
        cache[key] = extract_windows(mix, stems, cands, cohort, **kw)
        CACHE.write_bytes(pickle.dumps(cache))
        print(f"{key}: {len(cache[key])} windows ({time.time() - t0:.0f} s)", flush=True)

    for k, take in enumerate(TRAIN_TAKES + VAL_TAKES):
        run(f"train/{take}", lambda take=take, k=k: training_take(take, 375 + k), skip=PROFILE_CUT_S)
    root = FIX / "t368"
    for tgt in NAMES:
        for scene in sorted(p.name for p in (root / tgt).iterdir() if p.is_dir()):
            def build(tgt=tgt, scene=scene):
                d = root / tgt / scene / "stems"
                st = {k: th._load(d / f"{k}.wav") for k in ("target", "other", "mix")}
                n = min(map(len, st.values()))
                oth = NAMES[1 - NAMES.index(tgt)]
                return st["mix"][:n], {tgt: st["target"][:n], oth: st["other"][:n]}
            run(f"test/holdout/{tgt}/{scene}", build)
    for tgt in NAMES:
        oth = NAMES[1 - NAMES.index(tgt)]
        for take in ("same-01.wav", "same-02.wav"):
            for sir in (0.0, -6.0):
                run(f"test/same/{tgt}/{take}@{sir:+.0f}", lambda tgt=tgt, oth=oth, take=take, sir=sir: same_scene(tgt, oth, take, sir, 3750 + int(sir)))
    return 0


# --------------------------------------------------------------------------- scoring
def stack(cache, prefix):
    xs, ys = [], []
    for key, rows in cache.items():
        if key.startswith(prefix):
            for r in rows:
                for name in NAMES:
                    p = r["per"][name]
                    xs.append(p["x"])
                    ys.extend(p["y"])
    return np.concatenate(xs), np.asarray(ys)


def fit_head(x, y, cols, l2, threshold=0.5):
    mean, scale, coef, b = fit_logistic(x[:, cols], y, l2=l2)
    return mean, scale, coef, b, threshold


def head_prob(h, x, cols):
    mean, scale, coef, b, _ = h
    z = ((np.asarray(x)[:, cols] - mean) / scale) @ coef + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def window_outcomes(rows, scorer):
    """Per window and direction ``(any_target, accepted, right, t_active)``; ``scorer(x) -> (scores, accept_thr)``."""
    out = []
    for r in rows:
        for name in NAMES:
            p = r["per"][name]
            s, thr = scorer(p)
            k = int(np.argmax(s))
            out.append((any(p["y"]), bool(s[k] >= thr), bool(p["y"][k]), p["t_active"], name))
    return out


def metrics(out):
    o = np.array([(a, b, c, d) for a, b, c, d, _ in out], bool)
    anyt, acc, right, tact = o.T
    return {"n": len(o), "pick": acc & right & anyt, "wrong": acc & ~right, "absent_acc": acc & ~tact, "absent": ~tact,
            "anyt": anyt}


def boot(num, den, rng, block=10, b=2000):
    """Mean of ``num`` over ``den`` rows (bool arrays) with a block-bootstrap 95 % interval."""
    n = len(num)
    nb = max(n // block, 1)
    idx = np.arange(nb * block).reshape(nb, block)
    vals = []
    for _ in range(b):
        s = idx[rng.integers(0, nb, nb)].ravel()
        d = den[s].sum()
        vals.append(num[s].sum() / d if d else np.nan)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(num.sum() / max(den.sum(), 1)), float(lo), float(hi), int(num.sum()), int(den.sum())


def fmt(t, pct=True):
    v, lo, hi, k, n = t
    return f"{100 * v:.1f} % [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{n})"


def cmd_eval(args):
    cache = pickle.loads(CACHE.read_bytes())
    rng = np.random.default_rng(375)
    xtr, ytr = stack(cache, "train/profile-0")
    tr_keys = [k for k in cache if k.startswith("train/") and k.split("/")[1] in TRAIN_TAKES]
    va_keys = [k for k in cache if k.startswith("train/") and k.split("/")[1] in VAL_TAKES]

    def arr(keys):
        xs, ys = [], []
        for k in keys:
            for r in cache[k]:
                for nm in NAMES:
                    xs.append(r["per"][nm]["x"])
                    ys.extend(r["per"][nm]["y"])
        return np.concatenate(xs), np.asarray(ys)

    xtr, ytr = arr(tr_keys)
    xva, yva = arr(va_keys)
    base_cols = list(range(5))
    all_cols = list(range(len(Q_FEATURES)))
    heads = {}
    for tag, cols in (("retrain5", base_cols), ("quality8", all_cols)):
        best = None
        for l2 in (0.1, 1.0, 10.0, 100.0):
            h = fit_head(xtr, ytr, cols, l2)
            p = np.clip(head_prob(h, xva, cols), 1e-6, 1 - 1e-6)
            ll = float(-np.mean(yva * np.log(p) + (1 - yva) * np.log(1 - p)))
            if best is None or ll < best[0]:
                best = (ll, l2, h)
        heads[tag] = (best[2], cols)
        print(f"{tag}: l2={best[1]} val logloss {best[0]:.4f}  train n={len(ytr)} pos={int(ytr.sum())} val n={len(yva)} pos={int(yva.sum())}", flush=True)
    t368 = RES / "T-368_team_head_v3.json"
    t368_head = TeamHead.load(t368) if t368.is_file() else None

    def lda_scorer(p):
        return np.asarray(p["lda"]), 0.0

    def mk(h, cols):
        return lambda p: (head_prob(h, p["x"], cols), h[4])

    scorers = {"lda": lda_scorer}
    if t368_head is not None:
        scorers["T-368 head v3"] = lambda p: (t368_head.prob(p["x"][:, :5]), t368_head.threshold)
    for tag, (h, cols) in heads.items():
        scorers[tag] = mk(h, cols)
    groups = {"val (profile-07)": va_keys}
    for tgt in NAMES:
        for sir in ("sir0", "sir-6"):
            groups[f"holdout {tgt}:{NAMES[1 - NAMES.index(tgt)]} {sir}"] = [k for k in cache if k.startswith(f"test/holdout/{tgt}/holdout-{sir}-")]
    groups["same-passage (all)"] = [k for k in cache if k.startswith("test/same/")]
    groups["holdout (all 8 scenes)"] = [k for k in cache if k.startswith("test/holdout/")]
    report = {}
    lines = ["| row | pick | " + " | ".join(scorers) + " |", "|---|---|" + "---|" * len(scorers)]
    for g, keys in groups.items():
        rows = [r for k in keys for r in cache[k]]
        res = {}
        for sc, f in scorers.items():
            m = metrics(window_outcomes(rows, f))
            res[sc] = {
                "pick_acc": boot(m["pick"], m["anyt"], rng),  # of windows where a target-dominant stream exists
                "wrong_speaker": boot(m["wrong"], np.ones_like(m["wrong"]), rng),
                "absent_accept": boot(m["absent_acc"], m["absent"], rng),
            }
        report[g] = res
        for metric, label in (("pick_acc", "pick accuracy"), ("wrong_speaker", "wrong-speaker"), ("absent_accept", "target-absent accept")):
            lines.append(f"| {g} | {label} | " + " | ".join(fmt(res[sc][metric]) for sc in scorers) + " |")
    print("\n".join(lines))
    out = RES / "T-375_eval.json"
    out.write_text(json.dumps(report, indent=1))
    for tag, (h, cols) in heads.items():
        mean, scale, coef, b, thr = h
        print(tag, dict(zip([Q_FEATURES[c] for c in cols], np.round(coef, 3).tolist(), strict=True)), "intercept", round(b, 3))
        th_json = {"features": [Q_FEATURES[c] for c in cols], "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
                   "intercept": b, "threshold": thr, "meta": {"task": "T-375", "train_takes": list(TRAIN_TAKES)}}
        (RES / f"T-375_head_{tag}.json").write_text(json.dumps(th_json, indent=1))
    print("wrote", out)
    return 0


# --------------------------------------------------------------------------- enrollment augmentation
def augmented_variants(pieces, rng, n_each=2):
    """Noise / synthetic-RIR variants of profile pieces (arXiv 2409.09589 style)."""
    outs = []
    for p in pieces:
        p = np.asarray(p, np.float32)
        for _ in range(n_each):
            snr = rng.uniform(5, 20)
            nz = th._babble(len(p), rng)
            nz *= np.sqrt(np.mean(p**2) / max(np.mean(nz**2), 1e-12)) * 10 ** (-snr / 20)
            outs.append(p + nz)
            rt = rng.uniform(0.15, 0.5)
            L = int(rt * SR)
            ir = rng.standard_normal(L) * np.exp(-6.9 * np.arange(L) / L)
            ir[0] = 1.0
            y = np.convolve(p, ir / np.linalg.norm(ir) * 1.0)[: len(p)]
            outs.append((y * np.sqrt(np.mean(p**2) / max(np.mean(y**2), 1e-12))).astype(np.float32))
    return outs


def cmd_augment(args):
    from hearing.tse.enroll import profile_pieces

    cache = pickle.loads(CACHE.read_bytes())
    cands = candidates()
    emb = cands["aryan"][0].embedder
    rng = np.random.default_rng(2409)
    pieces = {n: profile_pieces(cands[n][1][0]) for n in NAMES}
    units = {n: np.stack([_unit_embedding(emb, p) for p in pieces[n]]) for n in NAMES}
    aug = {n: np.stack([_unit_embedding(emb, p) for p in augmented_variants(pieces[n], rng)]) for n in NAMES}
    print({n: (len(pieces[n]), len(aug[n])) for n in NAMES}, flush=True)
    rows = {k: v for k, v in cache.items() if k.startswith("test/")}

    def protos(u_t, u_o):
        wb = fit_lda(u_t, u_o)
        return wb

    res = {}
    for tag, uu in (("baseline", units), ("augmented", {n: np.concatenate([units[n], aug[n]]) for n in NAMES})):
        marg, wrong, absent_acc, n_abs, n_win = [], 0, 0, 0, 0
        for name in NAMES:
            other = NAMES[1 - NAMES.index(name)]
            w, b = protos(uu[name], uu[other])
            for rs in rows.values():
                for r in rs:
                    p = r["per"][name]
                    lda = [float(w @ e) - b for e in r["embs"]]
                    y = p["y"]
                    i = int(np.argmax(lda))
                    n_win += 1
                    if any(y) and not all(y):
                        t = next(lda[j] for j in range(len(y)) if y[j])
                        o = next(lda[j] for j in range(len(y)) if not y[j])
                        marg.append(t - o)
                    if lda[i] >= 0 and not y[i]:
                        wrong += 1
                    if not p["t_active"]:
                        n_abs += 1
                        absent_acc += int(lda[i] >= 0)
        res[tag] = {"mean_margin": float(np.mean(marg)), "windows_with_margin": len(marg), "wrong_accepts": wrong,
                    "target_absent_accepts": absent_acc, "target_absent_windows": n_abs, "windows": n_win}
        print(tag, json.dumps(res[tag]), flush=True)
    (RES / "T-375_augment.json").write_text(json.dumps(res, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("extract", "eval", "augment"))
    args = ap.parse_args(argv)
    return {"extract": cmd_extract, "eval": cmd_eval, "augment": cmd_augment}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
