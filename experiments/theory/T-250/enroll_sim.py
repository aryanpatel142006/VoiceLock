"""T-250 synthetic embedding prototype for active enrollment / profile maintenance.

Pure numpy, offline, no audio, no models. Embeddings are unit vectors:
speaker = unit centre; clip = centre + within-speaker noise + optional
channel (mic) offset; outliers = another speaker's clip (noisy enrolment).
Everything is synthetic: results show *mechanism behaviour on a toy
generative model*, not real ECAPA accuracy.

CLI: python experiments/theory/T-250/enroll_sim.py --seed 0 --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

DIM = 64
MAX_CASES = 2000  # hard cap per run
CONFIG = {
    "dim": DIM, "n_speakers": 12, "within_sigma": 0.9, "mic_offset": 1.5,
    "outlier_rate": 0.25, "drift_per_step": 0.08, "n_trials": 40,
    "clip_counts": [1, 2, 3, 5, 8], "open_set_impostors": 6, "e6_sigma": 2.5, "version": 2,
}


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def clip(rng, centre, sigma, offset=None):
    e = centre + sigma * rng.normal(size=centre.shape) / np.sqrt(DIM)
    if offset is not None:
        e = e + offset
    return unit(e)


# ---- aggregation -----------------------------------------------------
def agg_mean(E, **_):
    return unit(E.mean(0))


def agg_robust(E, keep_frac=0.7, **_):
    """Reject clips with lowest cosine to the leave-one-out centroid, then mean.
    Needs >=3 clips; with fewer it falls back to the plain mean (cannot judge)."""
    n = len(E)
    if n < 3:
        return unit(E.mean(0))
    loo = np.array([unit(np.delete(E, i, 0).mean(0)) @ E[i] for i in range(n)])
    k = max(2, int(np.ceil(keep_frac * n)))
    return unit(E[np.argsort(-loo)[:k]].mean(0))


def uncertainty_pick(E, k):
    """Confirmed-pool selection: pick k clips maximising spread (farthest-point
    after the medoid) so the profile covers variation; users confirm only these."""
    c = unit(E.mean(0))
    idx = [int(np.argmax(E @ c))]
    while len(idx) < min(k, len(E)):
        d = np.max(E @ E[idx].T, axis=1)
        d[idx] = np.inf
        idx.append(int(np.argmin(d)))
    return idx


def score(profile, X, neg=None, lam=0.5):
    s = X @ profile
    if neg is not None:
        s = s - lam * np.clip(X @ neg, 0, None)
    return s


def run(seed: int, max_seconds: float = 60.0):
    rng = np.random.default_rng(seed)
    c = CONFIG
    t0 = time.time()
    centres = unit(rng.normal(size=(c["n_speakers"], DIM)))
    out = {"seed": seed, "config": c, "experiments": {}}
    cases = 0

    # E1: clip count x outlier x aggregation (same mic)
    e1 = {}
    for n in c["clip_counts"]:
        for name, f in (("mean", agg_mean), ("robust", agg_robust)):
            accs, aucs = [], []
            for t in range(c["n_trials"]):
                cases += 1
                if cases > MAX_CASES or time.time() - t0 > max_seconds:
                    out["status"] = "stopped_cap"; return out
                s = t % c["n_speakers"]
                clips = np.stack([clip(rng, centres[s], c["within_sigma"]) for _ in range(n)])
                bad = rng.random(n) < c["outlier_rate"]
                for i in np.where(bad)[0]:
                    o = (s + 1 + rng.integers(c["n_speakers"] - 1)) % c["n_speakers"]
                    clips[i] = clip(rng, centres[o], c["within_sigma"])
                p = f(clips)
                pos = clip(rng, np.repeat(centres[s][None], 200, 0), c["within_sigma"]) @ p
                oth = np.delete(np.arange(c["n_speakers"]), s)
                neg = clip(rng, np.repeat(centres[oth], 20, 0), c["within_sigma"]) @ p
                aucs.append(float((pos[:, None] > neg[None, :]).mean()))
            e1[f"{name}_n{n}"] = float(np.mean(aucs))
    out["experiments"]["E1_auc_by_clipcount_outlier25pct"] = e1

    # E2: cross-mic mismatch, enroll on mic A, test on mic B; fix = add 1 confirmed clip from B
    e2 = {"enroll_A_test_B": [], "enroll_A+1B_test_B": [], "enroll_A_test_A": []}
    for t in range(c["n_trials"]):
        cases += 1
        s = t % c["n_speakers"]
        offA, offB = (unit(rng.normal(size=DIM)) * c["mic_offset"] for _ in range(2))
        mk = lambda n, off: np.stack([clip(rng, centres[s], c["within_sigma"], off) for _ in range(n)])
        mkimp = lambda n, off: np.stack([clip(rng, centres[(s + 1 + i) % c["n_speakers"]], c["within_sigma"], off) for i in range(n)])
        A = mk(4, offA)
        for key, prof, off in (("enroll_A_test_B", agg_mean(A), offB),
                               ("enroll_A+1B_test_B", agg_mean(np.vstack([A, mk(1, offB)])), offB),
                               ("enroll_A_test_A", agg_mean(A), offA)):
            pos = mk(100, off) @ prof
            neg = mkimp(100, off) @ prof
            e2[key].append(float((pos[:, None] > neg[None, :]).mean()))
    out["experiments"]["E2_cross_mic_auc"] = {k: float(np.mean(v)) for k, v in e2.items()}

    # E3: drift. Speaker centre rotates; static profile vs confirmed-only EMA update
    # vs unconfirmed self-training (accept anything with score>thr incl. impostors).
    e3 = {"static": [], "confirmed_ema": [], "self_train": []}
    for t in range(c["n_trials"]):
        cases += 1
        s = t % c["n_speakers"]
        cen = centres[s].copy()
        base = agg_mean(np.stack([clip(rng, cen, c["within_sigma"]) for _ in range(4)]))
        pr = {"static": base, "confirmed_ema": base.copy(), "self_train": base.copy()}
        for step in range(10):
            cen = unit(cen + c["drift_per_step"] * unit(rng.normal(size=DIM)))
            xs = clip(rng, np.repeat(cen[None], 30, 0), c["within_sigma"])
            # 20% of stream is an impostor close-ish to target (hard case)
            imp = clip(rng, np.repeat(unit(centres[s] + 0.4 * unit(rng.normal(size=DIM)))[None], 20, 0), c["within_sigma"])
            for k in ("confirmed_ema", "self_train"):
                p = pr[k]
                if k == "confirmed_ema":
                    # user confirms 2 clean target clips per step (ground truth stands for the human)
                    upd = xs[:2].mean(0)
                else:
                    pool = np.vstack([xs, imp])
                    sc = pool @ p
                    acc = pool[sc > np.quantile(sc, 0.5)]
                    upd = acc.mean(0)
                pr[k] = unit(0.8 * p + 0.2 * unit(upd))
        final = clip(rng, np.repeat(cen[None], 100, 0), c["within_sigma"])
        impf = clip(rng, np.repeat(unit(centres[s] + 0.4 * unit(rng.normal(size=DIM)))[None], 100, 0), c["within_sigma"])
        for k, p in pr.items():
            e3[k].append(float(((final @ p)[:, None] > (impf @ p)[None, :]).mean()))
    out["experiments"]["E3_drift_auc_after_10_steps"] = {k: float(np.mean(v)) for k, v in e3.items()}

    # E4: open-set. Closed-set argmax always picks someone; open-set adds a
    # threshold calibrated on enrolment-time clips only. Report false-accept of
    # unknown speakers and true-accept of target.
    e4 = {"closed_set_false_accept": [], "open_set_false_accept": [], "open_set_true_accept": [], "closed_set_true_accept": []}
    for t in range(c["n_trials"]):
        cases += 1
        enrolled = rng.choice(c["n_speakers"] - c["open_set_impostors"], 3, replace=False)
        profs = np.stack([agg_mean(np.stack([clip(rng, centres[e], c["within_sigma"]) for _ in range(5)])) for e in enrolled])
        # threshold: 5th percentile of genuine scores on 5 held-out confirmed clips (target-only evidence)
        gen = np.concatenate([np.stack([clip(rng, centres[e], c["within_sigma"]) for _ in range(5)]) @ profs[i] for i, e in enumerate(enrolled)])
        thr = float(np.quantile(gen, 0.05))
        unk = np.arange(c["n_speakers"] - c["open_set_impostors"], c["n_speakers"])
        ux = np.stack([clip(rng, centres[u], c["within_sigma"]) for u in unk for _ in range(20)])
        tx = np.stack([clip(rng, centres[enrolled[0]], c["within_sigma"]) for _ in range(100)])
        e4["closed_set_false_accept"].append(1.0)  # argmax always assigns an unknown to some enrolled id
        e4["open_set_false_accept"].append(float(((ux @ profs.T).max(1) > thr).mean()))
        e4["open_set_true_accept"].append(float(((tx @ profs.T).max(1) > thr).mean()))
        e4["closed_set_true_accept"].append(float(((tx @ profs.T).argmax(1) == 0).mean()))
    out["experiments"]["E4_open_vs_closed_set"] = {k: float(np.mean(v)) for k, v in e4.items()}

    # E5: negative profile (competing talker) effect on false accept at matched true accept
    e5 = {"no_neg": [], "with_neg": []}
    for t in range(c["n_trials"]):
        cases += 1
        s, o = t % c["n_speakers"], (t + 1) % c["n_speakers"]
        cen_o = unit(centres[o] + 2.0 * centres[s])  # similar interferer
        p = agg_mean(np.stack([clip(rng, centres[s], c["within_sigma"]) for _ in range(5)]))
        ng = agg_mean(np.stack([clip(rng, cen_o, c["within_sigma"]) for _ in range(3)]))
        pos = clip(rng, np.repeat(centres[s][None], 200, 0), c["within_sigma"])
        imp = clip(rng, np.repeat(cen_o[None], 200, 0), c["within_sigma"])
        for k, n_ in (("no_neg", None), ("with_neg", ng)):
            sp, si = score(p, pos, n_), score(p, imp, n_)
            e5[k].append(float((sp[:, None] > si[None, :]).mean()))
    out["experiments"]["E5_negative_profile_auc_similar_interferer"] = {k: float(np.mean(v)) for k, v in e5.items()}

    # E6: selection by uncertainty vs random for a budget of k confirmed clips (pool of 12)
    e6 = {"random_k3": [], "spread_k3": []}
    for t in range(c["n_trials"]):
        cases += 1
        s = t % c["n_speakers"]
        pool = np.stack([clip(rng, centres[s], c["e6_sigma"]) for _ in range(12)])
        test = clip(rng, np.repeat(centres[s][None], 100, 0), c["e6_sigma"])
        imp = clip(rng, np.repeat(centres[(s + 1) % c["n_speakers"]][None], 100, 0), c["e6_sigma"])
        for k, idx in (("random_k3", rng.choice(12, 3, replace=False)), ("spread_k3", uncertainty_pick(pool, 3))):
            p = agg_mean(pool[list(idx)])
            e6[k].append(float(((test @ p)[:, None] > (imp @ p)[None, :]).mean()))
    out["experiments"]["E6_selection_policy_auc"] = {k: float(np.mean(v)) for k, v in e6.items()}
    out["cases"] = cases
    out["status"] = "complete"
    return out


# ---- versioned store (identity update / revert / cached swap) ---------
class ProfileStore:
    """In-memory model of the local profile store: immutable versions, a
    `current` pointer, O(1) revert and swap (pre-normalised vectors cached)."""

    def __init__(self):
        self.versions: dict[str, list[np.ndarray]] = {}
        self.cur: dict[str, int] = {}
        self.cache: dict[str, np.ndarray] = {}

    def add_version(self, name, emb, confirmed: bool):
        if not confirmed:
            raise ValueError("only explicitly user-confirmed clean clips may update a profile")
        self.versions.setdefault(name, []).append(unit(emb))
        self.cur[name] = len(self.versions[name]) - 1
        self.cache[name] = self.versions[name][-1]

    def revert(self, name):
        if self.cur[name] == 0:
            return False
        self.cur[name] -= 1
        self.cache[name] = self.versions[name][self.cur[name]]
        return True

    def swap(self, name):
        return self.cache[name]  # no recompute


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=60.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.out and a.out.exists():
        prev = json.loads(a.out.read_text())
        if prev.get("seed") == a.seed and prev.get("status") == "complete" and prev.get("script_sha") == script_sha():
            print("resume: verified complete, skipping"); return 0
    res = run(a.seed, a.max_seconds)
    res["script_sha"] = script_sha()
    res["config_sha"] = hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]
    txt = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = a.out.with_suffix(".tmp"); tmp.write_text(txt); tmp.replace(a.out)
    else:
        print(txt)
    return 0 if res["status"] == "complete" else 2


def script_sha():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


if __name__ == "__main__":
    sys.exit(main())
