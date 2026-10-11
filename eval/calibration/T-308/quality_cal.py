"""T-308: quality-aware calibration of speaker-verification scores (Thienpondt et al., ICASSP 2021).

Replaces one fixed cosine threshold (``hearing.tse.enrolled.SAME_VOICE = 0.7``) with a logistic
regression on (cosine score, window duration, raw embedding norm) -> P(same speaker), so the
score needed for a given precision moves with window quality.

``fit(scores, durations, norms, labels)`` returns a ``QualityCalibrator``;
``cal.predict_threshold(duration, norm, precision=0.95)`` gives the cosine above which every
pair has calibrated P(same) >= ``precision`` (so precision over the accepted pairs is >= it when the
model is calibrated and the base rate matches ``prior``).

    .venv/bin/python eval/calibration/T-308/quality_cal.py            # synthetic comparison table
    .venv/bin/python eval/calibration/T-308/quality_cal.py --team     # + real teammate pairs (ECAPA)

numpy only (no sklearn): IRLS with a small L2 penalty. No network, no weights unless ``--team``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FIXED = 0.7  # hearing.tse.enrolled.SAME_VOICE
PRECISION = 0.95
SHORT_S, LONG_S = 0.5, 2.0
SR = 16_000


def features(scores, durations, norms):
    """Design matrix: [1, s, log d, log n, s*log d, s*log n] (quality enters as logs)."""
    s = np.asarray(scores, np.float64).reshape(-1)
    ld = np.log(np.maximum(np.asarray(durations, np.float64).reshape(-1), 1e-3))
    ln = np.log(np.maximum(np.asarray(norms, np.float64).reshape(-1), 1e-6))
    ld, ln = np.broadcast_to(ld, s.shape), np.broadcast_to(ln, s.shape)
    return np.stack([np.ones_like(s), s, ld, ln, s * ld, s * ln], axis=1)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


@dataclass
class QualityCalibrator:
    weights: np.ndarray  # one per column of ``features``
    prior: float = 0.5  # P(same) in the training pairs
    train_p: np.ndarray = field(default_factory=lambda: np.zeros(0))  # fitted P(same) of the training pairs
    train_y: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def posterior_cut(self, precision=PRECISION):
        """P(same) level whose accepted training pairs (P >= level) have ``precision`` (best recall).

        One cut on the calibrated output for every window, as in Thienpondt et al.; without
        training pairs it falls back to ``precision`` itself (every accepted pair >= ``precision``).
        """
        if not self.train_p.size:
            return precision
        return min(oracle_threshold(self.train_p, self.train_y, precision), 1 - 1e-12)

    def predict_proba(self, scores, durations, norms, prior=None):
        """Calibrated P(same speaker); ``prior`` re-targets the base rate (logit shift)."""
        z = features(scores, durations, norms) @ self.weights + self._shift(prior)
        return _sigmoid(z)

    def _shift(self, prior):
        if prior is None:
            return 0.0
        p0 = self.prior
        return float(np.log(prior / (1 - prior)) - np.log(p0 / (1 - p0)))

    def predict_threshold(self, duration, norm, precision=PRECISION, prior=None, per_pair=False):
        """Cosine score a window of this (duration, norm) needs for ``precision`` (default 95 %).

        Default: the score where P(same) reaches ``posterior_cut(precision)``, so the accepted set has
        that precision on data like the training pairs. ``per_pair=True``: the score where P(same)
        itself reaches ``precision`` (every accepted pair, so stricter, lower recall).
        The model is linear in the score for fixed (duration, norm): z = a + b*s. Returns ``inf``
        when the score does not increase P(same) there (b <= 0): nothing can be accepted.
        """
        level = precision if per_pair else self.posterior_cut(precision)
        d = np.asarray(duration, np.float64)
        n = np.asarray(norm, np.float64)
        d, n = np.broadcast_arrays(d, n)
        a = features(np.zeros(d.size), d.ravel(), n.ravel()) @ self.weights + self._shift(prior)
        b = features(np.ones(d.size), d.ravel(), n.ravel()) @ self.weights + self._shift(prior) - a
        target = np.log(level / (1 - level))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(b > 1e-9, (target - a) / b, np.inf)
        return float(t[0]) if t.size == 1 and np.ndim(duration) == 0 and np.ndim(norm) == 0 else t.reshape(d.shape)

    def to_dict(self):
        return {"weights": [float(w) for w in self.weights], "prior": self.prior,
                "posterior_cut_95": float(self.posterior_cut(0.95)),
                "features": ["1", "s", "log_d", "log_n", "s*log_d", "s*log_n"]}


def fit(scores, durations, norms, labels, l2=1e-3, iters=50):
    """Logistic regression (IRLS / Newton) of labels (1 = same speaker) on ``features``."""
    x = features(scores, durations, norms)
    y = np.asarray(labels, np.float64).reshape(-1)
    if x.shape[0] != y.size or y.size == 0:
        raise ValueError("scores, durations, norms and labels must have the same non-zero length")
    if not 0 < y.mean() < 1:
        raise ValueError("labels need both classes")
    mu, sd = x[:, 1:].mean(0), x[:, 1:].std(0) + 1e-9  # standardise for a well-conditioned Newton step
    xs = np.concatenate([x[:, :1], (x[:, 1:] - mu) / sd], axis=1)
    w = np.zeros(xs.shape[1])
    penalty = l2 * np.eye(xs.shape[1])
    penalty[0, 0] = 0.0
    for _ in range(iters):
        p = _sigmoid(xs @ w)
        grad = xs.T @ (p - y) + penalty @ w
        hess = (xs * (p * (1 - p))[:, None]).T @ xs + penalty
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    raw = np.empty_like(w)  # back to unstandardised features
    raw[1:] = w[1:] / sd
    raw[0] = w[0] - float(np.sum(w[1:] * mu / sd))
    return QualityCalibrator(weights=raw, prior=float(y.mean()), train_p=_sigmoid(x @ raw), train_y=y.astype(np.int64))


# ---------------------------------------------------------------- synthetic pairs

def synthetic_pairs(n=20_000, *, seed=0, dim=64, speakers=40, durations=None):
    """Labelled (score, duration, norm, same) pairs from a toy ECAPA-like generator.

    Speakers share a common "voice family" direction with a per-speaker weight, so impostor
    centroids lie at cosine ~0.2-0.8 (similar voices, like two male teammates). The enrollment
    is a clean 5 s embedding; a window of ``d`` seconds at quality ``q`` (SNR proxy, 0.4-1) gets
    noise of std 0.9 / (q * sqrt(d)) per unit centroid, and its raw norm grows with d and q
    (ECAPA norms shrink on short/noisy input), with log-normal jitter.
    """
    rng = np.random.default_rng(seed)
    family = rng.standard_normal(dim)
    family /= np.linalg.norm(family)
    r = rng.uniform(0.2, 0.85, speakers)
    own = rng.standard_normal((speakers, dim))
    own -= np.outer(own @ family, family)
    own /= np.linalg.norm(own, axis=1, keepdims=True)
    centroids = np.sqrt(r)[:, None] * family + np.sqrt(1 - r)[:, None] * own

    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    enroll_spk = rng.integers(0, speakers, n)
    same = rng.random(n) < 0.5
    other = (enroll_spk + rng.integers(1, speakers, n)) % speakers
    win_spk = np.where(same, enroll_spk, other)
    if durations is None:
        d = np.exp(rng.uniform(np.log(0.3), np.log(4.0), n))
    else:
        d = np.broadcast_to(np.asarray(durations, np.float64), (n,)).copy()
    q = rng.uniform(0.4, 1.0, n)
    noise = rng.standard_normal((n, dim)) / np.sqrt(dim)
    enroll = unit(centroids[enroll_spk] + 0.15 * rng.standard_normal((n, dim)) / np.sqrt(dim))
    window = unit(centroids[win_spk] + (0.9 / (q * np.sqrt(d)))[:, None] * noise)
    scores = np.sum(enroll * window, axis=1)
    norms = 20.0 * q * (1 - np.exp(-d / 0.8)) * np.exp(0.1 * rng.standard_normal(n))
    return scores, d, norms, same.astype(np.int64)


# ---------------------------------------------------------------- comparison

def precision_recall(accept, labels):
    accept = np.asarray(accept, bool)
    labels = np.asarray(labels, bool)
    tp = int(np.sum(accept & labels))
    fp = int(np.sum(accept & ~labels))
    pos = int(labels.sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    return {"precision": precision, "recall": tp / pos if pos else float("nan"), "accepted": tp + fp}


def oracle_threshold(scores, labels, precision=PRECISION):
    """Lowest single threshold on this set whose precision is >= ``precision`` (best recall)."""
    order = np.argsort(-np.asarray(scores))
    s, y = np.asarray(scores)[order], np.asarray(labels)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, y.size + 1)
    ok = np.nonzero(prec >= precision)[0]
    return float(s[ok[-1]]) if ok.size else float("inf")


def compare(cal, scores, durations, norms, labels, fixed=FIXED, precision=PRECISION):
    """Fixed vs adaptive (per-pair threshold from its duration and norm) vs this set's oracle."""
    thresholds = cal.predict_threshold(durations, norms, precision=precision)
    return {
        "n": len(labels),
        "same_rate": float(np.mean(labels)),
        "score_p50_same": float(np.median(np.asarray(scores)[np.asarray(labels) == 1])),
        "score_p50_diff": float(np.median(np.asarray(scores)[np.asarray(labels) == 0])),
        "fixed": {"threshold": fixed, **precision_recall(np.asarray(scores) >= fixed, labels)},
        "adaptive": {"threshold_p50": float(np.median(thresholds)),
                     **precision_recall(np.asarray(scores) >= thresholds, labels)},
        "oracle": {"threshold": (t := oracle_threshold(scores, labels, precision)),
                   **precision_recall(np.asarray(scores) >= t, labels)},
    }


def synthetic_report(seed=0, n_train=20_000, n_eval=10_000):
    cal = fit(*synthetic_pairs(n_train, seed=seed))
    out = {"model": cal.to_dict(), "bins": {}}
    for d in (SHORT_S, 1.0, LONG_S):
        out["bins"][f"{d:g}s"] = compare(cal, *synthetic_pairs(n_eval, seed=seed + 1000, durations=d))
    return cal, out


# ---------------------------------------------------------------- real teammate pairs (optional)

def team_pairs(root, embedder, crops=(SHORT_S, 1.0, LONG_S), hop_s=None):
    """Enroll each teammate on ``profile-*.wav``; score crops of their ``fixture-*.wav`` takes.

    Returns (scores, durations, norms, labels): every crop is scored against both profiles.
    """
    import soundfile as sf

    root = Path(root)
    speakers = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_"))

    def load(path):
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            from hearing.audio.io import resample

            x = resample(x, sr, SR)
        return x

    def emb(x):
        e = np.asarray(embedder.embed(x), np.float64).reshape(-1)
        return e, float(np.linalg.norm(e))

    profiles = {}
    for spk in speakers:
        clips = sorted((root / spk).glob("profile-*.wav"))
        if clips:
            e, n = emb(np.concatenate([load(p) for p in clips]))
            profiles[spk] = e / n
    rows = []
    for spk in speakers:
        for path in sorted((root / spk).glob("fixture-*.wav")):
            x = load(path)
            for crop in crops:
                step = int((hop_s or crop) * SR)
                size = int(crop * SR)
                for start in range(0, x.size - size + 1, step):
                    w = x[start:start + size]
                    if np.sqrt(np.mean(w ** 2)) < 1e-3:  # digital silence carries no voice
                        continue
                    e, n = emb(w)
                    for ref, centroid in profiles.items():
                        rows.append((float(np.dot(centroid, e / n)), crop, n, int(ref == spk)))
    if not rows:
        raise FileNotFoundError(f"no profile/fixture pairs under {root}")
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3].astype(np.int64)


def team_report(root, folds=5, seed=0):
    """Real ECAPA pairs; adaptive thresholds come from ``folds``-fold out-of-fold fits (by window)."""
    from hearing.tse.enrolled import _load_encoder

    s, d, n, y = team_pairs(root, _load_encoder())
    windows = np.repeat(np.arange(y.size // 2), 2)  # both profile scores of one crop stay in one fold
    fold = np.random.default_rng(seed).permutation(y.size // 2)[windows] % folds
    thresholds = np.empty_like(s)
    for k in range(folds):
        cal = fit(s[fold != k], d[fold != k], n[fold != k], y[fold != k])
        thresholds[fold == k] = cal.predict_threshold(d[fold == k], n[fold == k])
    cal = fit(s, d, n, y)
    out = {"pairs": int(y.size), "folds": folds, "model": cal.to_dict(), "bins": {}}
    for crop in np.unique(d):
        m = d == crop
        b = compare(cal, s[m], d[m], n[m], y[m])
        b["adaptive"] = {"threshold_p50": float(np.median(thresholds[m])),
                         **precision_recall(s[m] >= thresholds[m], y[m])}
        out["bins"][f"{crop:g}s"] = b
    out["norm_p50_by_crop"] = {f"{c:g}s": float(np.median(n[d == c])) for c in np.unique(d)}
    out["threshold_full_fit"] = {f"{c:g}s": float(cal.predict_threshold(c, np.median(n[d == c])))
                                 for c in np.unique(d)}
    return out


def _table(bins):
    lines = ["| window | n | same p50 | diff p50 | fixed 0.7 P / R | adaptive thr p50 | adaptive P / R | oracle thr | oracle P / R |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, b in bins.items():
        f, a, o = b["fixed"], b["adaptive"], b["oracle"]
        lines.append(
            f"| {name} | {b['n']} | {b['score_p50_same']:.2f} | {b['score_p50_diff']:.2f} | "
            f"{f['precision']:.3f} / {f['recall']:.3f} | {a['threshold_p50']:.2f} | "
            f"{a['precision']:.3f} / {a['recall']:.3f} | {o['threshold']:.2f} | {o['precision']:.3f} / {o['recall']:.3f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--team", action="store_true", help="also score real teammate pairs with ECAPA")
    parser.add_argument("--team-root", type=Path, default=Path(os.environ.get(
        "HEARING_FIXTURES_DIR", "eval/fixtures/generated")) / "team")
    parser.add_argument("--out", type=Path, default=None, help="JSON results path")
    args = parser.parse_args(argv)
    cal, syn = synthetic_report(seed=args.seed)
    print("## synthetic\n" + _table(syn["bins"]))
    for d, nrm in ((SHORT_S, 5.0), (SHORT_S, 9.0), (LONG_S, 12.0), (LONG_S, 18.0)):
        print(f"threshold@{PRECISION:.0%} d={d}s norm={nrm}: "
              f"{cal.predict_threshold(d, nrm):.3f}")
    result = {"synthetic": syn}
    if args.team:
        result["team"] = team_report(args.team_root)
        print("## team (ECAPA)\n" + _table(result["team"]["bins"]))
        print("norm p50 by crop:", result["team"]["norm_p50_by_crop"])
        print("full-fit threshold@95% at median norm:", result["team"]["threshold_full_fit"])
    out = args.out or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / "T-308_quality_cal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print("wrote", out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
