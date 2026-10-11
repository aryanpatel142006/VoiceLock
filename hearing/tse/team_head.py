"""T-336 team-tuned decision head: a calibrated logistic p(target dominates this separated stream).

Trained by ``eval/train_team_head.py`` on 1 s windows of aryan+joseph mixtures (enrollment cuts only)
run through the live separator, so it sees the input the LDA gate fails on. Opt-in: ``HEARING_TEAM_HEAD``
names the saved JSON weights; unset (or empty) leaves ``EnrolledCandidate`` exactly as before. Tuned to the
two teammates' voices; it only acts in closed-set mode (both voices saved profiles).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

ENV = "HEARING_TEAM_HEAD"
FEATURES = ("lda", "lda_rival", "cos_target", "cos_other", "rel_db")
FLOOR_DB = -60.0  # stream energy floor (dB re the loudest stream) so a silent stream stays finite
P_CLIP = 1e-6


def stream_features(lda, cos_target, cos_other, energy):
    """``(n_streams, len(FEATURES))``: per stream its LDA score, the best rival stream's LDA score, its
    cosines to the target and the other profile, and its energy in dB relative to the loudest stream."""
    lda = np.asarray(lda, np.float64)
    e = np.maximum(np.asarray(energy, np.float64), 1e-12)
    rel = np.maximum(10.0 * np.log10(e / e.max()), FLOOR_DB)
    rival = np.array([np.max(np.delete(lda, i)) if len(lda) > 1 else 0.0 for i in range(len(lda))])
    return np.column_stack([lda, rival, np.asarray(cos_target, np.float64), np.asarray(cos_other, np.float64), rel])


class TeamHead:
    """Standardised logistic regression; ``threshold`` is the calibrated accept probability."""

    def __init__(self, mean, scale, coef, intercept, threshold=0.5, meta=None):
        self.mean = np.asarray(mean, np.float64)
        self.scale = np.asarray(scale, np.float64)
        self.coef = np.asarray(coef, np.float64)
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.meta = meta or {}
        if not (self.mean.shape == self.scale.shape == self.coef.shape == (len(FEATURES),)):
            raise ValueError(f"team head expects {len(FEATURES)} features {FEATURES}")

    def prob(self, x):
        z = (np.asarray(x, np.float64) - self.mean) / self.scale @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))

    def scores(self, x):
        """Log-odds of each stream relative to the threshold: > 0 accepts (same sign rule as the LDA gate)."""
        p = np.clip(self.prob(x), P_CLIP, 1.0 - P_CLIP)
        t = min(max(self.threshold, P_CLIP), 1.0 - P_CLIP)
        return np.log(p / (1.0 - p)) - math.log(t / (1.0 - t))

    def to_dict(self):
        return {"features": list(FEATURES), "mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "coef": self.coef.tolist(), "intercept": self.intercept, "threshold": self.threshold,
                "meta": self.meta}

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if list(d.get("features", [])) != list(FEATURES):
            raise ValueError(f"{path}: features {d.get('features')} != {list(FEATURES)}")
        return cls(d["mean"], d["scale"], d["coef"], d["intercept"], d.get("threshold", 0.5), d.get("meta"))


def from_env():
    """The head named by ``$HEARING_TEAM_HEAD``, None when unset or empty."""
    path = os.environ.get(ENV, "").strip()
    return TeamHead.load(path) if path else None


def fit_logistic(x, y, l2=1.0, iters=50):
    """L2-regularised logistic regression by Newton steps on standardised features (CPU, numpy only)."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    w = np.zeros(z.shape[1])
    reg = np.full(z.shape[1], l2)
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -40.0, 40.0)))
        grad = z.T @ (p - y) + reg * w
        hess = (z * (p * (1 - p))[:, None]).T @ z + np.diag(reg) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return mean, scale, w[:-1], float(w[-1])
