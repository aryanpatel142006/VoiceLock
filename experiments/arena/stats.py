"""Paired confidence estimates that respect dependence.

The unit of resampling is a *cluster* of mixtures that share source audio (same target/interferer
utterances), never a window: adjacent 100 ms windows of one clip are strongly correlated, and so
are two mixtures built from the same two utterances. Per-mixture metrics are paired candidate -
baseline differences on the same mixture, so the baseline's difficulty cancels.
"""
from __future__ import annotations

import numpy as np


def paired_cluster_bootstrap(
    deltas: list[float], clusters: list[str], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Mean of paired ``deltas`` and a percentile CI from resampling whole clusters with replacement."""
    pairs = [(d, c) for d, c in zip(deltas, clusters, strict=True) if d is not None and np.isfinite(d)]
    if not pairs:
        return {"delta": None, "lo": None, "hi": None, "n": 0, "clusters": 0}
    d = np.array([p[0] for p in pairs], dtype=np.float64)
    keys = sorted({p[1] for p in pairs})
    sums = np.array([d[[p[1] == k for p in pairs]].sum() for k in keys])
    counts = np.array([sum(p[1] == k for p in pairs) for k in keys], dtype=np.float64)
    if len(keys) == 1:  # one cluster: no between-cluster variance is estimable
        return {"delta": float(d.mean()), "lo": None, "hi": None, "n": len(d), "clusters": 1}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": len(d), "clusters": len(keys)}


def verdict(ci: dict, direction: int, margin: float = 0.0) -> str:
    """``better`` / ``worse`` when the CI clears ``margin`` in the metric's good/bad direction,
    ``tied`` when it sits inside +-margin, else ``uncertain`` (which earns more evaluation)."""
    if ci["delta"] is None:
        return "missing"
    if ci["lo"] is None:
        return "uncertain"
    lo, hi = (ci["lo"], ci["hi"]) if direction > 0 else (-ci["hi"], -ci["lo"])
    if lo > margin:
        return "better"
    if hi < -margin:
        return "worse"
    if margin > 0 and lo >= -margin and hi <= margin:
        return "tied"
    return "uncertain"
