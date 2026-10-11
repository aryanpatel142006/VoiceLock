"""Downstream selector gate (T-334): selector arms on identical frozen streams.

Every arm sees the same per-window ECAPA embeddings of the same source streams, the same
gallery, the same tracker and the same dev-frozen threshold rule; only the embedding
transform differs. Selected/delivered audio and Arena-consumable rows are exported.
Embeddings are extracted once upstream (eval/embeddings/T-291); nothing here trains a
separator or touches hearing/config.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
EMA_ALPHA = 0.5  # frozen tracker
ABSENT_FA_TARGET = 0.05  # frozen threshold rule: 95th percentile of dev absent-case scores
SHRINKAGE = 0.30  # frozen Mahalanobis shrinkage (T-274 default)


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 1e-12, n, 1.0)


@dataclass
class Case:
    """One matched scene: K source streams, window-aligned, one gallery."""

    case_id: str
    split: str  # "dev" | "final"
    group: str  # independent-source group for uncertainty
    emb: np.ndarray  # (K, T, D) raw ECAPA per stream window
    head: np.ndarray | None  # (K, T, 32) projection-head outputs, or None
    target: int | None  # stream index of the target; None = target absent
    gallery: np.ndarray  # (G, D) raw ECAPA gallery windows
    win_s: float
    audio: list[str] | None = None  # per-stream wav path (for delivered audio)
    start_s: list[np.ndarray] | None = None  # per-stream window starts
    gallery_head: np.ndarray | None = None  # (G, 32) head outputs of the gallery


# ---------------------------------------------------------------- arms
class Arm:
    name = ""
    kind = "selector"
    checkpoint = ""

    def fit(self, fit_emb: np.ndarray, fit_lab: np.ndarray) -> None:
        pass

    def transform(self, case: Case, x: np.ndarray, which: str) -> np.ndarray:
        raise NotImplementedError


class RawArm(Arm):
    name = "raw_ecapa"

    def transform(self, case, x, which):
        return unit(x)


class IdentityArm(Arm):
    """No-op head through the head code path; must match raw bit-for-bit."""

    name = "identity_head"

    def transform(self, case, x, which):
        w = np.eye(x.shape[-1])
        return unit(np.asarray(x, dtype=np.float64) @ w)


class MahalanobisArm(Arm):
    name = "mahalanobis"

    def fit(self, fit_emb, fit_lab):
        x = unit(fit_emb)
        cs = [x[fit_lab == c] - x[fit_lab == c].mean(0) for c in np.unique(fit_lab)]
        cov = np.cov(np.concatenate(cs).T)
        cov = (1 - SHRINKAGE) * cov + SHRINKAGE * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0])
        vals, vecs = np.linalg.eigh(cov)
        self.w = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-8))) @ vecs.T

    def transform(self, case, x, which):
        return unit(unit(x) @ self.w)


class HeadArm(Arm):
    """Existing projection-head checkpoint; uses its stored outputs for stream windows."""

    name = "projection_head"

    def __init__(self, ckpt: str) -> None:
        self.checkpoint = ckpt

    def transform(self, case, x, which):
        if which == "gallery":
            return unit(case.gallery_head)
        return unit(x)


class BlockedArm(Arm):
    def __init__(self, name: str, reason: str) -> None:
        self.name, self.reason = name, reason


# ---------------------------------------------------------------- selection
def stream_scores(arm: Arm, case: Case) -> np.ndarray:
    """(K, T) cosine of each stream window to the frozen gallery centroid, in arm space."""
    k, t, _ = case.emb.shape
    src = case.head if isinstance(arm, HeadArm) else case.emb
    g = arm.transform(case, case.gallery_head if isinstance(arm, HeadArm) else case.gallery, "gallery")
    c = unit(g.mean(0))
    out = np.zeros((k, t))
    for i in range(k):
        out[i] = arm.transform(case, src[i], "stream") @ c
    return out


def track(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Frozen tracker: EMA per stream, argmax, accept only above threshold. -1 = none."""
    t = scores.shape[1]
    sel = np.full(t, -1, dtype=int)
    ema = scores[:, 0].copy()
    for j in range(t):
        ema = scores[:, j] if j == 0 else EMA_ALPHA * scores[:, j] + (1 - EMA_ALPHA) * ema
        best = int(np.argmax(ema))
        if ema[best] >= threshold:
            sel[j] = best
    return sel


def calibrate_threshold(arm: Arm, dev_cases: list[Case]) -> float:
    """Threshold from dev target-absent cases only: tracked score 95th pct (5% FA)."""
    vals = []
    for c in dev_cases:
        if c.target is None:
            s = stream_scores(arm, c)
            ema = s.copy()
            for j in range(1, s.shape[1]):
                ema[:, j] = EMA_ALPHA * s[:, j] + (1 - EMA_ALPHA) * ema[:, j - 1]
            vals.append(ema.max(0))
    if not vals:
        raise ValueError("no dev target-absent case to calibrate on")
    return float(np.quantile(np.concatenate(vals), 1 - ABSENT_FA_TARGET))


def eval_case(arm: Arm, case: Case, threshold: float) -> dict:
    t0 = time.perf_counter()
    s = stream_scores(arm, case)
    sel = track(s, threshold)
    ms = (time.perf_counter() - t0) * 1000 / s.shape[1]
    T = s.shape[1]
    r = {"case_id": case.case_id, "group": case.group, "windows": T, "selected": sel.tolist(),
         "switches": int(np.sum((sel[1:] != sel[:-1]) & (sel[1:] >= 0) & (sel[:-1] >= 0))),
         "ms_per_window": ms, "scores": s}
    if case.target is None:
        r.update(absent_false_accept=float(np.mean(sel >= 0)), top1=None, margin=None,
                 wrong_speaker_s=None, retention=None, reacquisition_windows=None)
    else:
        oth = [i for i in range(s.shape[0]) if i != case.target]
        r.update(absent_false_accept=None, top1=float(np.mean(np.argmax(s, 0) == case.target)),
                 margin=float(np.mean(s[case.target] - s[oth].max(0))),
                 wrong_speaker_s=float(np.sum((sel >= 0) & (sel != case.target)) * case.win_s),
                 retention=float(np.mean(sel == case.target)),
                 reacquisition_windows=int(np.argmax(sel == case.target)) if np.any(sel == case.target) else None)
    return r


def summarize(rows: list[dict]) -> dict:
    """Pool windows; uncertainty = bootstrap over independent groups (null if <2 groups)."""
    def pooled(key, subset):
        v = [(r[key], r["windows"]) for r in subset if r[key] is not None]
        return float(sum(a * w for a, w in v) / sum(w for _, w in v)) if v else None

    def block(subset):
        return {"top1": pooled("top1", subset), "margin": pooled("margin", subset),
                "retention": pooled("retention", subset),
                "absent_false_accept": pooled("absent_false_accept", subset),
                "wrong_speaker_s": float(sum(r["wrong_speaker_s"] for r in subset if r["wrong_speaker_s"] is not None)),
                "switches": int(sum(r["switches"] for r in subset))}

    out = block(rows)
    groups = sorted({r["group"] for r in rows})
    out["n_groups"] = len(groups)
    out["recovery_ratio"] = None  # needs separated-vs-oracle SI-SDR; not defined for selection-only
    out["unknown_false_accept"] = None  # no third-party stream in the bank
    reacq = [r["reacquisition_windows"] for r in rows if r["reacquisition_windows"] is not None]
    out["reacquisition_windows_median"] = float(np.median(reacq)) if reacq else None
    ms = [r["ms_per_window"] for r in rows]
    out["isolated_ms_per_window_p50"] = float(np.median(ms))
    out["isolated_ms_per_window_p95"] = float(np.quantile(ms, 0.95))
    if len(groups) >= 2:
        rng = np.random.default_rng(0)
        by = {g: [r for r in rows if r["group"] == g] for g in groups}
        boots = []
        for _ in range(500):
            pick = rng.choice(groups, len(groups))
            boots.append(pooled("top1", [r for g in pick for r in by[g]]))
        boots = [b for b in boots if b is not None]
        out["top1_ci95_by_group"] = [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))] if boots else None
    else:
        out["top1_ci95_by_group"] = None
    return out


def sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def delivered_audio(case: Case, sel: np.ndarray, read) -> np.ndarray:
    """Concatenate each window from the selected stream (silence when none)."""
    n = int(case.win_s * SR)
    out = np.zeros(len(sel) * n, dtype=np.float32)
    for j, k in enumerate(sel):
        if k < 0:
            continue
        a = read(case.audio[k], float(case.start_s[k][j]), n)
        out[j * n : j * n + len(a)] = a
    return out


def parity(raw: list[dict], ident: list[dict]) -> dict:
    d = max(float(np.max(np.abs(a["scores"] - b["scores"]))) for a, b in zip(raw, ident, strict=True))
    same = all(a["selected"] == b["selected"] for a, b in zip(raw, ident, strict=True))
    return {"max_abs_score_diff": d, "identical_selection": same, "ok": same and d < 1e-9}


def dumps(o) -> str:
    return json.dumps(o, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
