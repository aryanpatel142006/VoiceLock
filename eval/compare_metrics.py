"""Head-to-head speaker-verification metric comparison harness (T-292).

Compares speaker-verification metrics on frozen ECAPA and projection head embeddings:
1. Stock ECAPA cosine similarity (192-d)
2. Calibrated pairwise Mahalanobis metric (192-d, per T-274 spec)
3. Projection-head cosine similarity (32-d)

Computes:
- Equal Error Rate (EER) and operating threshold
- Minimum Detection Cost Function (minDCF, NIST SRE normalized)
- Margin distributions and percentiles (P05, P25, P50, P75, P95)

Outputs:
- Markdown summary table
- Machine-readable JSON with explicit provenance flags (backend: cosine|mahalanobis|head, measured: true)
- PNG visualizations: margin histograms, DET curves, metric summary bar charts.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local imports work when executed directly or via python -m
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.mahalanobis_metric import MahalanobisMetric

DEFAULT_DATASET = _ROOT / "eval" / "embeddings" / "T-291" / "team_slices.npz"
DEFAULT_OUT_DIR = _ROOT / "results" / "metrics" / "T-292"
WINDOWS = ("0.5s", "1.0s", "2.0s")
BACKENDS = ("cosine", "mahalanobis", "head")


# ---------------------------------------------------------------------------
# 1. Metric Backend Protocols & Implementations
# ---------------------------------------------------------------------------


class MetricBackend:
    """Base class for speaker-verification scoring backends."""

    backend_id: str
    is_measured: bool = True
    embedding_dim: int

    def fit(self, embeddings: np.ndarray, speakers: np.ndarray) -> None:
        pass

    def score_pairs(
        self,
        embeddings: np.ndarray,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class CosineBackend(MetricBackend):
    """Stock ECAPA cosine similarity in 192-d unit space."""

    def __init__(self) -> None:
        self.backend_id = "cosine"
        self.is_measured = True
        self.embedding_dim = 192

    def score_pairs(
        self,
        embeddings: np.ndarray,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
    ) -> np.ndarray:
        u = embeddings[idx_i]
        v = embeddings[idx_j]
        # Unit-normalized vectors dot product
        return np.sum(u * v, axis=1).astype(np.float32)


class HeadBackend(MetricBackend):
    """Discriminative projection head cosine similarity in 32-d unit space."""

    def __init__(self) -> None:
        self.backend_id = "head"
        self.is_measured = True
        self.embedding_dim = 32

    def score_pairs(
        self,
        embeddings: np.ndarray,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
    ) -> np.ndarray:
        u = embeddings[idx_i]
        v = embeddings[idx_j]
        return np.sum(u * v, axis=1).astype(np.float32)


class MahalanobisBackend(MetricBackend):
    """Calibrated pairwise Mahalanobis metric per T-274 spec.

    Estimates pooled intra-speaker covariance with Ledoit-Wolf shrinkage,
    inverts regularized covariance via whitening matrix L = Sigma_reg^{-1/2},
    and computes cosine similarity in the whitened feature space.
    """

    def __init__(self, shrinkage: float = 0.30, center: bool = True) -> None:
        self.backend_id = "mahalanobis"
        self.is_measured = True
        self.embedding_dim = 192
        self.shrinkage = shrinkage
        self.center = center
        self.metric: MahalanobisMetric | None = None
        self._whitened_cache: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, speakers: np.ndarray) -> None:
        unique_spk = np.unique(speakers)
        if len(unique_spk) >= 2:
            spk0_embs = embeddings[speakers == unique_spk[0]]
            spk1_embs = embeddings[speakers == unique_spk[1]]
            self.metric = MahalanobisMetric(shrinkage=self.shrinkage).fit(spk0_embs, spk1_embs)
        else:
            self.metric = MahalanobisMetric(shrinkage=self.shrinkage).fit(embeddings, None)

        self._whitened_cache = self.metric.transform(embeddings, center=self.center).astype(np.float32)

    def score_pairs(
        self,
        embeddings: np.ndarray,
        idx_i: np.ndarray,
        idx_j: np.ndarray,
    ) -> np.ndarray:
        if self._whitened_cache is None or len(self._whitened_cache) != len(embeddings):
            if self.metric is None:
                raise RuntimeError("MahalanobisBackend must be fit before scoring")
            z = self.metric.transform(embeddings, center=self.center).astype(np.float32)
        else:
            z = self._whitened_cache
        u = z[idx_i]
        v = z[idx_j]
        return np.sum(u * v, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Evaluation Mathematics: EER, minDCF, Margins
# ---------------------------------------------------------------------------


def compute_eer(pos_scores: np.ndarray, neg_scores: np.ndarray) -> tuple[float, float]:
    """Compute Equal Error Rate (EER) and decision threshold via linear interpolation.

    Parameters:
        pos_scores: Scores for target (same-speaker) trials.
        neg_scores: Scores for impostor (different-speaker) trials.

    Returns:
        (eer, threshold): EER as fraction in [0, 1] and operating threshold.
    """
    pos = np.sort(pos_scores)
    neg = np.sort(neg_scores)

    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0

    all_scores = np.sort(np.unique(np.concatenate([pos, neg])))
    # Include boundary thresholds (below min and above max)
    delta = max(1e-4, float((all_scores[-1] - all_scores[0]) * 1e-4)) if len(all_scores) > 1 else 1e-4
    all_scores = np.concatenate([
        [all_scores[0] - delta],
        all_scores,
        [all_scores[-1] + delta],
    ])

    # FRR: fraction of target trials < threshold
    # FAR: fraction of non-target trials >= threshold
    frr = np.searchsorted(pos, all_scores, side="left") / len(pos)
    far = (len(neg) - np.searchsorted(neg, all_scores, side="left")) / len(neg)

    diff = far - frr
    cross_idx = np.where(diff <= 0)[0]

    if len(cross_idx) == 0:
        return float(frr[-1]), float(all_scores[-1])

    i = cross_idx[0]
    if i == 0:
        return float(far[0]), float(all_scores[0])

    d_prev = diff[i - 1]
    d_curr = diff[i]
    denom = d_prev - d_curr
    alpha = float(d_prev / denom) if denom != 0 else 0.0

    eer = float(frr[i - 1] + alpha * (frr[i] - frr[i - 1]))
    thresh = float(all_scores[i - 1] + alpha * (all_scores[i] - all_scores[i - 1]))
    return max(0.0, min(1.0, eer)), thresh


def compute_mindcf(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> tuple[float, float]:
    """Compute normalized minimum Detection Cost Function (minDCF) per NIST SRE standard.

    Parameters:
        pos_scores: Scores for target trials.
        neg_scores: Scores for non-target trials.
        p_target: Prior probability of target speaker.
        c_miss: Cost of missed detection (false negative).
        c_fa: Cost of false alarm (false positive).

    Returns:
        (min_dcf, threshold): Normalized minDCF value and optimal threshold.
    """
    pos = np.sort(pos_scores)
    neg = np.sort(neg_scores)

    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0

    all_scores = np.sort(np.unique(np.concatenate([pos, neg])))
    delta = max(1e-4, float((all_scores[-1] - all_scores[0]) * 1e-4)) if len(all_scores) > 1 else 1e-4
    all_scores = np.concatenate([
        [all_scores[0] - delta],
        all_scores,
        [all_scores[-1] + delta],
    ])

    frr = np.searchsorted(pos, all_scores, side="left") / len(pos)
    far = (len(neg) - np.searchsorted(neg, all_scores, side="left")) / len(neg)

    c_det = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    c_default = min(c_miss * p_target, c_fa * (1.0 - p_target))

    if c_default <= 0:
        return 0.0, float(all_scores[0])

    c_norm = c_det / c_default
    best_idx = int(np.argmin(c_norm))
    return min(1.0, float(c_norm[best_idx])), float(all_scores[best_idx])


def compute_margins(
    scores: np.ndarray,
    is_same: np.ndarray,
    n_slices: int,
    triu_i: np.ndarray,
    triu_j: np.ndarray,
) -> dict[str, Any]:
    """Compute pairwise margin and per-slice decision margin distribution."""
    pos_scores = scores[is_same]
    neg_scores = scores[~is_same]

    pos_mean = float(np.mean(pos_scores)) if len(pos_scores) > 0 else 0.0
    neg_mean = float(np.mean(neg_scores)) if len(neg_scores) > 0 else 0.0
    pairwise_margin = pos_mean - neg_mean

    # Build per-slice score matrix
    score_matrix = np.zeros((n_slices, n_slices), dtype=np.float32)
    score_matrix[triu_i, triu_j] = scores
    score_matrix[triu_j, triu_i] = scores

    # Determine per-slice same and cross speaker masks
    slice_margins = np.zeros(n_slices, dtype=np.float32)
    for i in range(n_slices):
        pairs_i = np.where(triu_i == i)[0]
        pairs_j = np.where(triu_j == i)[0]

        same_sims = []
        cross_sims = []

        for p in pairs_i:
            (same_sims if is_same[p] else cross_sims).append(scores[p])
        for p in pairs_j:
            (same_sims if is_same[p] else cross_sims).append(scores[p])

        s_same = np.mean(same_sims) if same_sims else 0.0
        s_cross = np.mean(cross_sims) if cross_sims else 0.0
        slice_margins[i] = s_same - s_cross

    return {
        "pairwise_margin": pairwise_margin,
        "pos_score_mean": pos_mean,
        "neg_score_mean": neg_mean,
        "margin_mean": float(np.mean(slice_margins)),
        "margin_std": float(np.std(slice_margins)),
        "margin_min": float(np.min(slice_margins)),
        "margin_max": float(np.max(slice_margins)),
        "margin_p05": float(np.percentile(slice_margins, 5)),
        "margin_p25": float(np.percentile(slice_margins, 25)),
        "margin_p50": float(np.median(slice_margins)),
        "margin_p75": float(np.percentile(slice_margins, 75)),
        "margin_p95": float(np.percentile(slice_margins, 95)),
        "slice_margins": slice_margins,
    }


def build_trials(
    data: dict[str, Any],
    trial_mode: str = "all_pairs",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct speaker verification pairs and ground-truth same/cross labels.

    Returns:
        (triu_i, triu_j, is_same)
    """
    speakers = data["speaker"]
    n = len(speakers)
    triu_i, triu_j = np.triu_indices(n, k=1)

    if trial_mode == "cross_session" and "session" in data:
        sessions = data["session"]
        cross_mask = sessions[triu_i] != sessions[triu_j]
        triu_i = triu_i[cross_mask]
        triu_j = triu_j[cross_mask]

    is_same = speakers[triu_i] == speakers[triu_j]
    return triu_i, triu_j, is_same


# ---------------------------------------------------------------------------
# 3. Data Structures & Machine-Readable Provenance
# ---------------------------------------------------------------------------


@dataclass
class MetricRecord:
    """One evaluation result row with strict machine-readable provenance flags."""

    backend: str  # "cosine" | "mahalanobis" | "head"
    measured: bool  # strictly boolean True
    window: str  # "0.5s", "1.0s", "2.0s"
    window_s: float
    dim: int
    n_slices: int
    n_pos_trials: int
    n_neg_trials: int
    eer: float
    eer_percent: float
    eer_threshold: float
    min_dcf: float
    min_dcf_threshold: float
    pairwise_margin: float
    pos_score_mean: float
    neg_score_mean: float
    margin_mean: float
    margin_std: float
    margin_p05: float
    margin_p25: float
    margin_p50: float
    margin_p75: float
    margin_p95: float
    dcf_p_target: float = 0.01
    dcf_c_miss: float = 1.0
    dcf_c_fa: float = 1.0
    trial_mode: str = "all_pairs"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MetricReport:
    """Full experiment report encompassing all windows and backends."""

    task: str
    timestamp: str
    dataset_path: str
    trial_mode: str
    records: list[MetricRecord]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "timestamp": self.timestamp,
            "dataset_path": self.dataset_path,
            "trial_mode": self.trial_mode,
            "records": [r.to_dict() for r in self.records],
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# 4. Evaluation Engine
# ---------------------------------------------------------------------------


def load_window_data(npz_file: np.lib.npyio.NpzFile, window: str) -> dict[str, Any]:
    """Extract arrays for a specific window duration from npz archive."""
    session_key = f"meta/{window}/session"
    session_arr = npz_file[session_key] if session_key in npz_file.files else None
    return {
        "embeddings": npz_file[f"embeddings/{window}"],
        "head_out": npz_file[f"head_out/{window}"],
        "speaker": npz_file[f"meta/{window}/speaker"],
        "session": session_arr,
        "window": window,
    }


def evaluate_dataset(
    dataset_path: str | Path,
    windows: list[str] | tuple[str, ...] = WINDOWS,
    backends: list[str] | tuple[str, ...] = BACKENDS,
    trial_mode: str = "all_pairs",
    shrinkage: float = 0.30,
    center_mahalanobis: bool = True,
    p_target: float = 0.01,
    c_miss: float = 1.0,
    c_fa: float = 1.0,
) -> MetricReport:
    """Evaluate speaker verification metrics across specified windows and backends."""
    p = Path(dataset_path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset not found: {p}")

    records: list[MetricRecord] = []
    margin_distributions: dict[str, dict[str, np.ndarray]] = {}

    with np.load(p, allow_pickle=True) as z:
        for w in windows:
            w_s = float(w.rstrip("s"))
            data = load_window_data(z, w)
            triu_i, triu_j, is_same = build_trials(data, trial_mode=trial_mode)

            n_slices = len(data["speaker"])
            n_pos = int(np.sum(is_same))
            n_neg = int(np.sum(~is_same))

            margin_distributions[w] = {}

            for b_name in backends:
                backend: MetricBackend
                if b_name == "cosine":
                    backend = CosineBackend()
                    feature_matrix = data["embeddings"]
                elif b_name == "head":
                    backend = HeadBackend()
                    feature_matrix = data["head_out"]
                elif b_name == "mahalanobis":
                    backend = MahalanobisBackend(shrinkage=shrinkage, center=center_mahalanobis)
                    backend.fit(data["embeddings"], data["speaker"])
                    feature_matrix = data["embeddings"]
                else:
                    raise ValueError(f"Unknown backend {b_name!r}")

                scores = backend.score_pairs(feature_matrix, triu_i, triu_j)
                pos_scores = scores[is_same]
                neg_scores = scores[~is_same]

                eer, eer_th = compute_eer(pos_scores, neg_scores)
                min_dcf, dcf_th = compute_mindcf(
                    pos_scores, neg_scores, p_target=p_target, c_miss=c_miss, c_fa=c_fa
                )
                margin_info = compute_margins(scores, is_same, n_slices, triu_i, triu_j)

                margin_distributions[w][b_name] = margin_info["slice_margins"]

                rec = MetricRecord(
                    backend=b_name,
                    measured=True,
                    window=w,
                    window_s=w_s,
                    dim=backend.embedding_dim,
                    n_slices=n_slices,
                    n_pos_trials=n_pos,
                    n_neg_trials=n_neg,
                    eer=eer,
                    eer_percent=eer * 100.0,
                    eer_threshold=eer_th,
                    min_dcf=min_dcf,
                    min_dcf_threshold=dcf_th,
                    pairwise_margin=margin_info["pairwise_margin"],
                    pos_score_mean=margin_info["pos_score_mean"],
                    neg_score_mean=margin_info["neg_score_mean"],
                    margin_mean=margin_info["margin_mean"],
                    margin_std=margin_info["margin_std"],
                    margin_p05=margin_info["margin_p05"],
                    margin_p25=margin_info["margin_p25"],
                    margin_p50=margin_info["margin_p50"],
                    margin_p75=margin_info["margin_p75"],
                    margin_p95=margin_info["margin_p95"],
                    dcf_p_target=p_target,
                    dcf_c_miss=c_miss,
                    dcf_c_fa=c_fa,
                    trial_mode=trial_mode,
                )
                records.append(rec)

    provenance = {
        "cosine": {
            "backend": "cosine",
            "measured": True,
            "embedding_dim": 192,
            "backbone": "speechbrain/spkrec-ecapa-voxceleb",
            "notes": "Standard unit cosine similarity in isotropic 192-d ECAPA space",
        },
        "mahalanobis": {
            "backend": "mahalanobis",
            "measured": True,
            "embedding_dim": 192,
            "shrinkage": shrinkage,
            "centered": center_mahalanobis,
            "backbone": "speechbrain/spkrec-ecapa-voxceleb + T-274 regularized pooled whitening",
            "notes": "Closed-form whitening L = Sigma_reg^{-1/2} downweighting shared male vocal tract dimensions",
        },
        "head": {
            "backend": "head",
            "measured": True,
            "embedding_dim": 32,
            "model_path": "models/team_projection_head.pt",
            "notes": "Learned discriminative projection head 192->64->32 with contrastive triplet/margin training",
        },
    }

    report = MetricReport(
        task="T-292",
        timestamp=datetime.now(UTC).isoformat(),
        dataset_path=str(p),
        trial_mode=trial_mode,
        records=records,
        provenance=provenance,
    )
    # Stash raw distributions on report for plot generator
    report._margin_distributions = margin_distributions  # type: ignore[attr-defined]
    return report


# ---------------------------------------------------------------------------
# 5. Markdown Formatting & Visualizations
# ---------------------------------------------------------------------------


def format_markdown_table(records: list[MetricRecord]) -> str:
    """Format evaluation records into clean GitHub Flavored Markdown."""
    lines = [
        "| Window | Backend | Measured | Dim | EER (%) | minDCF (p=0.01) | Mean Margin | Median Margin | Same Cos | Cross Cos |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in records:
        lines.append(
            f"| `{r.window}` | **`{r.backend}`** | `{str(r.measured).lower()}` | {r.dim} | "
            f"{r.eer_percent:6.2f}% | {r.min_dcf:6.3f} | "
            f"{r.margin_mean:+6.3f} | {r.margin_p50:+6.3f} | "
            f"{r.pos_score_mean:+6.3f} | {r.neg_score_mean:+6.3f} |"
        )
    return "\n".join(lines)


def render_plots(report: MetricReport, out_dir: Path) -> list[Path]:
    """Generate professional PNG visualizations comparing metrics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # 1. Margin Histograms per window length (3-panel)
    dist_map = getattr(report, "_margin_distributions", {})
    if dist_map:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
        colors = {"cosine": "#d9534f", "mahalanobis": "#f0ad4e", "head": "#0275d8"}

        for ax, w in zip(axes, WINDOWS):
            w_dists = dist_map.get(w, {})
            for b_name in BACKENDS:
                vals = w_dists.get(b_name)
                if vals is not None:
                    ax.hist(
                        vals,
                        bins=25,
                        alpha=0.45,
                        color=colors.get(b_name, "#555555"),
                        label=f"{b_name} (mean={np.mean(vals):.2f})",
                        edgecolor="black",
                        linewidth=0.5,
                        density=True,
                    )
            ax.set_title(f"Window Duration: {w}", fontsize=13, fontweight="bold")
            ax.set_xlabel("Decision Margin (Same - Cross)", fontsize=11)
            ax.set_ylabel("Probability Density", fontsize=11)
            ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper left", fontsize=9)

        fig.suptitle("Speaker Margin Distributions Across Window Lengths (T-292 Head-to-Head)", fontsize=15, fontweight="bold")
        plt.tight_layout()
        hist_path = out_dir / "margin_histograms.png"
        fig.savefig(hist_path, dpi=200)
        plt.close(fig)
        generated.append(hist_path)

    # 2. Metric Comparison Bars (EER, minDCF, Mean Margin)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(WINDOWS))
    width = 0.26

    backend_colors = {"cosine": "#d9534f", "mahalanobis": "#f0ad4e", "head": "#0275d8"}

    for idx, b_name in enumerate(BACKENDS):
        recs = [r for r in report.records if r.backend == b_name]
        recs.sort(key=lambda r: WINDOWS.index(r.window) if r.window in WINDOWS else 0)
        eers = [r.eer_percent for r in recs]
        dcfs = [r.min_dcf for r in recs]
        margins = [r.margin_mean for r in recs]
        pos = x + (idx - 1) * width

        ax1.bar(pos, eers, width, label=b_name, color=backend_colors[b_name], alpha=0.85, edgecolor="black", linewidth=0.5)
        ax2.bar(pos, dcfs, width, label=b_name, color=backend_colors[b_name], alpha=0.85, edgecolor="black", linewidth=0.5)
        ax3.bar(pos, margins, width, label=b_name, color=backend_colors[b_name], alpha=0.85, edgecolor="black", linewidth=0.5)

    for ax, title, ylabel in [
        (ax1, "Equal Error Rate (EER %)", "EER (%) [Lower is Better]"),
        (ax2, "Minimum DCF (p=0.01)", "minDCF [Lower is Better]"),
        (ax3, "Mean Decision Margin", "Margin [Higher is Better]"),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels(WINDOWS, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=10)

    fig.suptitle("Metric Summary: Cosine vs Mahalanobis vs Projection Head", fontsize=15, fontweight="bold")
    plt.tight_layout()
    bar_path = out_dir / "metric_comparison_bars.png"
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)
    generated.append(bar_path)

    # 3. DET curves across windows
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    with np.load(report.dataset_path, allow_pickle=True) as z:
        for ax, w in zip(axes, WINDOWS):
            data = load_window_data(z, w)
            triu_i, triu_j, is_same = build_trials(data, trial_mode=report.trial_mode)

            for b_name in BACKENDS:
                if b_name == "cosine":
                    scores = CosineBackend().score_pairs(data["embeddings"], triu_i, triu_j)
                elif b_name == "head":
                    scores = HeadBackend().score_pairs(data["head_out"], triu_i, triu_j)
                else:
                    m = MahalanobisBackend(center=True)
                    m.fit(data["embeddings"], data["speaker"])
                    scores = m.score_pairs(data["embeddings"], triu_i, triu_j)

                pos = np.sort(scores[is_same])
                neg = np.sort(scores[~is_same])
                all_th = np.sort(np.unique(np.concatenate([pos, neg])))

                # Downsample threshold evaluations for smooth, fast curve
                if len(all_th) > 300:
                    all_th = np.linspace(all_th[0], all_th[-1], 300)

                frr = np.searchsorted(pos, all_th, side="left") / max(len(pos), 1)
                far = (len(neg) - np.searchsorted(neg, all_th, side="left")) / max(len(neg), 1)

                ax.plot(far * 100.0, frr * 100.0, label=b_name, color=backend_colors[b_name], linewidth=2.0)

            ax.set_title(f"DET Curve: {w}", fontsize=13, fontweight="bold")
            ax.set_xlabel("False Alarm Rate (FAR %)", fontsize=11)
            ax.set_ylabel("False Reject Rate (FRR %)", fontsize=11)
            ax.set_xlim(-1, 50)
            ax.set_ylim(-1, 50)
            ax.plot([0, 50], [0, 50], color="gray", linestyle="--", alpha=0.5, label="EER Line")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(fontsize=9)

    fig.suptitle("Detection Error Tradeoff (DET) Curves Across Windows", fontsize=15, fontweight="bold")
    plt.tight_layout()
    det_path = out_dir / "det_curves.png"
    fig.savefig(det_path, dpi=200)
    plt.close(fig)
    generated.append(det_path)

    return generated


generate_plots = render_plots


# ---------------------------------------------------------------------------
# 6. Top-level Runner & CLI
# ---------------------------------------------------------------------------


def run_comparison(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = DEFAULT_OUT_DIR,
    windows: list[str] | tuple[str, ...] = WINDOWS,
    backends: list[str] | tuple[str, ...] = BACKENDS,
    trial_mode: str = "all_pairs",
    shrinkage: float = 0.30,
    center_mahalanobis: bool = True,
    p_target: float = 0.01,
    generate_plots: bool = True,
) -> MetricReport:
    """Execute complete comparison, write markdown table, JSON schema, and plots."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    report = evaluate_dataset(
        dataset_path=dataset_path,
        windows=windows,
        backends=backends,
        trial_mode=trial_mode,
        shrinkage=shrinkage,
        center_mahalanobis=center_mahalanobis,
        p_target=p_target,
    )

    # 1. Write Markdown table
    md_content = format_markdown_table(report.records)
    md_file = out_p / "metrics_table.md"
    md_file.write_text(md_content + "\n", encoding="utf-8")

    # 2. Write Summary JSON
    json_file = out_p / "summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    # 3. Generate PNG visualizations
    if generate_plots:
        render_plots(report, out_p)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Head-to-head speaker verification metric comparison harness (T-292)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to team_slices.npz dataset (default: eval/embeddings/T-291/team_slices.npz)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for markdown table, plots, and summary JSON (default: results/metrics/T-292)",
    )
    parser.add_argument(
        "--trial-mode",
        choices=["all_pairs", "cross_session"],
        default="all_pairs",
        help="Trial pairing mode (default: all_pairs)",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=list(WINDOWS),
        help="Window durations to evaluate (default: 0.5s 1.0s 2.0s)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(BACKENDS),
        help="Metric backends to evaluate (default: cosine mahalanobis head)",
    )
    parser.add_argument(
        "--shrinkage",
        type=float,
        default=0.30,
        help="Shrinkage factor for Mahalanobis metric (default: 0.30)",
    )
    parser.add_argument(
        "--no-center",
        action="store_true",
        help="Disable centroid centering in Mahalanobis whitening",
    )
    parser.add_argument(
        "--p-target",
        type=float,
        default=0.01,
        help="Prior probability of target speaker for minDCF (default: 0.01)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation",
    )

    args = parser.parse_args(argv)

    print("=" * 72)
    print("  T-292 METRIC COMPARISON HARNESS (COSINE vs MAHALANOBIS vs HEAD)")
    print("=" * 72)
    print(f"Dataset:    {args.dataset}")
    print(f"Output dir: {args.output_dir}")
    print(f"Windows:    {args.windows}")
    print(f"Backends:   {args.backends}")
    print(f"Trial mode: {args.trial_mode}")
    print("-" * 72)

    report = run_comparison(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        windows=args.windows,
        backends=args.backends,
        trial_mode=args.trial_mode,
        shrinkage=args.shrinkage,
        center_mahalanobis=not args.no_center,
        p_target=args.p_target,
        generate_plots=not args.no_plots,
    )

    print("\n" + format_markdown_table(report.records) + "\n")
    print(f"Artifacts saved to {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
