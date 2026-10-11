"""Numpy-only loader for eval/embeddings/T-291/team_slices.npz.

Provides helper routines to inspect and load the team embedding analysis dataset
built for muse-c's T-249 speaker calibration and margin analysis work.

Dataset Schema (stored in team_slices.npz):
  embeddings/<w>                 (N, 192) float32 — ECAPA-TDNN 192-d unit embeddings
  head_out/<w>                   (N, 32) float32  — TeamProjectionHead 32-d unit embeddings
  meta/<w>/speaker               (N,) int64       — 0: aryan, 1: joseph
  meta/<w>/speaker_id            (N,) str         — "aryan" | "joseph"
  meta/<w>/speaker_name          (N,) str         — "aryan" | "joseph"
  meta/<w>/session               (N,) int64       — recording take index per speaker
  meta/<w>/recording             (N,) str         — source wav filename
  meta/<w>/start_s               (N,) float32     — offset in source recording (seconds)
  meta/<w>/window_length         (N,) float32     — duration (0.5, 1.0, 2.0 seconds)
  meta/<w>/snr_db                (N,) float32     — per-slice SNR estimate (dB)
  meta/<w>/cosine_margin_stock   (N,) float32     — per-slice stock cosine margin
  meta/<w>/projected_margin_head (N,) float32     — per-slice projected head margin
  margins_json                   JSON string      — summary margin statistics per window
  provenance_json                JSON string      — full dataset provenance
where <w> in {"0.5s", "1.0s", "2.0s"}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

WINDOWS = ("0.5s", "1.0s", "2.0s")
DEFAULT_PATH = Path(__file__).parent / "team_slices.npz"


def _open(path: str | Path | None = None) -> np.lib.npyio.NpzFile:
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} not found. Extract it with: .venv/bin/python eval/embeddings/T-291/extract_slices.py"
        )
    return np.load(p, allow_pickle=True)


def provenance(path: str | Path | None = None) -> dict[str, Any]:
    """Retrieve per-field dataset provenance and metadata."""
    with _open(path) as z:
        return json.loads(str(z["provenance_json"]))


def load_slices(window: str = "0.5s", path: str | Path | None = None) -> dict[str, Any]:
    """Load all embeddings and per-slice metadata for a specific window duration."""
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    with _open(path) as z:
        return {
            "embeddings": z[f"embeddings/{window}"],
            "head_out": z[f"head_out/{window}"],
            "speaker": z[f"meta/{window}/speaker"],
            "speaker_id": z[f"meta/{window}/speaker_id"],
            "speaker_name": z[f"meta/{window}/speaker_name"],
            "session": z[f"meta/{window}/session"],
            "recording": z[f"meta/{window}/recording"],
            "start_s": z[f"meta/{window}/start_s"],
            "window_length": z[f"meta/{window}/window_length"],
            "snr_db": z[f"meta/{window}/snr_db"],
            "cosine_margin_stock": z[f"meta/{window}/cosine_margin_stock"],
            "projected_margin_head": z[f"meta/{window}/projected_margin_head"],
            "window": window,
        }


def _pairwise_margin_metrics(embs: np.ndarray, speaker: np.ndarray) -> dict[str, float]:
    """Compute aggregate same-speaker, cross-speaker, and margin cosines."""
    n = len(embs)
    same, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            cs = float(embs[i] @ embs[j])
            if speaker[i] == speaker[j]:
                same.append(cs)
            else:
                cross.append(cs)
    same_arr = np.array(same) if same else np.empty(0, dtype=np.float32)
    cross_arr = np.array(cross) if cross else np.empty(0, dtype=np.float32)
    mean_same = float(np.mean(same_arr)) if len(same_arr) else float("nan")
    mean_cross = float(np.mean(cross_arr)) if len(cross_arr) else float("nan")
    return {
        "n_same": len(same_arr),
        "n_cross": len(cross_arr),
        "mean_same": mean_same,
        "mean_cross": mean_cross,
        "margin": mean_same - mean_cross if np.isfinite(mean_same) and np.isfinite(mean_cross) else float("nan"),
    }


def load_margins(path: str | Path | None = None) -> dict[str, Any]:
    """Load summary margin distributions (stock cosine and projected head) per window."""
    with _open(path) as z:
        if "margins_json" in z:
            margins_summary = json.loads(str(z["margins_json"]))
            out = {}
            for w in WINDOWS:
                w_summary = margins_summary.get(w, {})
                d = load_slices(w, path)
                stock_pair = _pairwise_margin_metrics(d["embeddings"], d["speaker"])
                head_pair = _pairwise_margin_metrics(d["head_out"], d["speaker"])
                # Combine pairwise stats with per-window distribution stats
                out[w] = {
                    "stock_cosine": {
                        **stock_pair,
                        **w_summary.get("stock_cosine_margin", {}),
                    },
                    "head_projected": {
                        **head_pair,
                        **w_summary.get("projected_head_margin", {}),
                    },
                    "n_slices": w_summary.get("n_slices", len(d["embeddings"])),
                }
            return out

    # Fallback to computing on the fly if margins_json is absent
    out = {}
    for w in WINDOWS:
        d = load_slices(w, path)
        stock = _pairwise_margin_metrics(d["embeddings"], d["speaker"])
        proj = _pairwise_margin_metrics(d["head_out"], d["speaker"])
        out[w] = {
            "stock_cosine": stock,
            "head_projected": proj,
            "n_slices": len(d["embeddings"]),
        }
    return out


def train_test_split_by_session(
    window: str = "0.5s",
    test_session: int = 1,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Provide a session-based split across takes."""
    d = load_slices(window, path)
    sessions = d["session"]
    speakers = d["speaker"]

    test_mask = sessions == test_session
    train_mask = ~test_mask

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_speakers": np.unique(speakers[train_idx]).tolist(),
        "test_speakers": np.unique(speakers[test_idx]).tolist(),
        "window": window,
    }
