"""Tests for T-291 embedding analysis dataset, loader, and stats script."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "eval" / "embeddings" / "T-291"))
import loader


def test_team_slices_file_exists():
    path = REPO_ROOT / "eval/embeddings/T-291/team_slices.npz"
    assert path.is_file(), f"{path} does not exist"
    assert path.stat().st_size > 100_000, "team_slices.npz should be at least 100 KB"


def test_team_slices_all_windows_present():
    path = REPO_ROOT / "eval/embeddings/T-291/team_slices.npz"
    z = np.load(path, allow_pickle=True)
    assert set(z["windows"]) == {"0.5s", "1.0s", "2.0s"}
    assert set(z["speakers"]) == {"aryan", "joseph"}


@pytest.mark.parametrize("window,win_sec", [("0.5s", 0.5), ("1.0s", 1.0), ("2.0s", 2.0)])
def test_slice_shapes_and_types(window: str, win_sec: float):
    d = loader.load_slices(window)
    n = d["embeddings"].shape[0]
    assert n > 20, f"Expected substantial number of slices for {window}, got {n}"

    # Embeddings: (N, 192) float32, unit norm
    assert d["embeddings"].shape == (n, 192)
    assert d["embeddings"].dtype == np.float32
    norms = np.linalg.norm(d["embeddings"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    # Projection head: (N, 32) float32, unit norm
    assert d["head_out"].shape == (n, 32)
    assert d["head_out"].dtype == np.float32
    head_norms = np.linalg.norm(d["head_out"], axis=1)
    np.testing.assert_allclose(head_norms, 1.0, atol=1e-5)

    # Per-window metadata
    assert d["speaker"].shape == (n,)
    assert set(np.unique(d["speaker"])) == {0, 1}

    assert d["speaker_name"].shape == (n,)
    assert set(np.unique(d["speaker_name"])) == {"aryan", "joseph"}

    assert d["window_length"].shape == (n,)
    np.testing.assert_allclose(d["window_length"], win_sec)

    assert d["snr_db"].shape == (n,)
    assert np.all(np.isfinite(d["snr_db"]))
    assert np.all(d["snr_db"] > 0), "Speech slices should have positive SNR estimates"

    assert d["cosine_margin_stock"].shape == (n,)
    assert np.all(np.isfinite(d["cosine_margin_stock"]))

    assert d["projected_margin_head"].shape == (n,)
    assert np.all(np.isfinite(d["projected_margin_head"]))


def test_margin_collapse_reproduced():
    """Verify that stock ECAPA exhibits the 0.45 -> 0.15 margin collapse."""
    margins = loader.load_margins()

    m05 = margins["0.5s"]["stock_cosine"]["mean"]
    m10 = margins["1.0s"]["stock_cosine"]["mean"]
    m20 = margins["2.0s"]["stock_cosine"]["mean"]

    # Margin must monotonically increase with window length
    assert m05 < m10 < m20, f"Expected margin collapse as window shrinks: 0.5s={m05:.3f}, 1.0s={m10:.3f}, 2.0s={m20:.3f}"

    # 0.5s margin collapses to around 0.12 - 0.15
    assert 0.08 <= m05 <= 0.18, f"Expected 0.5s margin in collapse range [0.08, 0.18], got {m05:.3f}"

    # 2.0s margin is in the 0.35 - 0.45 baseline range
    assert 0.32 <= m20 <= 0.48, f"Expected 2.0s margin in baseline range [0.32, 0.48], got {m20:.3f}"

    # Projection head expands the collapsed 0.5s margin
    h05 = margins["0.5s"]["head_projected"]["mean"]
    assert h05 > 0.80, f"Head should expand 0.5s margin to >0.80, got {h05:.3f}"
    assert h05 / m05 > 4.0, f"Head should provide at least 4x margin expansion, got {h05/m05:.2f}x"


def test_loader_invalid_window_raises():
    with pytest.raises(ValueError, match="window must be one of"):
        loader.load_slices("3.0s")


def test_loader_provenance():
    prov = loader.provenance()
    assert "ecapa" in prov["models"]
    assert "head" in prov["models"]
    assert "aryan" in prov["source_recordings"]
    assert "joseph" in prov["source_recordings"]


def test_loader_train_test_split():
    sp = loader.train_test_split_by_session(window="0.5s", test_session=1)
    assert len(sp["train_idx"]) > 0
    assert len(sp["test_idx"]) > 0
    assert len(set(sp["train_idx"]).intersection(set(sp["test_idx"]))) == 0


def test_stats_script_execution():
    stats_script = REPO_ROOT / "eval/embeddings/T-291/stats.py"
    res = subprocess.run(
        [sys.executable, str(stats_script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"stats.py failed with: {res.stderr}"
    assert "T-291 SPEAKER EMBEDDING ANALYSIS" in res.stdout
    assert "VERIFICATION OF THE 0.45 -> 0.15 MARGIN COLLAPSE CLAIM" in res.stdout
    assert "Stock margin collapses from" in res.stdout
