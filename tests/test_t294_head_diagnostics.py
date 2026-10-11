"""Unit tests for T-294 projection head theory diagnostics (head_diagnostics.py).

Tests:
1. Synthetic dataset generator properties (unit norms, separation).
2. Pairwise cosine margin calculation without self-pair diagonal inflation.
3. Shuffled-label diagnostic execution and output schema.
4. PAC-style generalization margin bound computation.
5. Failure-mode stress simulation functions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_EXP_DIR = Path(__file__).resolve().parent.parent / "experiments" / "theory" / "T-294"
_MODULE_PATH = _EXP_DIR / "head_diagnostics.py"


def _load_diagnostics_module():
    spec = importlib.util.spec_from_file_location("head_diagnostics", _MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {_MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["head_diagnostics"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def diag():
    return _load_diagnostics_module()


def test_generate_synthetic_embeddings(diag):
    """Synthetic dataset generates expected shapes, unit norms, and labels."""
    embs, labels, takes = diag.generate_synthetic_embeddings(
        n_spk0_train=20,
        n_spk1_train=20,
        n_spk0_test=10,
        n_spk1_test=10,
        in_dim=192,
        seed=123,
    )
    assert embs["train"].shape == (40, 192)
    assert embs["test"].shape == (20, 192)
    assert labels["train"].shape == (40,)
    assert labels["test"].shape == (20,)
    assert takes["train"].shape == (40,)
    assert takes["test"].shape == (20,)

    # Norms must be strictly unit
    train_norms = np.linalg.norm(embs["train"], axis=-1)
    test_norms = np.linalg.norm(embs["test"], axis=-1)
    assert np.allclose(train_norms, 1.0, atol=1e-5)
    assert np.allclose(test_norms, 1.0, atol=1e-5)


def test_pairwise_margin_excludes_diagonal(diag):
    """Pairwise margin excludes self-similarity diagonal to prevent small-N inflation."""
    # Orthogonal vectors for speaker 0
    spk0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    # Antipodal vectors for speaker 1
    spk1 = np.array([[-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)

    X = np.vstack([spk0, spk1])
    y = np.array([0, 0, 1, 1], dtype=np.int64)

    res = diag.compute_pairwise_margin(X, y)
    # intra cosine: between (1,0) and (0,1) is 0; between (-1,0) and (0,-1) is 0
    assert pytest.approx(res["intra"], abs=1e-5) == 0.0
    # cross cosine: between (1,0) and (-1,0) is -1, (1,0) and (0,-1) is 0, etc. Mean is -0.5
    assert pytest.approx(res["cross"], abs=1e-5) == -0.5
    # margin = intra - cross = 0 - (-0.5) = 0.5
    assert pytest.approx(res["margin"], abs=1e-5) == 0.5


def test_pac_generalization_bound(diag):
    """PAC margin gap bound decreases with sample size N and increases with parameter complexity."""
    bound_n100 = diag.compute_pac_margin_bound(n_samples=100, n_params=14560, delta=0.05, gamma=0.5)
    bound_n1000 = diag.compute_pac_margin_bound(n_samples=1000, n_params=14560, delta=0.05, gamma=0.5)
    bound_higher_dim = diag.compute_pac_margin_bound(n_samples=100, n_params=50000, delta=0.05, gamma=0.5)

    assert bound_n100["generalization_bound"] > 0
    assert bound_n1000["generalization_bound"] < bound_n100["generalization_bound"]
    assert bound_higher_dim["generalization_bound"] > bound_n100["generalization_bound"]
    assert "rademacher_term" in bound_n100
    assert "confidence_term" in bound_n100


def test_run_shuffled_diagnostic_fast(diag):
    """Shuffled label diagnostic runs fast on synthetic data and returns valid structure."""
    res = diag.run_shuffled_diagnostic(
        dataset="synthetic",
        epochs=10,
        seed=42,
        n_spk0_train=25,
        n_spk1_train=25,
        n_spk0_test=15,
        n_spk1_test=15,
    )
    assert "stock" in res
    assert "head_true" in res
    assert "head_shuffled" in res
    assert "pac_bound" in res
    assert "hypotheses" in res

    # True labels head should have higher test margin than shuffled labels head
    assert res["head_true"]["test_margin"] > res["head_shuffled"]["test_margin_eval_true"]
    assert res["head_shuffled"]["test_margin_eval_shuffled"] < 0.25


def test_simulate_failure_modes(diag):
    """Failure mode stress testing produces expected degradation metrics."""
    modes = diag.simulate_failure_mode_stresses(seed=42)
    assert len(modes) >= 5
    expected_keys = {"reverb", "codec", "overlap", "pitch", "whisper"}
    assert expected_keys.issubset(set(modes.keys()))
    for data in modes.values():
        assert "baseline_margin" in data
        assert "stressed_margin" in data
        assert "margin_drop_pct" in data
        assert "hypothesis" in data
        assert data["stressed_margin"] < data["baseline_margin"]
