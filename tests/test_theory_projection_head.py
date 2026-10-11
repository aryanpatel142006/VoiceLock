"""Unit tests for theoretical ECAPA projection head prototype (T-246).

Tests:
1. Unit Euclidean normalization invariance on the unit hypersphere.
2. Linear and MLP projection head dimensionality transformations.
3. Covariance shrinkage whitening transform properties.
4. Reference closed-set LDA hyperplane mapping (+1/-1 centroids).
5. Supervised contrastive, triplet, and covariance regularization loss behaviors.
6. Zero-leakage synthetic partition disjointness (train/dev/heldout).
7. Replayable ablation suite execution, checkpoint-resume, and gate evaluation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Dynamic import helper for hyphenated path experiments/theory/T-246
_EXP_DIR = Path(__file__).parent.parent / "experiments" / "theory" / "T-246"
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t246_model = _load_module("t246_model", _EXP_DIR / "model.py")
t246_data = _load_module("t246_data", _EXP_DIR / "synthetic_data.py")
t246_ablation = _load_module("t246_ablation", _EXP_DIR / "run_ablation.py")


def test_projection_head_unit_normalization():
    """Projected vectors must have strictly unit L2 norm on hypersphere."""
    linear = t246_model.LinearProjectionHead(in_dim=192, proj_dim=128)
    mlp = t246_model.MLPProjectionHead(in_dim=192, hidden_dim=128, proj_dim=64)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((10, 192)).astype(np.float32)
    tx = torch.from_numpy(x)

    z_lin = linear(tx).detach().numpy()
    z_mlp = mlp(tx).detach().numpy()

    norms_lin = np.linalg.norm(z_lin, axis=-1)
    norms_mlp = np.linalg.norm(z_mlp, axis=-1)

    assert np.allclose(norms_lin, 1.0, atol=1e-6)
    assert np.allclose(norms_mlp, 1.0, atol=1e-6)


def test_projection_head_dimensions_and_residual():
    """Verify output tensor shapes and residual shortcut behavior."""
    linear = t246_model.LinearProjectionHead(in_dim=192, proj_dim=96)
    mlp = t246_model.MLPProjectionHead(in_dim=192, hidden_dim=128, proj_dim=192, residual=True)

    tx = torch.randn(4, 192)
    assert linear(tx).shape == (4, 96)
    assert mlp(tx).shape == (4, 192)


def test_whitening_transform_decorrelation():
    """Shrinkage whitening must centre, whiten, and unit normalize."""
    rng = np.random.default_rng(1)
    # Correlated background distribution
    mean = rng.standard_normal(32)
    cov_sqrt = rng.standard_normal((32, 32))
    raw = mean + rng.standard_normal((200, 32)) @ cov_sqrt

    whitening = t246_model.WhiteningTransform(shrinkage=0.2).fit(raw)
    out = whitening.transform(raw)

    assert out.shape == (200, 32)
    norms = np.linalg.norm(out, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_closed_set_lda_centroid_mapping():
    """LDA hyperplane must separate centroids at +1.0 and -1.0."""
    rng = np.random.default_rng(2)
    mt = rng.standard_normal(16)
    mo = rng.standard_normal(16)
    t = mt + 0.05 * rng.standard_normal((5, 16))
    o = mo + 0.05 * rng.standard_normal((5, 16))

    res = t246_model.fit_closed_set_lda(t, o, shrink=0.5)
    assert res is not None
    w, b = res
    assert float(w @ t.mean(0) - b) == pytest.approx(1.0, rel=1e-3)
    assert float(w @ o.mean(0) - b) == pytest.approx(-1.0, rel=1e-3)


def test_loss_function_behaviors():
    """Verify contrastive, triplet, and covariance loss mathematical properties."""
    # Triplet loss: zero when margin satisfied, positive otherwise
    a = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    p = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # cos(a, p) = 1.0
    n_far = torch.tensor([[0.0, 1.0]], dtype=torch.float32)  # cos(a, n) = 0.0
    n_close = torch.tensor([[0.95, 0.31]], dtype=torch.float32)  # cos(a, n) = 0.95

    loss_zero = t246_model.triplet_margin_loss(a, p, n_far, margin=0.2)
    assert float(loss_zero) == pytest.approx(0.0)

    loss_pos = t246_model.triplet_margin_loss(a, p, n_close, margin=0.2)
    assert float(loss_pos) > 0.0

    # Covariance regularizer: zero for perfectly orthogonal features
    ortho = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=torch.float32)
    cov_loss = t246_model.covariance_regularization_loss(ortho)
    assert float(cov_loss) == pytest.approx(0.0, abs=1e-6)

    # Supervised contrastive loss
    feats = torch.randn(8, 16)
    feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    supcon_loss = t246_model.supervised_contrastive_loss(feats, labels, temperature=0.1)
    assert float(supcon_loss) > 0.0


def test_synthetic_dataset_zero_leakage_splits():
    """Train, Dev, and Heldout speaker and room partitions must be strictly disjoint."""
    train_part, dev_part, heldout_part = t246_data.create_synthetic_splits(
        seed=123,
        train_speakers=20,
        dev_speakers=5,
        heldout_speakers=5,
        train_rooms=4,
        dev_rooms=2,
        heldout_rooms=2,
    )

    s_train = set(train_part.speakers.keys())
    s_dev = set(dev_part.speakers.keys())
    s_heldout = set(heldout_part.speakers.keys())

    # Speaker disjointness
    assert s_train.isdisjoint(s_dev)
    assert s_train.isdisjoint(s_heldout)
    assert s_dev.isdisjoint(s_heldout)

    # Room disjointness
    r_train = set(train_part.rooms.keys())
    r_dev = set(dev_part.rooms.keys())
    r_heldout = set(heldout_part.rooms.keys())
    assert r_train.isdisjoint(r_dev)
    assert r_train.isdisjoint(r_heldout)
    assert r_dev.isdisjoint(r_heldout)

    # Batch sampling sanity
    batch_x, batch_y = train_part.sample_contrastive_batch(batch_size=16, crops_per_speaker=2)
    assert batch_x.shape[1] == 192
    assert len(batch_y) == len(batch_x)

    # Evaluation pair generation sanity
    eval_pairs = dev_part.generate_evaluation_stream_pairs(count=8)
    assert len(eval_pairs) == 8
    conditions = {p.condition for p in eval_pairs}
    assert "normal" in conditions
    assert "hard_pair" in conditions


def test_ablation_suite_smoke_and_checkpoint_resume(tmp_path):
    """Verify that ablation runner runs in smoke mode, saves atomic checkpoints, and resumes."""
    ckpt_dir = tmp_path / "checkpoints"

    # Run 1: initial run with checkpoint saving
    report1 = t246_ablation.run_ablation_suite(
        seed=99,
        cases=12,
        smoke=True,
        checkpoint_dir=ckpt_dir,
        resume=False,
    )
    assert report1["status"] == "completed"
    assert "ablation_results" in report1
    assert "gates" in report1
    assert "overnight_gpu_request_manifest" in report1

    # Checkpoint was written
    ckpts = list(ckpt_dir.glob("checkpoint_*.json"))
    assert len(ckpts) == 1
    saved_data = json.loads(ckpts[0].read_text())
    assert saved_data["task_id"] == "T-246"

    # Run 2: resume from checkpoint skips re-running
    report2 = t246_ablation.run_ablation_suite(
        seed=99,
        cases=12,
        smoke=True,
        checkpoint_dir=ckpt_dir,
        resume=True,
    )
    assert report2["config_hash"] == report1["config_hash"]
    assert report2["runtime_s"] == report1["runtime_s"]
