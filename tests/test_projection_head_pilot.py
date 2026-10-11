"""Tests for Aryan vs Joseph dedicated contrastive projection head pilot (T-273).

All tests run fully offline without network access or model weight downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.projection_head_pilot import (
    ArcFaceLoss,
    ContrastiveProjectionLoss,
    TeamProjectionHead,
    compute_metric_stats,
    evaluate_flips_on_heldout,
    generate_synthetic_team_dataset,
    measure_inference_overhead,
    run_projection_head_pilot,
    slice_audio,
    train_projection_head,
)


def test_projection_head_architecture_and_shapes() -> None:
    head = TeamProjectionHead(in_dim=192, hidden_dim=64, out_dim=32)
    batch_size = 8
    x = torch.randn(batch_size, 192)
    z = head(x)

    assert z.shape == (batch_size, 32)
    norms = torch.linalg.norm(z, dim=-1)
    np.testing.assert_allclose(norms.detach().numpy(), np.ones(batch_size), atol=1e-5)


def test_projection_head_parameter_count() -> None:
    head = TeamProjectionHead(in_dim=192, hidden_dim=64, out_dim=32)
    total_params = sum(p.numel() for p in head.parameters())
    # 192*64 + 64 (linear1) + 64*2 (layernorm) + 64*32 + 32 (linear2) = 12288 + 64 + 128 + 2048 + 32 = 14560
    assert total_params == 14560


def test_arcface_loss_computation_and_gradients() -> None:
    loss_fn = ArcFaceLoss(in_features=32, num_classes=2, scale=16.0, margin=0.5)
    raw = torch.randn(6, 32, requires_grad=True)
    features = torch.nn.functional.normalize(raw, p=2, dim=-1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    loss, logits = loss_fn(features, labels)
    assert loss.item() > 0.0
    assert logits.shape == (6, 2)

    loss.backward()
    assert raw.grad is not None
    assert loss_fn.weight.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_contrastive_loss_computation_and_gradients() -> None:
    loss_fn = ContrastiveProjectionLoss(target_neg_cos=-0.50, neg_weight=2.0)
    raw = torch.randn(6, 32, requires_grad=True)
    features = torch.nn.functional.normalize(raw, p=2, dim=-1)
    labels = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    loss = loss_fn(features, labels)
    assert loss.item() >= 0.0

    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()


def test_slice_audio_filtering() -> None:
    sr = 16000
    # 1.0 s of sine wave at amplitude 0.2
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    # 1.0 s of near-silence
    silence = np.zeros(sr, dtype=np.float32)
    audio = np.concatenate([tone, silence])

    chunks = slice_audio(audio, sr=sr, chunk_s=0.5, step_s=0.25)
    assert len(chunks) > 0
    # Every returned chunk must have length = 8000 samples and meet energy threshold
    for c in chunks:
        assert len(c) == int(0.5 * sr)
        assert np.max(np.abs(c)) >= 0.02


def test_synthetic_team_dataset_properties() -> None:
    X_a_train, X_j_train, X_a_test, X_j_test = generate_synthetic_team_dataset(
        seed=123, num_train=40, num_test=20, baseline_similarity=0.60
    )

    assert X_a_train.shape == (40, 192)
    assert X_j_train.shape == (40, 192)
    assert X_a_test.shape == (20, 192)
    assert X_j_test.shape == (20, 192)

    # Unit norms
    norms = np.linalg.norm(X_a_train, axis=-1)
    np.testing.assert_allclose(norms, np.ones(40), atol=1e-5)

    stats = compute_metric_stats(X_a_test, X_j_test)
    assert -0.2 < stats["mean_cross_cosine"] < 0.8
    assert stats["angular_distance_deg"] > 30.0


def test_train_projection_head_contrastive_convergence() -> None:
    X_a_tr, X_j_tr, _, _ = generate_synthetic_team_dataset(seed=42, num_train=30, num_test=10)
    _model, meta = train_projection_head(
        X_a_tr,
        X_j_tr,
        loss_type="contrastive",
        epochs=30,
        lr=2e-3,
        device="cpu",
        target_neg_cos=-0.50,
    )

    assert meta["loss_type"] == "contrastive"
    assert meta["final_loss"] < meta["initial_loss"]


def test_train_projection_head_arcface_convergence() -> None:
    X_a_tr, X_j_tr, _, _ = generate_synthetic_team_dataset(seed=42, num_train=30, num_test=10)
    _model, meta = train_projection_head(
        X_a_tr,
        X_j_tr,
        loss_type="arcface",
        epochs=30,
        lr=2e-3,
        device="cpu",
    )

    assert meta["loss_type"] == "arcface"
    assert meta["final_loss"] < meta["initial_loss"]


def test_evaluate_flips_on_heldout() -> None:
    model = TeamProjectionHead()
    # Mock orthogonal test embeddings
    X_a_test = np.zeros((10, 192), dtype=np.float32)
    X_a_test[:, 0] = 1.0
    X_j_test = np.zeros((10, 192), dtype=np.float32)
    X_j_test[:, 1] = 1.0

    proto_a = np.zeros(192, dtype=np.float32)
    proto_a[0] = 1.0
    proto_j = np.zeros(192, dtype=np.float32)
    proto_j[1] = 1.0

    res = evaluate_flips_on_heldout(model, X_a_test, X_j_test, proto_a, proto_j)
    assert "baseline" in res
    assert "projected" in res
    assert res["baseline"]["aryan_total"] == 10
    assert res["baseline"]["joseph_total"] == 10


def test_inference_overhead_measurement() -> None:
    model = TeamProjectionHead()
    overhead = measure_inference_overhead(model, iters=100)
    assert overhead["param_count"] == 14560.0
    # Latency should comfortably be under 5 ms even on modest CPU
    assert overhead["latency_ms"] < 5.0


def test_full_pilot_pipeline_synthetic_mode(tmp_path: Path) -> None:
    weights_path = tmp_path / "weights.pt"
    report = run_projection_head_pilot(
        loss_type="contrastive",
        use_synthetic=True,
        epochs=30,
        lr=2e-3,
        device="cpu",
        save_weights_path=weights_path,
    )

    assert report["pilot_id"] == "T-273"
    assert report["data_source"] == "synthetic"
    assert "metric_geometry" in report
    assert "heldout_speaker_flips" in report
    assert "runtime_overhead" in report
    assert "gates" in report
    assert weights_path.exists()
