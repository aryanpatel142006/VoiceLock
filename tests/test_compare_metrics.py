"""Unit tests for metric-comparison eval harness (T-292).

Tests speaker-verification metrics head-to-head comparison:
- Stock ECAPA cosine
- Mahalanobis distance / metric (T-274 spec)
- Projection head cosine
Validates EER, minDCF, margin distributions, trial generation,
provenance schemas, and CLI execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from eval.compare_metrics import (
    CosineBackend,
    HeadBackend,
    MahalanobisBackend,
    MetricRecord,
    build_trials,
    compute_eer,
    compute_margins,
    compute_mindcf,
    evaluate_dataset,
    run_comparison,
)

# ---------------------------------------------------------------------------
# 1. EER unit tests
# ---------------------------------------------------------------------------


def test_compute_eer_perfect_separation():
    """Positives strictly above negatives -> EER = 0."""
    pos = np.array([0.7, 0.8, 0.9, 0.95], dtype=np.float32)
    neg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    eer, thresh = compute_eer(pos, neg)
    assert eer == pytest.approx(0.0, abs=1e-6)
    assert 0.4 <= thresh <= 0.7


def test_compute_eer_identical_distributions():
    """Identical distributions -> EER ~ 0.5."""
    rng = np.random.default_rng(42)
    pos = rng.normal(0.0, 1.0, size=500).astype(np.float32)
    neg = rng.normal(0.0, 1.0, size=500).astype(np.float32)
    eer, _ = compute_eer(pos, neg)
    assert eer == pytest.approx(0.5, abs=0.08)


def test_compute_eer_known_crossing():
    """Symmetric discrete distributions with known crossing."""
    pos = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    neg = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    eer, thresh = compute_eer(pos, neg)
    assert 0.30 <= eer <= 0.45
    assert 1.0 <= thresh <= 3.0


# ---------------------------------------------------------------------------
# 2. minDCF unit tests
# ---------------------------------------------------------------------------


def test_compute_mindcf_perfect_separation():
    """Separated scores yield minDCF = 0."""
    pos = np.array([0.8, 0.85, 0.9, 0.95], dtype=np.float32)
    neg = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    min_dcf, thresh = compute_mindcf(pos, neg, p_target=0.01)
    assert min_dcf == pytest.approx(0.0, abs=1e-6)
    assert 0.4 <= thresh <= 0.85


def test_compute_mindcf_cost_normalization():
    """For totally overlapping distributions, minDCF is bounded near 1.0."""
    rng = np.random.default_rng(123)
    pos = rng.normal(0.0, 1.0, size=1000).astype(np.float32)
    neg = rng.normal(0.0, 1.0, size=1000).astype(np.float32)
    min_dcf, _ = compute_mindcf(pos, neg, p_target=0.01)
    # Default decision (all reject or all accept) cost is 1.0
    assert min_dcf <= 1.0
    assert min_dcf >= 0.85


# ---------------------------------------------------------------------------
# 3. Trial generation tests
# ---------------------------------------------------------------------------


def test_build_trials_all_pairs():
    """All-pairs should generate N*(N-1)/2 pairs with matching same/cross labels."""
    speakers = np.array([0, 0, 1, 1], dtype=np.int64)
    sessions = np.array([0, 1, 0, 1], dtype=np.int64)
    data = {"speaker": speakers, "session": sessions}

    triu_i, triu_j, is_same = build_trials(data, trial_mode="all_pairs")
    assert len(triu_i) == 6  # 4 * 3 / 2
    assert len(triu_j) == 6
    assert len(is_same) == 6

    # Pairs: (0,1)-same, (0,2)-diff, (0,3)-diff, (1,2)-diff, (1,3)-diff, (2,3)-same
    expected_same = np.array([True, False, False, False, False, True])
    np.testing.assert_array_equal(is_same, expected_same)


def test_build_trials_cross_session():
    """Cross-session should exclude intra-session pairs."""
    speakers = np.array([0, 0, 1, 1], dtype=np.int64)
    sessions = np.array([0, 0, 1, 1], dtype=np.int64)
    data = {"speaker": speakers, "session": sessions}

    triu_i, _triu_j, is_same = build_trials(data, trial_mode="cross_session")
    # Only pairs between session 0 and session 1: (0,2), (0,3), (1,2), (1,3)
    assert len(triu_i) == 4
    # All these pairs cross speakers too:
    assert not np.any(is_same)


# ---------------------------------------------------------------------------
# 4. Backend tests
# ---------------------------------------------------------------------------


def test_cosine_backend_scores():
    """Cosine backend computes dot products on unit normalized vectors."""
    backend = CosineBackend()
    assert backend.backend_id == "cosine"
    assert backend.is_measured is True

    embs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    scores = backend.score_pairs(embs, np.array([0, 0, 1]), np.array([1, 2, 2]))
    # cos(0,1)=0, cos(0,2)=1, cos(1,2)=0
    np.testing.assert_allclose(scores, [0.0, 1.0, 0.0], atol=1e-5)


def test_head_backend_scores():
    """Head backend scores 32-d head outputs."""
    backend = HeadBackend()
    assert backend.backend_id == "head"
    assert backend.is_measured is True

    head = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    scores = backend.score_pairs(head, np.array([0]), np.array([1]))
    np.testing.assert_allclose(scores, [0.0], atol=1e-5)


def test_mahalanobis_backend_fit_and_scores():
    """Mahalanobis backend fits pooled covariance and whitens embeddings."""
    backend = MahalanobisBackend(shrinkage=0.30, center=True)
    assert backend.backend_id == "mahalanobis"
    assert backend.is_measured is True

    # Synthetic 2-speaker data with shared first dimension
    rng = np.random.default_rng(10)
    spk0 = rng.normal(0.0, 1.0, size=(20, 16)).astype(np.float32)
    spk1 = rng.normal(0.0, 1.0, size=(20, 16)).astype(np.float32)
    # Unit norm
    spk0 /= np.linalg.norm(spk0, axis=1, keepdims=True)
    spk1 /= np.linalg.norm(spk1, axis=1, keepdims=True)

    all_embs = np.vstack([spk0, spk1])
    speakers = np.array([0] * 20 + [1] * 20, dtype=np.int64)

    backend.fit(all_embs, speakers)
    scores = backend.score_pairs(all_embs, np.array([0, 20]), np.array([1, 21]))
    assert len(scores) == 2
    assert np.all(np.isfinite(scores))


# ---------------------------------------------------------------------------
# 5. Margin computation tests
# ---------------------------------------------------------------------------


def test_compute_margins_distribution():
    """Verify margin calculation yields valid summary percentiles."""
    n = 10
    triu_i, triu_j = np.triu_indices(n, k=1)
    speakers = np.array([0] * 5 + [1] * 5, dtype=np.int64)
    is_same = speakers[triu_i] == speakers[triu_j]

    # Artificial scores: same = 0.8, cross = 0.2
    scores = np.where(is_same, 0.8, 0.2).astype(np.float32)
    margin_res = compute_margins(scores, is_same, n_slices=n, triu_i=triu_i, triu_j=triu_j)

    assert margin_res["pairwise_margin"] == pytest.approx(0.6, abs=1e-5)
    assert margin_res["margin_mean"] == pytest.approx(0.6, abs=1e-5)
    assert margin_res["margin_p50"] == pytest.approx(0.6, abs=1e-5)
    assert margin_res["margin_std"] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 6. Provenance schema tests
# ---------------------------------------------------------------------------


def test_metric_record_provenance_schema():
    """MetricRecord must strictly include machine-readable provenance flags."""
    rec = MetricRecord(
        backend="mahalanobis",
        measured=True,
        window="0.5s",
        window_s=0.5,
        dim=192,
        n_slices=100,
        n_pos_trials=2000,
        n_neg_trials=2000,
        eer=0.095,
        eer_percent=9.5,
        eer_threshold=0.15,
        min_dcf=0.79,
        min_dcf_threshold=0.20,
        pairwise_margin=0.13,
        pos_score_mean=0.23,
        neg_score_mean=0.10,
        margin_mean=0.13,
        margin_std=0.05,
        margin_p05=0.02,
        margin_p25=0.08,
        margin_p50=0.14,
        margin_p75=0.18,
        margin_p95=0.22,
    )

    as_dict = rec.to_dict()
    assert as_dict["backend"] in {"cosine", "mahalanobis", "head"}
    assert as_dict["measured"] is True
    assert isinstance(as_dict["measured"], bool)

    # Valid JSON
    s = json.dumps(as_dict)
    loaded = json.loads(s)
    assert loaded["backend"] == "mahalanobis"
    assert loaded["measured"] is True


# ---------------------------------------------------------------------------
# 7. Integration test on synthetic dataset
# ---------------------------------------------------------------------------


def test_evaluate_dataset_synthetic(tmp_path: Path):
    """End-to-end evaluation on synthetic npz archive."""
    npz_path = tmp_path / "test_slices.npz"

    # Construct synthetic dataset with two windows
    rng = np.random.default_rng(99)
    n = 20
    embs = rng.normal(0.0, 1.0, size=(n, 192)).astype(np.float32)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    head = rng.normal(0.0, 1.0, size=(n, 32)).astype(np.float32)
    head /= np.linalg.norm(head, axis=1, keepdims=True)
    spk = np.array([0] * 10 + [1] * 10, dtype=np.int64)
    sess = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5, dtype=np.int64)

    np.savez_compressed(
        npz_path,
        **{
            "windows": np.array(["0.5s", "1.0s"]),
            "speakers": np.array(["aryan", "joseph"]),
            "embeddings/0.5s": embs,
            "head_out/0.5s": head,
            "meta/0.5s/speaker": spk,
            "meta/0.5s/session": sess,
            "embeddings/1.0s": embs,
            "head_out/1.0s": head,
            "meta/1.0s/speaker": spk,
            "meta/1.0s/session": sess,
        },
    )

    report = evaluate_dataset(
        dataset_path=npz_path,
        windows=["0.5s"],
        backends=["cosine", "mahalanobis", "head"],
    )

    assert len(report.records) == 3
    backends_seen = {r.backend for r in report.records}
    assert backends_seen == {"cosine", "mahalanobis", "head"}

    for r in report.records:
        assert r.measured is True
        assert 0.0 <= r.eer <= 1.0
        assert 0.0 <= r.min_dcf <= 1.0


# ---------------------------------------------------------------------------
# 8. Full run on T-291 real dataset if available
# ---------------------------------------------------------------------------


def test_run_comparison_on_t291_real(tmp_path: Path):
    """Run comparison on eval/embeddings/T-291/team_slices.npz and write outputs."""
    t291_path = Path("eval/embeddings/T-291/team_slices.npz")
    if not t291_path.is_file():
        pytest.skip("eval/embeddings/T-291/team_slices.npz not present in checkout")

    out_dir = tmp_path / "T-292"
    report = run_comparison(
        dataset_path=t291_path,
        output_dir=out_dir,
        windows=["0.5s", "1.0s", "2.0s"],
        generate_plots=True,
    )

    assert len(report.records) == 9  # 3 windows * 3 backends

    # Check output artifacts
    assert (out_dir / "metrics_table.md").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "margin_histograms.png").is_file()
    assert (out_dir / "det_curves.png").is_file()
    assert (out_dir / "metric_comparison_bars.png").is_file()

    # Verify table formatting
    md = (out_dir / "metrics_table.md").read_text()
    assert "| Window | Backend | Measured |" in md
    assert "EER (%)" in md
    assert "mahalanobis" in md
    assert "cosine" in md
    assert "head" in md

    # Verify summary JSON machine-readable flags
    with open(out_dir / "summary.json") as f:
        meta = json.load(f)
    assert meta["task"] == "T-292"
    assert len(meta["records"]) == 9
    for rec in meta["records"]:
        assert rec["backend"] in {"cosine", "mahalanobis", "head"}
        assert rec["measured"] is True


def test_cli_execution_main(tmp_path: Path):
    """Test CLI main() with arguments."""
    from eval.compare_metrics import main

    t291_path = Path("eval/embeddings/T-291/team_slices.npz")
    if not t291_path.is_file():
        pytest.skip("eval/embeddings/T-291/team_slices.npz not present in checkout")

    out_dir = tmp_path / "cli_out"
    ret = main([
        "--dataset", str(t291_path),
        "--output-dir", str(out_dir),
        "--windows", "0.5s",
        "--backends", "cosine", "head",
        "--no-plots",
    ])
    assert ret == 0
    assert (out_dir / "metrics_table.md").is_file()
    assert (out_dir / "summary.json").is_file()
    assert not (out_dir / "margin_histograms.png").exists()  # --no-plots respected

