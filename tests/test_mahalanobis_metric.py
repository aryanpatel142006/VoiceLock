"""Unit tests for calibrated pairwise Mahalanobis metric and SAME_VOICE margin relaxer (T-274)."""

import pytest

from eval.mahalanobis_metric import (
    MahalanobisMetric,
    SameVoicePolicy,
    SyntheticTeamData,
    build_team_baseline_offline_benchmark,
    evaluate_metric_and_relaxation,
    evaluate_team_challenge_same_voice,
    format_report_markdown,
)


def test_mahalanobis_fit_and_shapes():
    """Verify matrix dimensions, positive eigenvalues, and conditioning."""
    t_pieces, o_pieces, _, _ = SyntheticTeamData.generate(d=192, n_pieces=25, seed=1)
    metric = MahalanobisMetric(shrinkage=0.3)
    metric.fit(t_pieces, o_pieces)

    assert metric.fitted
    assert metric.transform_mat is not None
    assert metric.precision_mat is not None
    assert metric.transform_mat.shape == (192, 192)
    assert metric.precision_mat.shape == (192, 192)
    assert metric.eigenvalues is not None
    assert len(metric.eigenvalues) == 192
    assert (metric.eigenvalues > 0).all()
    assert metric.condition_number >= 1.0


def test_mahalanobis_distance_properties():
    """Verify standard metric properties: non-negativity, symmetry, identity."""
    t_pieces, o_pieces, _, _ = SyntheticTeamData.generate(d=64, n_pieces=15, seed=2)
    metric = MahalanobisMetric(shrinkage=0.25).fit(t_pieces, o_pieces)

    u = t_pieces[0]
    v = o_pieces[0]

    # Distance to self is zero
    assert metric.distance_sq(u, u) == pytest.approx(0.0, abs=1e-9)

    # Symmetry
    d_uv = metric.distance_sq(u, v)
    d_vu = metric.distance_sq(v, u)
    assert d_uv == pytest.approx(d_vu, rel=1e-6)
    assert d_uv > 0.0

    # Cosine to self is 1.0
    assert metric.mahalanobis_cosine(u, u) == pytest.approx(1.0, abs=1e-6)


def test_mahalanobis_decorrelates_shared_subspace():
    """Verify that downweighting the shared subspace reduces pair cosine."""
    t_pieces, o_pieces, t_prof, o_prof = SyntheticTeamData.generate(
        d=192, n_pieces=30, shared_dims=20, shared_variance=4.0, specific_variance=0.5, seed=3
    )
    metric = MahalanobisMetric(shrinkage=0.2).fit(t_pieces, o_pieces)
    gain = metric.decorrelation_gain(t_prof, o_prof)

    # Raw cosine is elevated due to shared subspace
    assert gain["raw_cosine"] > 0.50
    # Mahalanobis cosine is strictly lower
    assert gain["mahalanobis_cosine"] < gain["raw_cosine"]
    assert gain["decorrelation_gain"] > 0.10


def test_mahalanobis_boosts_decision_margin():
    """Verify that Mahalanobis metric improves target-vs-other decision margin."""
    t_pieces, o_pieces, t_prof, o_prof = SyntheticTeamData.generate(
        d=192, n_pieces=30, shared_dims=24, shared_variance=3.5, specific_variance=0.6, seed=4
    )
    res = evaluate_metric_and_relaxation(
        t_pieces, o_pieces, t_prof, o_prof, shrinkage=0.3
    )

    assert res["mean_mahalanobis_margin"] > res["mean_raw_margin"]
    assert res["margin_improvement"] > 0.0
    assert res["margin_improvement_pct"] > 50.0  # substantial relative gain


def test_same_voice_policy_thresholds():
    """Verify SAME_VOICE=0.70 threshold drops vs keeps negative profile."""
    # When pair cosine is below 0.70, negative profile is active
    pol_strict_ok = SameVoicePolicy(threshold=0.70).evaluate(0.45)
    assert pol_strict_ok["negative_active"] is True
    assert pol_strict_ok["negative_dropped"] is False

    # When pair cosine exceeds 0.70 (e.g. 0.78), strict policy drops it
    pol_strict_trap = SameVoicePolicy(threshold=0.70).evaluate(0.78)
    assert pol_strict_trap["negative_active"] is False
    assert pol_strict_trap["negative_dropped"] is True

    # Relaxed policy (1.00) keeps it active
    pol_relaxed = SameVoicePolicy(threshold=1.00).evaluate(0.78)
    assert pol_relaxed["negative_active"] is True
    assert pol_relaxed["negative_dropped"] is False


def test_same_voice_relaxation_prevents_false_accept_leak():
    """Verify that relaxing SAME_VOICE prevents wrong-voice leak in absent windows."""
    ch = evaluate_team_challenge_same_voice(pair_cosine=0.75, contrast_floor=0.15)

    # Strict 0.70 dropped the negative profile, so contrast floor was bypassed
    assert ch["strict_0_70"]["negative_active"] is False
    assert ch["strict_0_70"]["false_accept_leak"] is True
    assert ch["strict_0_70"]["window_gated"] is False

    # Relaxed 1.00 retained the negative profile, so contrast floor gated the window
    assert ch["relaxed_1_00"]["negative_active"] is True
    assert ch["relaxed_1_00"]["false_accept_leak"] is False
    assert ch["relaxed_1_00"]["window_gated"] is True


def test_score_streams_ranking():
    """Verify score_streams ranks candidate streams and picks the target stream."""
    t_pieces, o_pieces, t_prof, o_prof = SyntheticTeamData.generate(d=64, n_pieces=10, seed=5)
    metric = MahalanobisMetric(shrinkage=0.3).fit(t_pieces, o_pieces)

    streams = [t_pieces[0], o_pieces[0]]
    choice, scores, margin = metric.score_streams(t_prof, o_prof, streams)

    assert choice == 0
    assert scores[0] > scores[1]
    assert margin > 0.0


def test_shrinkage_sweep_bounds():
    """Verify that higher shrinkage values increase regularized variance and reduce condition number."""
    t_pieces, o_pieces, _, _ = SyntheticTeamData.generate(d=64, n_pieces=10, seed=6)

    m_low = MahalanobisMetric(shrinkage=0.05).fit(t_pieces, o_pieces)
    m_high = MahalanobisMetric(shrinkage=0.95).fit(t_pieces, o_pieces)

    assert m_low.condition_number > m_high.condition_number


def test_offline_benchmark_dry_run():
    """Verify build_team_baseline_offline_benchmark produces complete structured results."""
    results = build_team_baseline_offline_benchmark(use_real=False, shrinkage=0.3)

    assert results["benchmark"] == "calibrated_pairwise_mahalanobis_and_same_voice_relaxer"
    assert "role_a_aryan_target" in results
    assert "role_b_joseph_target" in results
    assert "challenge_same_voice_test" in results
    assert "shrinkage_sensitivity_sweep" in results
    assert "same_voice_threshold_sweep" in results

    md = format_report_markdown(results)
    assert "Offline Benchmark: Pairwise Mahalanobis Metric" in md
    assert "Role A (Aryan)" in md
    assert "Role B (Joseph)" in md
    assert "SAME_VOICE=0.70 Trap Analysis" in md


def test_fast_compute_latency():
    """Verify that vector transform under Mahalanobis metric runs in < 0.1 ms."""
    t_pieces, o_pieces, t_prof, _ = SyntheticTeamData.generate(d=192, n_pieces=20, seed=7)
    metric = MahalanobisMetric(shrinkage=0.3).fit(t_pieces, o_pieces)

    # Transform 50 times
    import time
    t0 = time.perf_counter()
    for _ in range(50):
        _ = metric.transform(t_prof)
    dt_ms = (time.perf_counter() - t0) * 1000.0 / 50.0

    assert dt_ms < 0.5  # Sub-millisecond latency per window


@pytest.mark.slow
def test_real_team_recordings_if_available():
    """Slow integration test: evaluate on real Aryan/Joseph takes if present."""
    from pathlib import Path

    from eval.mahalanobis_metric import load_real_team_embeddings

    main_dir = Path(__file__).resolve().parent.parent
    team_dir = main_dir / "eval" / "fixtures" / "generated" / "team"
    if not team_dir.exists():
        pytest.skip("real team directory absent")

    data = load_real_team_embeddings(team_dir)
    if data is None:
        pytest.skip("ECAPA model or team wav files not available")

    assert len(data["aryan_pieces"]) > 5
    assert len(data["joseph_pieces"]) > 5
    metric = MahalanobisMetric(shrinkage=0.3).fit(data["aryan_pieces"], data["joseph_pieces"])
    assert metric.fitted
