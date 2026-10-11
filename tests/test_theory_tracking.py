"""Tests for T-295 streaming speaker-tracking theory.

- Anchor reproduction: sim margins match measured team-audio anchors.
- alpha* predictions: decreases as window shortens; phone < directional.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "experiments", "theory", "T-295"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "docs", "research", "theory"))
import streaming_tracking_the as st
import tracking_sim as ts


def test_anchors_reproduced():
    m2 = ts.measure_margin_no_ema("cosine", "phone", 2.0, n=2000, seed=0)
    m05 = ts.measure_margin_no_ema("cosine", "phone", 0.5, n=2000, seed=0)
    mh05 = ts.measure_margin_no_ema("head", "phone", 0.5, n=2000, seed=0)
    assert abs(m2 - ts.MARGIN_COS_2S) < 0.06, f"2.0s margin {m2}"
    assert abs(m05 - ts.MARGIN_COS_05S) < 0.06, f"0.5s margin {m05}"
    assert abs(mh05 - ts.MARGIN_HEAD_05S) < 0.12, f"head margin {mh05}"


def test_alpha_star_decreases_with_window():
    alphas = [0.2, 0.4, 0.6, 0.8]
    stars = {}
    for w in (0.5, 2.0):
        a_star, _ = ts.find_alpha_star("cosine", "phone", w, alphas=alphas,
                                       seed=0)
        stars[w] = a_star
    assert stars[0.5] <= stars[2.0], f"alpha* not decreasing: {stars}"


def test_phone_needs_more_smoothing_than_directional():
    alphas = [0.2, 0.4, 0.6, 0.8]
    a_phone, _ = ts.find_alpha_star("cosine", "phone", 0.5, alphas=alphas, seed=0)
    a_dir, _ = ts.find_alpha_star("cosine", "directional", 0.5, alphas=alphas, seed=0)
    assert a_phone <= a_dir, f"phone {a_phone} vs directional {a_dir}"


def test_mic_sir_ordering():
    assert ts.mic_target_frac("directional") > ts.mic_target_frac("phone")
    assert 0.5 < ts.mic_target_frac("phone") < 0.8


def test_tracker_runs_and_reports():
    profiles = ts.make_profiles(ts.ANGLE, seed=0)
    tr = ts.StreamingTracker(profiles, metric="cosine", mic="phone",
                             alpha=0.5, seed=0)
    fr, mm = tr.run(0.5, n_hops=400, warmup=50)
    assert 0.0 <= fr <= 1.0
    assert math.isfinite(mm)


def test_conf_weighted_tracker_runs():
    profiles = ts.make_profiles(ts.ANGLE, seed=0)
    tr = ts.StreamingTracker(profiles, metric="cosine", mic="phone",
                             conf_weighted=True, alpha_base=0.5, seed=0)
    fr, mm = tr.run(0.5, n_hops=400, warmup=50)
    assert 0.0 <= fr <= 1.0
    assert math.isfinite(mm)


def test_analytical_alpha_star_predictions():
    # P1: alpha* decreases as window shortens
    a_05 = st.derive_optimal_alpha(0.5, "phone", "cosine")
    a_10 = st.derive_optimal_alpha(1.0, "phone", "cosine")
    a_20 = st.derive_optimal_alpha(2.0, "phone", "cosine")
    assert a_05 < a_10 <= a_20

    # P2: phone alpha* < directional alpha*
    a_dir_05 = st.derive_optimal_alpha(0.5, "directional", "cosine")
    assert a_05 < a_dir_05

    # P4: alpha*_head > alpha*_mahalanobis > alpha*_cosine
    a_mah = st.derive_optimal_alpha(0.5, "phone", "mahalanobis")
    a_head = st.derive_optimal_alpha(0.5, "phone", "head")
    assert a_head > a_mah > a_05


def test_bayesian_sequential_step():
    rng = np.random.default_rng(42)
    profiles = rng.normal(size=(2, 192))
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True)
    z_init = profiles[0].copy()
    x_obs = profiles[0] + rng.normal(scale=0.1, size=192)
    x_obs /= np.linalg.norm(x_obs)

    z_next, speaker, margin, prob = st.bayesian_sequential_step(
        z_init, x_obs, profiles, alpha_t=0.3
    )
    assert z_next.shape == (192,)
    assert speaker == 0
    assert margin > 0.0
    assert 0.5 <= prob <= 1.0
