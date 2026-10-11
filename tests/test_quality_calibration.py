"""T-308: quality-aware calibration (eval/calibration/T-308/quality_cal.py) vs the fixed 0.7 cut."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[1] / "eval" / "calibration" / "T-308" / "quality_cal.py"
_spec = importlib.util.spec_from_file_location("quality_cal", _PATH)
qc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("quality_cal", qc)  # dataclasses resolve their module by name
_spec.loader.exec_module(qc)


@pytest.fixture(scope="module")
def cal():
    return qc.fit(*qc.synthetic_pairs(20_000, seed=0))


def test_short_windows_adaptive_beats_fixed_at_95_precision(cal):
    s, d, n, y = qc.synthetic_pairs(8_000, seed=7, durations=qc.SHORT_S)
    fixed = qc.precision_recall(s >= qc.FIXED, y)
    adaptive = qc.precision_recall(s >= cal.predict_threshold(d, n), y)
    assert adaptive["precision"] >= 0.93
    assert fixed["recall"] < 0.05  # 0.7 throws away almost every true 0.5 s window
    assert adaptive["recall"] > fixed["recall"] + 0.3


def test_pooled_single_threshold_loses_short_windows(cal):
    """One raw-cosine cut tuned for 95 % precision on mixed windows vs the per-window threshold."""
    short = qc.synthetic_pairs(5_000, seed=11, durations=qc.SHORT_S)
    long = qc.synthetic_pairs(5_000, seed=12, durations=qc.LONG_S)
    s, d, n, y = (np.concatenate(pair) for pair in zip(short, long, strict=True))
    single = qc.oracle_threshold(s, y)
    adaptive = cal.predict_threshold(d, n)
    m = d == qc.SHORT_S
    assert qc.precision_recall(s[m] >= adaptive[m], y[m])["recall"] > \
        qc.precision_recall(s[m] >= single, y[m])["recall"]
    assert qc.precision_recall(s >= adaptive, y)["precision"] >= 0.93


def test_threshold_shape_and_monotonic_in_duration(cal):
    t = cal.predict_threshold(qc.SHORT_S, 8.0)
    assert isinstance(t, float)
    grid = cal.predict_threshold(np.array([0.5, 1.0, 2.0]), np.array([8.0, 8.0, 8.0]))
    assert grid.shape == (3,) and np.all(np.diff(grid) > 0)
    assert cal.predict_threshold(1.0, 8.0, per_pair=True) >= cal.predict_threshold(1.0, 8.0)


def test_fit_recovers_known_logistic_and_probabilities_are_calibrated():
    rng = np.random.default_rng(3)
    s = rng.uniform(-0.2, 1.0, 30_000)
    d = np.exp(rng.uniform(np.log(0.3), np.log(4.0), s.size))
    n = rng.uniform(2, 20, s.size)
    true = np.array([-3.0, 6.0, 0.5, 0.0, 1.0, 0.0])
    p = 1 / (1 + np.exp(-(qc.features(s, d, n) @ true)))
    y = (rng.random(s.size) < p).astype(int)
    cal = qc.fit(s, d, n, y)
    assert np.allclose(cal.weights, true, atol=0.5)
    pred = cal.predict_proba(s, d, n)
    for lo in (0.2, 0.5, 0.8):
        m = (pred >= lo) & (pred < lo + 0.1)
        assert abs(y[m].mean() - pred[m].mean()) < 0.05


def test_fit_rejects_single_class():
    with pytest.raises(ValueError):
        qc.fit([0.1, 0.2], [1.0, 1.0], [5.0, 5.0], [1, 1])
