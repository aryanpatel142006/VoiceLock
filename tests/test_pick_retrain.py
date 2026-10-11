"""T-375: pure helpers of eval/pick_retrain.py (no weights, no audio files)."""

import numpy as np

from eval import pick_retrain as pr


def test_voiced_s_counts_loud_frames():
    y = np.zeros(16000, np.float32)
    y[:8000] = 0.1
    assert abs(pr.voiced_s(y) - 0.5) < 0.03
    assert pr.voiced_s(np.zeros(16000, np.float32)) == 0.0 or pr.voiced_s(np.zeros(16000, np.float32)) <= 1.0


def test_quality_columns_shape_and_cohort_zscore():
    rng = np.random.default_rng(0)
    cohort = rng.standard_normal((30, 8))
    cohort /= np.linalg.norm(cohort, axis=1, keepdims=True)
    embs = cohort[:2]
    q = pr.quality_columns(embs, [0.5, 0.1], [0.2, 0.3], cohort, [1.0, 0.4], [-1.0, -9.0])
    assert q.shape == (2, len(pr.QUALITY))
    assert q[0, 3] > q[1, 3]  # higher target cosine -> higher z-score


def test_boot_interval_brackets_mean():
    num = np.array([1, 0] * 50, bool)
    v, lo, hi, k, n = pr.boot(num, np.ones(100, bool), np.random.default_rng(1), b=200)
    assert (k, n) == (50, 100) and lo <= v <= hi


def test_augmented_variants_count_and_length():
    p = [np.random.default_rng(2).standard_normal(16000).astype(np.float32) * 0.1]
    out = pr.augmented_variants(p, np.random.default_rng(3), n_each=2)
    assert len(out) == 4 and all(len(o) == 16000 for o in out)
