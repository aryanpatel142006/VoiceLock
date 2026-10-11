"""T-305: helpers of eval/evidence/T-305/reproduce_margins.py (no weights, no audio files)."""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "eval/evidence/T-305/reproduce_margins.py"
_spec = importlib.util.spec_from_file_location("t305_reproduce_margins", _PATH)
rm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rm)


def _unit_rows(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_margin_excludes_self_pairs_and_matches_pilot_convention():
    a = _unit_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
    j = _unit_rows(np.array([[-1.0, 0.0], [0.0, -1.0]]))
    m = rm.margins(a, j)
    assert m["intra"] == 0.0  # the two a rows are orthogonal; self-pairs ignored
    assert m["margin_pilot"] - m["margin"] == 0.5  # diagonal of 1s adds (1 - 0) / 2 per speaker


def test_bootstrap_duplicates_do_not_count_as_intra_pairs():
    a = _unit_rows(np.array([[1.0, 0.0], [0.0, 1.0]]))
    j = _unit_rows(np.array([[-1.0, 0.0], [0.0, -1.0]]))
    dup = np.array([0, 0])
    m = rm.margins(a, j, dup, np.array([0, 1]))
    assert np.isnan(m["intra"])  # only pairs of one source window left


def test_wilson_interval_bounds():
    lo, hi = rm.wilson(97, 100)
    assert 0.9 < lo < 0.97 < hi <= 1.0
    assert rm.wilson(10, 10)[1] == 1.0


def test_silent_windows_skip_speech():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(16000 * 3) * 1e-3).astype(np.float32)
    x[16000:32000] += 0.3 * np.sin(np.arange(16000) / 5.0).astype(np.float32)
    quiet = rm.silent_windows(x, 0.5)
    assert len(quiet) >= 3
    assert all(float(np.abs(w).max()) < 0.05 for w in quiet)
    voiced, starts = rm.voiced_windows(x, 0.5, 0.5)
    assert starts == [1.0, 1.5]
    assert len(voiced) == 2
