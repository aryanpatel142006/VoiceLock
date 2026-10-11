"""T-311: Streaming causality regression tests.

A streaming audio separator or processing stage must be strictly causal:
Output at time t must not depend on input after t (beyond documented algorithmic lookahead).
Specifically, if an impulse or disturbance is introduced at sample t_0,
the output for all samples t < t_0 must remain unchanged.
"""

from __future__ import annotations

import numpy as np

from hearing.tse.base import SR
from hearing.tse.mock import IdentityCandidate
from hearing.tse.windowed import SlidingWindowExtractor


class ReversingCandidate:
    """Deliberately NON-causal: output[t] depends on input[end - 1 - t].

    Used to prove the causality test methodology catches violations.
    """

    name = "reversing-test"
    licence = "n/a (test only)"
    needs = frozenset()
    device = "cpu"

    def load(self) -> None:
        pass

    def extract(
        self,
        mix: np.ndarray,
        sr: int,
        *,
        lips: np.ndarray | None = None,
        lips_fps: float = 25.0,
        p_speaking: np.ndarray | None = None,
        enroll: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.asarray(mix, dtype=np.float32)[::-1].copy()


class LookaheadCandidate:
    """Deliberately NON-causal: peeks ahead by lookahead_samples."""

    name = "lookahead-test"
    licence = "n/a (test only)"
    needs = frozenset()
    device = "cpu"

    def __init__(self, lookahead_samples: int = 160) -> None:
        self.lookahead = lookahead_samples

    def load(self) -> None:
        pass

    def extract(
        self,
        mix: np.ndarray,
        sr: int,
        *,
        lips: np.ndarray | None = None,
        lips_fps: float = 25.0,
        p_speaking: np.ndarray | None = None,
        enroll: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(mix, dtype=np.float32)
        # Shift future samples backward into the present
        out = np.zeros_like(x)
        out[: -self.lookahead] = x[self.lookahead :]
        return out


class CausalAverageCandidate:
    """Causal dummy: output[t] = mean(input[:t+1]). Reads strictly past and present samples."""

    name = "causal-avg-test"
    licence = "n/a (test only)"
    needs = frozenset()
    device = "cpu"

    def load(self) -> None:
        pass

    def extract(
        self,
        mix: np.ndarray,
        sr: int,
        *,
        lips: np.ndarray | None = None,
        lips_fps: float = 25.0,
        p_speaking: np.ndarray | None = None,
        enroll: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(mix, dtype=np.float64)
        return (np.cumsum(x) / np.arange(1, len(x) + 1)).astype(np.float32)


class CausalEMACandidate:
    """Causal exponential moving average: y[t] = alpha*y[t-1] + (1-alpha)*x[t]."""

    name = "causal-ema-test"
    licence = "n/a (test only)"
    needs = frozenset()
    device = "cpu"

    def __init__(self, alpha: float = 0.9) -> None:
        self.alpha = alpha

    def load(self) -> None:
        pass

    def extract(
        self,
        mix: np.ndarray,
        sr: int,
        *,
        lips: np.ndarray | None = None,
        lips_fps: float = 25.0,
        p_speaking: np.ndarray | None = None,
        enroll: np.ndarray | None = None,
    ) -> np.ndarray:
        x = np.asarray(mix, dtype=np.float64)
        y = np.zeros_like(x)
        acc = 0.0
        for t in range(len(x)):
            acc = self.alpha * acc + (1.0 - self.alpha) * x[t]
            y[t] = acc
        return y.astype(np.float32)


def _impulse_at(mix: np.ndarray, pos: int = -1, amp: float = 1.0) -> np.ndarray:
    out = mix.copy()
    out[pos] += amp
    return out


def test_causal_identity_candidate_passes():
    """Identity candidate (causal): impulse at the last sample must not change any earlier output."""
    rng = np.random.default_rng(11)
    w = 8000
    mix = rng.standard_normal(w).astype(np.float32)
    cand = IdentityCandidate()

    y1 = cand.extract(mix, SR)
    y2 = cand.extract(_impulse_at(mix, pos=w - 1), SR)

    assert np.array_equal(y1[: w - 1], y2[: w - 1]), (
        "Causal candidate: output before the impulse must be strictly unchanged"
    )
    assert not np.array_equal(y1, y2), "Sanity: the impulse must alter output at/after pos"


def test_causal_moving_average_candidate_passes():
    """Causal moving average: output before impulse position must be invariant."""
    rng = np.random.default_rng(12)
    w = 8000
    mix = rng.standard_normal(w).astype(np.float32)
    cand = CausalAverageCandidate()

    y1 = cand.extract(mix, SR)
    y2 = cand.extract(_impulse_at(mix, pos=w - 1), SR)

    assert np.array_equal(y1[: w - 1], y2[: w - 1]), (
        "Causal moving average: output before the impulse must be strictly unchanged"
    )
    assert not np.array_equal(y1, y2), "Sanity: the impulse must alter output"


def test_causal_ema_candidate_passes():
    """Causal EMA: impulse anywhere in the signal leaves preceding output unchanged."""
    rng = np.random.default_rng(15)
    w = 4000
    mix = rng.standard_normal(w).astype(np.float32)
    cand = CausalEMACandidate()

    impulse_pos = w // 2
    y1 = cand.extract(mix, SR)
    y2 = cand.extract(_impulse_at(mix, pos=impulse_pos), SR)

    assert np.array_equal(y1[:impulse_pos], y2[:impulse_pos]), (
        f"Causal EMA: output before pos={impulse_pos} must be strictly unchanged"
    )
    assert not np.array_equal(y1[impulse_pos:], y2[impulse_pos:]), (
        "Sanity: output at and after pos must be changed"
    )


def test_noncausal_candidate_is_caught():
    """A reversing (non-causal) candidate MUST change early output when the last sample changes."""
    rng = np.random.default_rng(13)
    w = 8000
    mix = rng.standard_normal(w).astype(np.float32)
    cand = ReversingCandidate()

    y1 = cand.extract(mix, SR)
    y2 = cand.extract(_impulse_at(mix, pos=w - 1), SR)

    assert not np.array_equal(y1[: w - 1], y2[: w - 1]), (
        "Test methodology check: a non-causal candidate must be caught "
        "(its early output changed when only the last input sample changed)"
    )


def test_noncausal_lookahead_candidate_is_caught():
    """A lookahead candidate that peeks 160 samples ahead MUST fail causality before impulse_pos."""
    rng = np.random.default_rng(19)
    w = 4000
    mix = rng.standard_normal(w).astype(np.float32)
    cand = LookaheadCandidate(lookahead_samples=160)

    impulse_pos = 2000
    y1 = cand.extract(mix, SR)
    y2 = cand.extract(_impulse_at(mix, pos=impulse_pos), SR)

    # In a lookahead filter, sample at impulse_pos affects output at impulse_pos - 160
    assert not np.array_equal(y1[:impulse_pos], y2[:impulse_pos]), (
        "Lookahead candidate must be caught: output before impulse_pos changed"
    )


def test_windowed_wrapper_causality():
    """SlidingWindowExtractor: output[:K - window] must be invariant to changes
    in input[K:]. The wrapper's algorithmic latency is one window, so anything
    earlier cannot have seen the changed input.
    """
    rng = np.random.default_rng(14)
    sr = SR
    window_s, hop_s = 0.5, 0.25
    window = int(window_s * sr)
    n = int(3.0 * sr)
    mix = rng.standard_normal(n).astype(np.float32)

    ext = SlidingWindowExtractor(IdentityCandidate(), window_s=window_s, hop_s=hop_s, sr=sr)
    y1 = ext.extract(mix, sr)

    k = n - window  # change only the last window's worth of input
    mix2 = mix.copy()
    mix2[k:] = 0.0
    ext2 = SlidingWindowExtractor(IdentityCandidate(), window_s=window_s, hop_s=hop_s, sr=sr)
    y2 = ext2.extract(mix2, sr)

    safe = k - window
    assert safe > 0
    assert np.array_equal(y1[:safe], y2[:safe]), (
        f"Wrapper causality violated: output[:{safe}] changed when only input[{k}:] was modified"
    )
    assert not np.array_equal(y1, y2), "Sanity: zeroing the tail must change something"


def test_windowed_wrapper_causality_various_configurations():
    """SlidingWindowExtractor causality holds across diverse window/hop configurations."""
    rng = np.random.default_rng(21)
    sr = SR
    configs = [
        (0.2, 0.1),
        (0.4, 0.2),
        (0.5, 0.25),
    ]

    for window_s, hop_s in configs:
        window = int(window_s * sr)
        n = int(2.5 * sr)
        mix = rng.standard_normal(n).astype(np.float32)

        ext = SlidingWindowExtractor(IdentityCandidate(), window_s=window_s, hop_s=hop_s, sr=sr)
        y1 = ext.extract(mix, sr)

        k = n - window
        mix2 = mix.copy()
        mix2[k:] += 2.0
        ext2 = SlidingWindowExtractor(IdentityCandidate(), window_s=window_s, hop_s=hop_s, sr=sr)
        y2 = ext2.extract(mix2, sr)

        safe = k - window
        assert safe > 0
        assert np.array_equal(y1[:safe], y2[:safe]), (
            f"Config ({window_s}s, {hop_s}s): output[:{safe}] changed when only input[{k}:] modified"
        )
