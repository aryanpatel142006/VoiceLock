"""Offline reference candidates so the bake-off always has rows: a lower and an upper bound."""

from __future__ import annotations

import numpy as np

from hearing.tse.base import conform_output, register_candidate


@register_candidate("identity")
class IdentityCandidate:
    """Returns the mixture unchanged: the "do nothing" baseline (SI-SDRi = 0 dB)."""

    name = "identity"
    licence = "n/a (mock)"
    needs: frozenset[str] = frozenset()
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
        return np.asarray(mix, dtype=np.float32).copy()


@register_candidate("oracle")
class OracleCandidate:
    """Returns the clean target passed via ``enroll``: the upper bound.

    ``oracle = True`` tells the harness (and :class:`~hearing.tse.windowed.SlidingWindowExtractor`)
    to pass the clean target sample-aligned with ``mix`` as ``enroll`` instead of an enrollment clip.
    """

    name = "oracle"
    licence = "n/a (mock)"
    needs: frozenset[str] = frozenset({"enroll"})
    oracle = True
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
        if enroll is None:
            raise ValueError("oracle needs the clean target as enroll")
        n = len(mix)
        y = np.asarray(enroll, dtype=np.float32)[:n]
        return conform_output(np.pad(y, (0, n - len(y))), n, self.name)
