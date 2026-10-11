"""Identity bouncer (T-413, ranked-approaches #1): check the extracted voice is the enrolled one.

Identity is a **gate, never a reward**: nothing here is differentiable or fed back into
training. After any extractor runs, the output window is embedded and compared with the
enrollment (and, when known, with the other enrolled voices). On a mismatch the bouncer
blocks, and offline/windowed callers may retry once on the residual (``mix - output``):
when an extractor locked onto the interferer, the residual is mostly the target.

* ``IdentityGate.score(audio)`` -> ``GateScore`` (one window).
* ``bounce(mix, extract, gate)`` -> ``BounceResult``: pass / retry_pass / blocked (windowed).
* ``IdentityGateStream`` gates a streaming extractor's output chunk by chunk with zero added
  latency: decisions come from the last ``window_s`` of output, re-scored every ``hop_s``, and
  apply to the following chunks through a short gain ramp; blocked chunks play the room at
  ``floor_db`` below the mix (the T-180 gated-window floor) instead of the wrong voice.

Thresholds are cosine numbers for the speechbrain ECAPA embedder (``hearing.tse.enrolled``);
measured by ``eval/identity_gate_eval.py`` on the team recordings (see the module constants).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

SR = 16000
# Picked from eval/identity_gate_eval.py (T-413, ECAPA, 1.5 s windows, 3 team speakers, SIR 0/-5/-15
# dB, 30 items per bin): clean target blocked 3.9 %, wrong-speaker lock passed 3.3 %, all leaky
# outputs false-block 17.3 % / false-pass 7.9 %. 0.30 / 0.05 blocked the clean target 10 %.
# See docs/research/T-413_RANKED_APPROACHES_PLAN.md.
ACCEPT_COS = 0.25  # output-to-enrollment cosine below this: not the enrolled voice
MARGIN_COS = 0.0  # with other profiles: target cosine must beat the best other by this much
SILENT_DBFS = -55.0  # quieter output windows carry no identity and are passed as "silent"
WINDOW_S = 1.5
HOP_S = 0.25
FLOOR_DB = 20.0
RAMP_MS = 20.0


def _unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else np.zeros_like(v)


def _dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, np.float64)
    return float(10.0 * np.log10(np.mean(x * x) + 1e-20)) if x.size else -200.0


@dataclass(frozen=True)
class GateScore:
    target_cos: float
    other_cos: float | None  # best cosine to another enrolled voice, None without others
    passed: bool
    reason: str  # ok | silent | low_similarity | closer_to_other

    @property
    def margin(self) -> float | None:
        return None if self.other_cos is None else self.target_cos - self.other_cos


class IdentityGate:
    """``embedder.embed(audio) -> vector``; ``enrollment`` audio or a precomputed embedding."""

    def __init__(self, embedder, enrollment, others: Sequence = (), accept_cos: float = ACCEPT_COS,
                 margin_cos: float = MARGIN_COS, silent_dbfs: float = SILENT_DBFS):
        self.embedder = embedder
        self.target = self._embedding(enrollment)
        self.others = [self._embedding(o) for o in others]
        self.accept_cos, self.margin_cos, self.silent_dbfs = accept_cos, margin_cos, silent_dbfs

    def _embedding(self, x) -> np.ndarray:
        x = np.asarray(x)
        # 1-D float audio is longer than any embedding (ECAPA 192, mock 8); 0.1 s = 1600 samples.
        if x.ndim == 1 and x.size >= SR // 10:
            return _unit(self.embedder.embed(x.astype(np.float32)))
        return _unit(x)

    def score(self, audio: np.ndarray) -> GateScore:
        if _dbfs(audio) < self.silent_dbfs:
            return GateScore(0.0, None, True, "silent")
        e = _unit(self.embedder.embed(np.asarray(audio, np.float32)))
        t = float(e @ self.target)
        o = max((float(e @ v) for v in self.others), default=None)
        if t < self.accept_cos:
            return GateScore(t, o, False, "low_similarity")
        if o is not None and t - o < self.margin_cos:
            return GateScore(t, o, False, "closer_to_other")
        return GateScore(t, o, True, "ok")


@dataclass
class BounceResult:
    audio: np.ndarray
    action: str  # pass | retry_pass | blocked
    scores: list[GateScore] = field(default_factory=list)


def bounce(mix: np.ndarray, extract: Callable[[np.ndarray], np.ndarray], gate: IdentityGate,
           retries: int = 1, floor_db: float | None = FLOOR_DB) -> BounceResult:
    """Extract, check identity, retry on ``mix - output`` up to ``retries`` times, else block.

    A blocked window returns the mix ``floor_db`` down (``None``: silence), never the
    rejected extraction.
    """
    mix = np.asarray(mix, np.float32)
    source, scores = mix, []
    for attempt in range(retries + 1):
        out = np.asarray(extract(source), np.float32)
        s = gate.score(out)
        scores.append(s)
        if s.passed:
            return BounceResult(out, "pass" if attempt == 0 else "retry_pass", scores)
        source = (source - out).astype(np.float32)  # the wrong voice out; the target is left
    fallback = np.zeros_like(mix) if floor_db is None else mix * np.float32(10.0 ** (-floor_db / 20.0))
    return BounceResult(fallback, "blocked", scores)


class IdentityGateStream:
    """Causal per-chunk bouncer for a streaming extractor (``process(out_chunk, mix_chunk)``).

    ``fail_hops`` consecutive failed scores close the gate, ``pass_hops`` passes reopen it.
    Before the first full window the gate is open (no evidence yet). Adds no latency; a wrong
    voice can play for up to ``window_s * fraction + hop_s`` plus the embed time before it is
    blocked.
    """

    def __init__(self, gate: IdentityGate, window_s: float = WINDOW_S, hop_s: float = HOP_S,
                 floor_db: float = FLOOR_DB, ramp_ms: float = RAMP_MS, fail_hops: int = 1,
                 pass_hops: int = 2, sr: int = SR):
        self.gate, self.sr = gate, sr
        self.window, self.hop = int(window_s * sr), int(hop_s * sr)
        self.floor = float(10.0 ** (-floor_db / 20.0))
        self.step = 1.0 / max(1, int(ramp_ms * sr / 1000))
        self.fail_hops, self.pass_hops = fail_hops, pass_hops
        self.reset()

    def reset(self) -> None:
        self.buf = np.zeros(0, np.float32)
        self.since = 0
        self.open = True
        self.gain = 1.0
        self.fails = self.passes = 0
        self.last: GateScore | None = None
        self.history: list[tuple[int, GateScore, bool]] = []  # (sample, score, open after)
        self.t = 0

    def _decide(self) -> None:
        s = self.gate.score(self.buf[-self.window :])
        self.last = s
        if s.passed:
            self.passes, self.fails = self.passes + 1, 0
            if not self.open and self.passes >= self.pass_hops:
                self.open = True
        else:
            self.fails, self.passes = self.fails + 1, 0
            if self.open and self.fails >= self.fail_hops:
                self.open = False
        self.history.append((self.t, s, self.open))

    def process(self, out_chunk: np.ndarray, mix_chunk: np.ndarray) -> np.ndarray:
        out = np.asarray(out_chunk, np.float32)
        mix = np.asarray(mix_chunk, np.float32)
        self.buf = np.concatenate([self.buf, out])[-self.window :]
        self.since += len(out)
        self.t += len(out)
        if len(self.buf) >= self.window and self.since >= self.hop:
            self.since = 0
            self._decide()
        goal = 1.0 if self.open else 0.0
        g = np.empty(len(out), np.float32)
        for k in range(len(out)):  # linear ramp toward the decision, ~RAMP_MS long
            self.gain += float(np.clip(goal - self.gain, -self.step, self.step))
            g[k] = self.gain
        return g * out + (1.0 - g) * self.floor * mix
