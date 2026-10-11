"""Output AGC + soft limiter (T-135): a slow gain to a fixed loudness, then a soft clip.

Without it the judge page's SELECTIVE side followed the room (isolated speech was as loud as the
babble it came from) and hard-clipped at ``float_to_pcm16``. :class:`Agc` tracks the block level
(10 ms RMS in dBFS) with an envelope that rises in ``attack_ms`` and falls in ``release_ms``, and
plays ``target_db - envelope`` (at most ``max_gain_db`` up, ``max_cut_db`` down). Blocks below
``floor_db`` (silence), more than ``hold_db`` under the envelope (the residual in an isolated
voice's pauses) or within ``NOISE_MARGIN_DB`` of the tracked noise floor (steady room noise, T-169:
without it the envelope drifted down onto -46 dBFS room noise and raised it the full 12 dB) hold the envelope, so pauses are never pumped up. After ``PAUSE_HANG_MS`` of such
blocks (a pause, not a gap between syllables) the applied gain also relaxes to at most 0 dB in
``PAUSE_RELEASE_MS``, and comes back in ``PAUSE_RETURN_MS`` when the speech does: the held speech gain
used to play on through the room noise of every pause (T-169: +4.5 dB median on quiet windows of a live
A100 session). T-179, opt-in (:data:`FLOOR_RULE`, ``HEARING_AGC_FLOOR_RULE=1`` on the server): a block under
the effective floor ``max(floor_db, noise floor + NOISE_MARGIN_DB)`` never raises the applied gain: it holds or
moves back toward 0 dB (a positive gain decays, a cut relaxes up to 0 dB at most), and the gain is capped at +9 dB
(at low confidence the mixer ducks the room only 6 dB, so cafe noise at -45 dBFS used to pass the -50 floor and
come out up to +12 dB, sessions 20261009T203531Z-2 / 211905Z-8). Off by default (Q-485): it meets the pause-lift
bar but not every T-179 bar, so the shipped default stays the T-169 AGC until the orchestrator decides. The gain moves once per block and is ramped linearly over the next block, so feeding any chunk
size gives the same output (no look-ahead, zero latency). :func:`soft_limit` then bends everything
above ``limit`` (-1.9 dBFS) smoothly toward full scale instead of clipping.

``web/console/audio-worklet.js`` (``AgcState``) runs the same level rule with the same numbers on the raw
side of the A/B switch (the floor rule when telemetry ``agc.floor_rule`` is true), so RAW and SELECTIVE play at the same loudness. Registered as ``agc``; the
session applies it to its output under the judge profile (``hearing.server.judge.JUDGE_AGC``).
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import replace

import numpy as np

from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.types import AudioChunk, StageContext

TARGET_DB = -20.0  # dBFS RMS the output is pulled toward
MAX_GAIN_DB = 12.0  # quiet speech is raised at most this much
MAX_CUT_DB = 20.0  # loud input is lowered at most this much
ATTACK_MS = 50.0  # envelope rise (getting louder: gain falls fast)
RELEASE_MS = 800.0  # envelope fall (getting quieter: gain rises slowly)
FLOOR_DB = -50.0  # blocks quieter than this hold the envelope (no pumping in pauses)
# Blocks this far under the envelope (the target's pauses: an isolated voice's residual sits 15-20 dB
# under it) hold it too. Without it the 800 ms release followed the residual and the gain raised it
# (demo_regress focus_hq under the judge: leak -18.0 -> -11.5 dB, SI-SDRi +8.9 -> +0.6 dB, T-135).
HOLD_DB = 12.0
# T-169: the noise floor is the NOISE_PERCENTILE-th percentile of the last NOISE_WINDOW_MS of block levels
# (speech always has gaps in that long; a percentile, not the minimum, so a few clean blocks of TSE silence
# do not drag it under the room); blocks within NOISE_MARGIN_DB of it are room noise and hold the envelope
# like a pause. It sits 3-4 dB under steady room noise and 10 ms blocks of it spread +-4 dB, hence 10 dB.
# Replay of 20261009T203531Z-2: percentile 0/10/20/30 -> quiet windows > in + 3 dB 14/10/7/7 %, speech
# active level -21.4/-22.1/-22.4/-22.6 dBFS median; 10 is the compromise.
NOISE_WINDOW_MS = 1500.0
NOISE_MARGIN_DB = 10.0
NOISE_PERCENTILE = 10.0
# T-169: a pause this long (consecutive held blocks) stops being boosted; shorter gaps keep the gain
PAUSE_HANG_MS = 300.0
PAUSE_RELEASE_MS = 150.0  # the boost fades out over a pause
PAUSE_RETURN_MS = 20.0  # and comes back this fast when speech resumes
PAUSE_ONSET_MS = 30.0  # speech = this many ms of voiced blocks in a row (one loud noise block is not)
BLOCK_MS = 10.0
# T-179 (opt-in, Q-485): never raise the gain under the effective floor, and cap it at +9 dB (the room under
# quiet speech rose with the cap). On with HEARING_AGC_FLOOR_RULE=1 (read when a session starts).
FLOOR_RULE = {"max_gain_db": 9.0, "never_up": True}
FLOOR_RULE_ENV = "HEARING_AGC_FLOOR_RULE"
LIMIT = 0.8  # soft limiter knee (-1.9 dBFS): linear below, tanh into [LIMIT, 1) above
_EPS = 1e-10


def soft_limit(x: np.ndarray, limit: float = LIMIT) -> np.ndarray:
    """Identity below ``limit``; above it a tanh knee that never reaches 1.0 (no hard clipping)."""
    x = np.asarray(x, dtype=np.float32)
    a = np.abs(x)
    over = a > limit
    if not over.any():
        return x
    head = 1.0 - limit
    y = x.copy()
    y[over] = np.sign(x[over]) * (limit + head * np.tanh((a[over] - limit) / head))
    return y.astype(np.float32, copy=False)


def floor_rule_kwargs() -> dict:
    """:data:`FLOOR_RULE` when ``HEARING_AGC_FLOOR_RULE=1``, else ``{}`` (the default T-169 AGC)."""
    return dict(FLOOR_RULE) if os.environ.get(FLOOR_RULE_ENV, "").strip() == "1" else {}


def _coef(ms: float, block_ms: float) -> float:
    """One-pole smoothing coefficient per block for a time constant of ``ms``."""
    return 0.0 if ms <= 0 else math.exp(-block_ms / ms)


class Agc:
    """Streaming AGC + soft limiter on mono float32 at ``sample_rate``. See the module doc."""

    def __init__(self, sample_rate: int = PIPELINE_SR, *, target_db: float = TARGET_DB,
                 max_gain_db: float = MAX_GAIN_DB, max_cut_db: float = MAX_CUT_DB, attack_ms: float = ATTACK_MS,
                 release_ms: float = RELEASE_MS, floor_db: float = FLOOR_DB, limit: float | None = LIMIT,
                 hold_db: float = HOLD_DB, noise_margin_db: float | None = NOISE_MARGIN_DB,
                 pause_hang_ms: float | None = PAUSE_HANG_MS, never_up: bool = False):
        self.hold_db = float(hold_db)
        self.never_up = bool(never_up)  # True: the T-179 floor rule (FLOOR_RULE); False: a held block may raise the gain
        # None turns a T-169 rule off (the T-135 AGC: eval/replay of sessions recorded before it)
        self.noise_margin_db = -math.inf if noise_margin_db is None else float(noise_margin_db)
        self.sample_rate = int(sample_rate)
        self.block = max(1, round(self.sample_rate * BLOCK_MS / 1000.0))
        self.target_db, self.max_gain_db, self.max_cut_db = float(target_db), float(max_gain_db), float(max_cut_db)
        self.floor_db = float(floor_db)
        self.limit = limit
        self._attack = _coef(attack_ms, BLOCK_MS)
        self._release = _coef(release_ms, BLOCK_MS)
        self._pause_release = _coef(PAUSE_RELEASE_MS, BLOCK_MS)
        self._pause_return = _coef(PAUSE_RETURN_MS, BLOCK_MS)
        self._levels: deque[float] = deque(maxlen=round(NOISE_WINDOW_MS / BLOCK_MS))
        self._onset_blocks = round(PAUSE_ONSET_MS / BLOCK_MS)
        self._hang_blocks = math.inf if pause_hang_ms is None else round(pause_hang_ms / BLOCK_MS)
        self.reset()

    def reset(self) -> None:
        self.env_db: float | None = None  # None until the first block above the floor
        self.gain_db = 0.0  # the speech gain (held through pauses)
        self.noise_db: float | None = None  # tracked noise floor (T-169)
        self._levels.clear()
        self.pause_db = 0.0  # <= 0: the cut that keeps a long pause from being boosted (T-169)
        self.applied_db = 0.0  # the gain played this block (gain_db + pause_db, clamped under the floor, T-179)
        self._held = 0  # held blocks since the speech last ran PAUSE_ONSET_MS
        self._voiced = 0  # consecutive voiced blocks
        self._g0 = self._g1 = 1.0  # linear gain at the start / end of the current block
        self._pos = 0  # samples into the current block
        self._acc = 0.0  # sum of squares of the current block

    def _target_gain_db(self) -> float:
        if self.env_db is None:
            return 0.0
        return float(np.clip(self.target_db - self.env_db, -self.max_cut_db, self.max_gain_db))

    def effective_floor_db(self) -> float:
        """Blocks under ``max(floor_db, noise floor + NOISE_MARGIN_DB)`` are room noise or silence (T-179)."""
        if self.noise_db is None:
            return self.floor_db
        return max(self.floor_db, self.noise_db + self.noise_margin_db)

    def _end_block(self) -> None:
        level = 10.0 * math.log10(self._acc / self.block + _EPS)
        self._levels.append(level)
        self.noise_db = float(np.percentile(self._levels, NOISE_PERCENTILE))
        below = level < self.effective_floor_db()
        if not below and (self.env_db is None or level >= self.env_db - self.hold_db):
            self._voiced += 1
            if self._voiced >= self._onset_blocks:
                self._held = 0
            if self.env_db is None:
                self.env_db = level
            else:
                c = self._attack if level > self.env_db else self._release
                self.env_db = c * self.env_db + (1.0 - c) * level
        else:
            self._held, self._voiced = self._held + 1, 0
        self.gain_db = self._target_gain_db()
        if self._held >= self._hang_blocks:
            cut, c = -max(self.gain_db, 0.0), self._pause_release
        else:
            cut, c = 0.0, self._pause_return
        self.pause_db = c * self.pause_db + (1.0 - c) * cut
        if self.pause_db > -0.01:
            self.pause_db = 0.0
        applied = self.gain_db + self.pause_db
        if below and self.never_up:  # under the floor the gain only moves back toward 0 dB, never up
            applied = min(applied, max(self.applied_db, 0.0))
        self.applied_db = applied
        self._g0, self._g1 = self._g1, 10.0 ** (applied / 20.0)
        self._pos, self._acc = 0, 0.0

    def process_array(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        y = np.empty_like(x)
        i = 0
        while i < len(x):
            n = min(len(x) - i, self.block - self._pos)
            seg = x[i:i + n]
            ramp = self._g0 + (self._g1 - self._g0) * (np.arange(self._pos + 1, self._pos + n + 1) / self.block)
            y[i:i + n] = seg * ramp
            self._acc += float(np.dot(seg, seg))
            self._pos += n
            i += n
            if self._pos == self.block:
                self._end_block()
        return soft_limit(y, self.limit) if self.limit is not None else y

    def status(self) -> dict:
        return {"gain_db": round(self.gain_db, 1), "pause_db": round(self.pause_db, 1),
                "applied_db": round(self.applied_db, 1), "floor_rule": self.never_up,
                "level_db": None if self.env_db is None else round(self.env_db, 1), "target_db": self.target_db}


@registry.register("agc")
class AgcStage:
    """``agc(target_db=-20, max_gain_db=12, attack_ms=50, release_ms=800, floor_db=-50, limit=0.8)``."""

    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def __init__(self, target_db: float = TARGET_DB, max_gain_db: float = MAX_GAIN_DB, max_cut_db: float = MAX_CUT_DB,
                 attack_ms: float = ATTACK_MS, release_ms: float = RELEASE_MS, floor_db: float = FLOOR_DB,
                 limit: float | None = LIMIT):
        self.agc = Agc(PIPELINE_SR, target_db=target_db, max_gain_db=max_gain_db, max_cut_db=max_cut_db,
                       attack_ms=attack_ms, release_ms=release_ms, floor_db=floor_db, limit=limit)
        self.name = f"agc({target_db:g} dB)"

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        return replace(chunk, samples=self.agc.process_array(chunk.samples))

    def reset(self) -> None:
        self.agc.reset()


# Live test Oct 11 (session 20261011T092456Z-3): the AGC's pause rule swung the applied gain 0 <-> +12 dB on a
# steady single voice (out.wav level SD 6.9 dB vs 3.0 dB before the AGC), heard as the voice going up and down.
# HEARING_FIXED_GAIN (default on): the output gets ONE constant makeup gain (HEARING_FIXED_GAIN_DB, default +22 dB:
# the iPhone mic's -45 dBFS speech lands near -23 dBFS) and the soft limiter for peaks only. Nothing time-varying.
FIXED_ENV = "HEARING_FIXED_GAIN"  # "0"/"off"/"false" = the T-169 AGC as before
FIXED_DB_ENV = "HEARING_FIXED_GAIN_DB"
FIXED_GAIN_DB = 22.0


def fixed_gain_enabled() -> bool:
    return os.environ.get(FIXED_ENV, "").strip().lower() not in ("0", "off", "false")


def fixed_gain_db() -> float:
    try:
        v = float(os.environ.get(FIXED_DB_ENV, "") or FIXED_GAIN_DB)
    except ValueError:
        return FIXED_GAIN_DB
    return v if math.isfinite(v) and -40.0 <= v <= 40.0 else FIXED_GAIN_DB


class FixedGain:
    """Constant makeup gain + soft limiter (same interface as :class:`Agc`): the level never moves."""

    never_up = False

    def __init__(self, sample_rate: int = PIPELINE_SR, *, gain_db: float | None = None, limit: float | None = LIMIT):
        self.sample_rate = int(sample_rate)
        self.gain_db = fixed_gain_db() if gain_db is None else float(gain_db)
        self._g = np.float32(10.0 ** (self.gain_db / 20.0))
        self.limit = limit

    def reset(self) -> None:
        pass

    def process_array(self, x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float32) * self._g
        return soft_limit(y, self.limit) if self.limit is not None else y

    def status(self) -> dict:
        return {"gain_db": round(self.gain_db, 1), "pause_db": 0.0, "applied_db": round(self.gain_db, 1),
                "floor_rule": False, "level_db": None, "target_db": None, "fixed": True}
