"""Target gate: duck the room by the slider whenever the target is not speaking.

Tier 3 of the fallback ladder (camera, no TSE). Each chunk reads the target track's
``p_speaking`` from ``ctx`` and picks a *goal* gain: 1.0 while the target speaks,
``duck_gain(ctx.suppression)`` (``hearing.fusion.mixer``: the T-093 slider curve, -3 dB
per 10 slider points, floored at -30 dB) otherwise. A per-sample envelope moves
the applied gain toward the goal with a fixed linear slope: ``attack_ms`` to swing
from fully ducked to open and ``release_ms`` to swing from open to fully ducked, so
the gain never jumps and never clicks. ``hold_ms`` keeps the gate open after the
target stops speaking so short pauses between words do not pump.

The speaking decision has hysteresis: it turns on at ``p >= on_threshold`` and off
only below ``off_threshold``. Without a target (``ctx.target.track_id`` is None or
not among ``ctx.tracks``) the stage bypasses: it releases the envelope toward 1.0
with the attack slope (no click when the target is lost) and clears the hold.

The envelope and hold are carried across chunks, so 10 ms streaming equals a
whole-file run for a fixed context. ``latency_ms`` is 0: nothing is buffered.

T-096 (gate-only Focus HQ on the Mac):

* **Duck floor**: the duck gain is :func:`hearing.fusion.mixer.duck_gain` (``max(room_gain(s), -30 dB)``),
  so suppression 100 never goes to digital silence.
* **Both talking**: while the target speaks and another face is also over ``on_threshold``,
  the goal is a partial gain from the ASD margin ``m = p_target - max(p_other)``:
  ``overlap_min_db`` at ``m <= -overlap_span``, 0 dB at ``m >= +overlap_span``, linear in dB
  between (never below the duck gain). The target stays audible, the other talker is pushed down.
* **Lip-confidence partial duck**: while the target is "not speaking" but its ``p`` is in
  ``[partial_lo, off_threshold)`` (ASD unsure), the goal is between the duck gain (at
  ``partial_lo``) and ``partial_max_db`` (at ``off_threshold``), linear in dB, so a lukewarm
  ASD only half-ducks and a mis-scored word stays audible.
* **Stale-ASD guard**: when the newest ``FaceTrack.last_seen_ts`` is more than ``stale_ms``
  older than the chunk's ``capture_ts`` (vision starved), the gate holds its current gain and
  makes no new decision. Tracks with ``last_seen_ts <= 0`` (no clock: fixtures) and ages
  outside ``[0, 10 s]`` (clocks that do not match) are treated as fresh.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.fusion.mixer import duck_gain
from hearing.types import AudioChunk, StageContext

# Tuned on the two-face demo clip (T-096, replayed telemetry through dfn_ll>gate): attack 30 /
# release 300 / hold 400 ms and on/off 0.3/0.2 cut true dropouts 5.5 -> 0.9 s per 30 s with leak
# unchanged within 0.5 dB; the old 20/250/150 ms and 0.5/0.35 ducked on every ASD dip.
DEFAULT_ATTACK_MS = 30.0
DEFAULT_RELEASE_MS = 300.0
DEFAULT_HOLD_MS = 400.0
DEFAULT_ON_THRESHOLD = 0.3
DEFAULT_OFF_THRESHOLD = 0.2
DEFAULT_OVERLAP_MIN_DB = -12.0  # both talking, target clearly the quieter mouth
DEFAULT_OVERLAP_SPAN = 0.3  # |margin| at which the overlap gain saturates
DEFAULT_PARTIAL_LO = 0.05  # p below this ducks fully
DEFAULT_PARTIAL_MAX_DB = -6.0  # gain at p just under off_threshold
DEFAULT_STALE_MS = 200.0
MAX_CLOCK_AGE_S = 10.0  # larger ages mean the audio and video clocks are unrelated


def _db(x: float) -> float:
    return 20.0 * float(np.log10(max(x, 1e-9)))


def _lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _slope_per_sample(ms: float, sr: int) -> float:
    """Gain change per sample for a full 0→1 swing over ``ms``; instant when ms <= 0."""
    n = ms * sr / 1000.0
    return 1.0 if n <= 1.0 else 1.0 / n


@registry.register("gate")
class Gate:
    """Duck non-target time by the slider with an attack/hold/release envelope."""

    sample_rate = PIPELINE_SR
    latency_ms = 0.0

    def __init__(
        self,
        attack_ms: float = DEFAULT_ATTACK_MS,
        release_ms: float = DEFAULT_RELEASE_MS,
        hold_ms: float = DEFAULT_HOLD_MS,
        on_threshold: float = DEFAULT_ON_THRESHOLD,
        off_threshold: float = DEFAULT_OFF_THRESHOLD,
        overlap_min_db: float = DEFAULT_OVERLAP_MIN_DB,
        overlap_span: float = DEFAULT_OVERLAP_SPAN,
        partial_lo: float = DEFAULT_PARTIAL_LO,
        partial_max_db: float = DEFAULT_PARTIAL_MAX_DB,
        stale_ms: float = DEFAULT_STALE_MS,
        max_duck_db: float | None = None,
    ):
        for label, value in (
            ("attack_ms", attack_ms),
            ("release_ms", release_ms),
            ("hold_ms", hold_ms),
        ):
            if value < 0:
                raise ValueError(f"gate {label} must be non-negative, got {value}")
        if not 0.0 <= off_threshold <= on_threshold <= 1.0:
            raise ValueError(
                "gate thresholds must satisfy 0 <= off_threshold <= on_threshold <= 1, "
                f"got off={off_threshold} on={on_threshold}"
            )
        if overlap_min_db > 0 or partial_max_db > 0 or overlap_span <= 0:
            raise ValueError("gate overlap_min_db/partial_max_db must be <= 0 dB and overlap_span > 0")
        if not 0.0 <= partial_lo <= off_threshold:
            raise ValueError(f"gate partial_lo must be in [0, off_threshold], got {partial_lo}")
        if stale_ms < 0:
            raise ValueError(f"gate stale_ms must be non-negative, got {stale_ms}")
        self.overlap_min_db = float(overlap_min_db)
        self.overlap_span = float(overlap_span)
        self.partial_lo = float(partial_lo)
        self.partial_max_db = float(partial_max_db)
        self.stale_ms = float(stale_ms)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.hold_ms = float(hold_ms)
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        if max_duck_db is not None and max_duck_db < 0:
            raise ValueError(f"max_duck_db must be >= 0, got {max_duck_db}")
        # Deepest duck allowed (None = down to room_gain(slider)). A bounded duck turns an ASD miss into
        # a quieter word instead of a hole (T-095, the Focus HQ ladder's gate: 20 dB).
        self.max_duck_db = None if max_duck_db is None else float(max_duck_db)
        self._floor = 0.0 if self.max_duck_db is None else 10.0 ** (-self.max_duck_db / 20.0)
        self.name = (
            f"gate(attack_ms={self.attack_ms:g},release_ms={self.release_ms:g},"
            f"hold_ms={self.hold_ms:g}"
            + ("" if self.max_duck_db is None else f",max_duck_db={self.max_duck_db:g}") + ")"
        )
        self.attack_step = _slope_per_sample(self.attack_ms, self.sample_rate)
        self.release_step = _slope_per_sample(self.release_ms, self.sample_rate)
        self._hold_samples = round(self.hold_ms * self.sample_rate / 1000.0)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self._gain = 1.0  # Applied gain after the last sample.
        self._speaking = False  # Hysteresis state of the speaking decision.
        self._hold_left = 0  # Samples of hold remaining after speech stopped.
        self._open_goal = 1.0  # Goal while the target last spoke (partial when both talked).

    @property
    def gain(self) -> float:
        """Current envelope value (the gain applied to the last sample)."""
        return self._gain

    # --------------------------------------------------------------- decision
    def _target_p_speaking(self, ctx: StageContext) -> float | None:
        track_id = ctx.target.track_id
        if track_id is None:
            return None
        for track in ctx.tracks:
            if track.track_id == track_id:
                return float(track.p_speaking)
        return None

    def _other_p_speaking(self, ctx: StageContext) -> float:
        """Highest ``p_speaking`` among the non-target faces (0 when there are none)."""
        tid = ctx.target.track_id
        return max((float(t.p_speaking) for t in ctx.tracks if t.track_id != tid), default=0.0)

    def _stale(self, ctx: StageContext, now: float) -> bool:
        """True when the newest face observation is older than ``stale_ms`` (vision starved)."""
        seen = max((float(t.last_seen_ts) for t in ctx.tracks), default=0.0)
        if seen <= 0.0:
            return False
        age = now - seen
        return self.stale_ms / 1000.0 < age <= MAX_CLOCK_AGE_S

    def open_gain(self, p: float, p_other: float, duck: float) -> float:
        """Goal gain while the target speaks: 1.0, or the margin-driven partial gain when
        another face speaks too (never below ``duck``)."""
        if p_other < self.on_threshold:
            return 1.0
        frac = min(max((p - p_other + self.overlap_span) / (2.0 * self.overlap_span), 0.0), 1.0)
        return max(_lin(self.overlap_min_db * (1.0 - frac)), duck)

    def closed_gain(self, p: float, duck: float) -> float:
        """Goal gain while the target is silent: ``duck``, lifted toward ``partial_max_db``
        when the ASD is unsure (``partial_lo <= p < off_threshold``)."""
        top = _lin(self.partial_max_db)
        if p < self.partial_lo or top <= duck or self.off_threshold <= self.partial_lo:
            return duck
        frac = min((p - self.partial_lo) / (self.off_threshold - self.partial_lo), 1.0)
        return _lin(_db(duck) + frac * (self.partial_max_db - _db(duck)))

    def _update_speaking(self, p: float) -> bool:
        if self._speaking:
            self._speaking = p >= self.off_threshold
        else:
            self._speaking = p >= self.on_threshold
        return self._speaking

    # ---------------------------------------------------------------- process
    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        x = np.asarray(chunk.samples, dtype=np.float32)
        n = len(x)
        p = self._target_p_speaking(ctx)

        duck = max(duck_gain(ctx.suppression), self._floor)
        stale = False
        if p is None:
            # Bypass: no target to follow. Open the gate smoothly and forget the hold.
            self._speaking = False
            self._hold_left = 0
            self._open_goal = 1.0
            hold_open, goal, active = n, 1.0, False
        elif self._stale(ctx, float(chunk.capture_ts)):
            # Vision starved: p_speaking is old news. Keep the gain where it is.
            stale = True
            hold_open, goal, active = 0, self._gain, True
        elif self._update_speaking(p):
            self._hold_left = self._hold_samples
            goal = self._open_goal = self.open_gain(p, self._other_p_speaking(ctx), duck)
            hold_open, active = 0, True
        else:
            # Spend the hold first (gate stays at the open goal), then ramp toward the duck.
            hold_open = min(self._hold_left, n)
            self._hold_left -= hold_open
            goal = self.closed_gain(p, duck)
            active = True

        if n == 0:
            out = x
        elif self._gain == 1.0 and goal == 1.0 and (hold_open == 0 or self._open_goal == 1.0):
            out = x  # Fully open and staying open: no multiply needed.
        else:
            env, self._gain = self._envelope(goal, n, hold_open, self._open_goal)
            out = (x * env).astype(np.float32, copy=False)

        ctx.telemetry.extras["gate"] = {"gain": float(self._gain), "goal": float(goal), "active": active,
                                        "stale": stale}
        return replace(chunk, samples=out)

    def _envelope(self, goal: float, n: int, hold_open: int, hold_goal: float = 1.0) -> tuple[np.ndarray, float]:
        """Per-sample gain ramping from the carried state toward the segment goals.

        The first ``hold_open`` samples aim at ``hold_goal`` (hold window still running), the
        rest aim at ``goal``. Each sample moves by at most one attack step (rising)
        or release step (falling), then clamps at the goal, so the slope is bounded.
        """
        env = np.empty(n, dtype=np.float32)
        g = self._gain
        pos = 0
        for seg_goal, length in ((hold_goal, hold_open), (goal, n - hold_open)):
            if length <= 0:
                continue
            if g == seg_goal:
                env[pos : pos + length] = seg_goal
            else:
                step = self.attack_step if seg_goal > g else -self.release_step
                ramp = g + step * np.arange(1, length + 1, dtype=np.float64)
                if step > 0:
                    np.minimum(ramp, seg_goal, out=ramp)
                else:
                    np.maximum(ramp, seg_goal, out=ramp)
                env[pos : pos + length] = ramp
                g = float(ramp[-1])
            pos += length
        return env, g
