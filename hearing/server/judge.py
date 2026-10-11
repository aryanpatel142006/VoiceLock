"""The judge profile (T-088): one ``control`` locks the session to the configuration we present with.

``{"type": "control", "profile": "judge"}`` applies :data:`JUDGE_FIELDS` (Focus HQ at suppression
100, strict isolation, sticky attention, ``synth_feed`` auto, ``hq_backend`` auto when a GPU url is
configured), the latency preset for the backend playing (:data:`JUDGE_LATENCY`), and turns on
:class:`AutoCalibrator`: once a target is selected (the wearer looked at them past the dwell, or
tapped them) and ASD sees them speaking alone for
``SOLO_S`` (1 s), it opens the shared enroller's manual calibration window (T-085) by itself, for
``AUTO_CALIBRATE_S`` (2 s, the T-061 enrollment length; the manual window is 3 s) so lock to
Isolated fits in 6 s on the replay clip. The page's C key (``control.enroll``) stays the backup.
"""

from __future__ import annotations

import json
import os

from hearing.latency_preset import apply_adapt_hop
from hearing.registry import _parse_stage, _split
from hearing.tse.enroll import AutoEnroller
from hearing.types import StageContext
from hearing.vision.target import ATTENTION_PRESETS

PROFILES = ("judge",)
JUDGE_FIELDS: dict = {
    "mode": "focus_hq",
    "suppression": 100,
    "isolation": "strict",
    "synth_feed": "auto",
    "hq_backend": "auto",  # skipped when no GPU url is configured (it would only answer an error)
    "attention": {**ATTENTION_PRESETS["sticky"], "preset": "sticky"},
}
# Natural under the judge profile (the camera-lost fallback, or a mode switch without a slider value):
# plain dfn_ll at s=80 (T-103: on demo-mf-easy it beat the gate chain at s=50, DNSMOS 2.92 vs 2.12).
JUDGE_NATURAL_SUPPRESSION = 80
# Focus HQ latency preset per backend playing when the profile is applied. GPU (T-086): balanced, which
# on the A100 ladder (config/tiers-remote.yaml) is enrolled 0.5 s / 0.125 s = 625 ms + the 200 ms buffer.
JUDGE_LATENCY = {"local": "balanced", "remote": "balanced"}

AUTO_CALIBRATE_S = 2.0  # the T-061 enrollment length (auto clips are 1.75 s since T-092); 1 s less than the manual window
# Ladder hysteresis under the judge profile: a tier must stay eligible this long before the ladder
# moves up (config/tiers.yaml: 1.0 s). 0.5 s = the enrolled hop at the balanced preset, so the first
# good enrolled window is already playing when the switch lands (no fallback mix after it).
JUDGE_UPGRADE_AFTER_S = 0.5
# T-135: config/tiers.yaml allows one promotion per 10 s (min_switch_s, T-095's anti-flap). Under the
# judge the audio-only -> gate step lands ~1 s after Start, so the calibrated Enrolled tier (eligible
# ~4.8 s) waited until ~10.9 s (lock -> Isolated 10.8 s on demo-mf-easy). 3 s still holds back a
# re-promotion after a demotion; forced demotions were never held.
JUDGE_MIN_SWITCH_S = 3.0
# T-135: the tse hop under the profile. Each HOP_STEPS change (hearing/tse/stage.py) moves the
# enrolled delay by a hop and plays as an audible dip/replay, but a hop pinned from Start cost the
# lock on a loaded machine (Q-341). So the ladders keep the adaptive hop (JUDGE_ADAPT_HOP) until the
# target is first isolated, then the session pins every tse stage at its current hop
# (JUDGE_PIN_HOP_ON_ISOLATED, Session._pin_hops): no hop change while the judge listens.
JUDGE_ADAPT_HOP = os.environ.get("HEARING_JUDGE_ADAPT_HOP", "1") != "0"  # "0": pinned from Start (A/B, T-135)
JUDGE_PIN_HOP_ON_ISOLATED = True
# T-135: output loudness under the profile: hearing.audio.agc (-20 dBFS, max +12 dB, 50/800 ms, soft
# limiter). The page applies the same AGC to the raw side of its A/B switch (audio-worklet.js), so
# RAW and SELECTIVE play level-matched instead of SELECTIVE following the babble and clipping.
JUDGE_AGC = True
SOLO_S = 1.0  # the locked target speaking alone this long starts a calibration
SOLO_GAP_S = 0.3  # ASD dips shorter than this (pauses between words) keep the solo clock
P_TARGET = 0.5  # target p_speaking at or above: speaking
P_OTHER = 0.5  # every other face below: alone
MAX_ATTEMPTS = 3  # per target track; a failed window (too little speech) re-arms


def judge_spec(pipeline: str) -> str:
    """``pipeline`` with :data:`JUDGE_UPGRADE_AFTER_S` and :data:`JUDGE_MIN_SWITCH_S` on every ``tiers``
    ladder that does not set its own; the tse hop pinned from the start only if :data:`JUDGE_ADAPT_HOP` is False."""
    pipeline = apply_adapt_hop(pipeline, JUDGE_ADAPT_HOP)
    stages = _split(pipeline, ">")
    changed = False
    for i, stage in enumerate(stages):
        name, kwargs = _parse_stage(stage)
        if name != "tiers" or ("upgrade_after_s" in kwargs and "min_switch_s" in kwargs):
            continue
        kwargs.setdefault("upgrade_after_s", JUDGE_UPGRADE_AFTER_S)
        kwargs.setdefault("min_switch_s", JUDGE_MIN_SWITCH_S)
        stages[i] = f"{name}({', '.join(f'{k}={json.dumps(v)}' for k, v in kwargs.items())})"
        changed = True
    return ">".join(stages) if changed else pipeline


class AutoCalibrator:
    """Starts :meth:`AutoEnroller.calibrate` when the selected target talks alone for ``solo_s``.

    Call :meth:`update` once per hop after the enroller saw it. ``status()["state"]``: ``off``,
    ``waiting`` (no target selected yet, or not alone), ``calibrating``, ``done`` (the enroller holds a
    calibrated clip of this target, auto or manual) or ``gave_up`` (``max_attempts`` failed windows).
    """

    def __init__(self, enabled: bool = False, *, solo_s: float = SOLO_S, gap_s: float = SOLO_GAP_S,
                 seconds: float = AUTO_CALIBRATE_S, p_target: float = P_TARGET, p_other: float = P_OTHER,
                 max_attempts: int = MAX_ATTEMPTS):
        self.enabled = enabled
        self.solo_s, self.gap_s, self.seconds = solo_s, gap_s, seconds
        self.p_target, self.p_other, self.max_attempts = p_target, p_other, max_attempts
        self.attempts = 0
        self._track: int | None = None
        self._solo = 0.0
        self._gap = 0.0
        self._state = "waiting"

    def update(self, dt: float, ctx: StageContext, enroller: AutoEnroller) -> bool:
        """True when this hop started a calibration window."""
        if not self.enabled:
            return False
        target = ctx.target
        if target.track_id != self._track:
            self._track, self.attempts, self._solo = target.track_id, 0, 0.0
        if enroller.calibrating:
            self._state, self._solo = "calibrating", 0.0
            return False
        if enroller.calibrated and target.track_id is not None and enroller.track_id == target.track_id:
            self._state = "done"
            return False
        if target.track_id is None:  # selected by attention or tapped; ``locked`` is only the tap
            self._state, self._solo = "waiting", 0.0
            return False
        if self.attempts >= self.max_attempts:
            self._state = "gave_up"
            return False
        self._state = "waiting"
        p = {t.track_id: t.p_speaking for t in ctx.tracks}
        alone = (p.get(target.track_id, 0.0) >= self.p_target
                 and all(v < self.p_other for k, v in p.items() if k != target.track_id))
        if alone:
            self._solo += dt
            self._gap = 0.0
        else:
            self._gap += dt
            if self._gap > self.gap_s:
                self._solo = 0.0
        if self._solo < self.solo_s:
            return False
        enroller.calibrate(self.seconds)
        self.attempts += 1
        self._solo = 0.0
        self._state = "calibrating"
        return True

    def status(self) -> dict:
        return {"state": self._state if self.enabled else "off", "solo_s": round(self._solo, 2),
                "attempts": self.attempts}
