"""Target selection: centre-of-view with dwell, hysteresis, hold and explicit lock.

`TargetSelector.update(tracks, ts)` turns the current face tracks into a
`TargetState` once per video frame. It consumes only `hearing.types`, so it
runs without the tracker, MediaPipe or ASD.
"""

from __future__ import annotations

import math
from collections import deque

from hearing.types import FaceTrack, TargetState

_EPS = 1e-6
_TIE_GAP = 0.2  # mean p_speaking lead needed to override the more central face
_MIN_EVIDENCE_S = 0.5  # a real tie decides on no less speaking evidence than this
_SPEAK_WINDOW_S = 1.5  # tiebreak looks at p_speaking over this long

# Live attention tunables (`control.attention`): (min, max) per field.
ATTENTION_RANGES: dict[str, tuple[float, float]] = {
    "dwell_s": (0.1, 3.0),
    "hold_s": (0.2, 10.0),
    "switch_margin": (0.0, 0.5),
    "tie_margin": (0.0, 0.5),
    "hold_lock_s": (1.0, 60.0),
}
# hold_s defaults to 2.0 s (was 1.0): people look down, glance away or are occluded for
# 1-1.5 s while still being the person they are listening to; 1 s dropped the target
# (and the other talker took over) in exactly those moments.
# hold_lock_s (T-107): an explicit lock outlives a lost face for this long (vision stalls under load,
# the wearer looks down); after it the lock is "lost" but never falls back to free selection.
ATTENTION_DEFAULTS: dict[str, float] = {"dwell_s": 0.5, "hold_s": 2.0, "switch_margin": 0.04,
                                         "tie_margin": 0.15, "hold_lock_s": 10.0}
ATTENTION_PRESETS: dict[str, dict[str, float]] = {
    "glance": {"dwell_s": 0.3, "hold_s": 1.0, "switch_margin": 0.02, "tie_margin": 0.15, "hold_lock_s": 10.0},
    "normal": dict(ATTENTION_DEFAULTS),
    "sticky": {"dwell_s": 1.0, "hold_s": 4.0, "switch_margin": 0.10, "tie_margin": 0.15, "hold_lock_s": 10.0},
}
REATTACH_DIST = 0.15  # normalised box-centre distance: a new track this close to a lost locked face is that face


def check_attention(values: dict[str, float]) -> dict[str, float]:
    """Validate a partial attention update; raises ValueError naming the field."""
    out = {}
    for k, v in values.items():
        if k not in ATTENTION_RANGES:
            raise ValueError(f"unknown attention field {k!r}; expected {sorted(ATTENTION_RANGES)}")
        lo, hi = ATTENTION_RANGES[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
            raise ValueError(f"attention.{k} must be a number in [{lo}, {hi}], got {v!r}")
        out[k] = float(v)
    return out


class TargetSelector:
    """Pick the face the wearer is looking at.

    - Cost of a track = distance of its box centre to the image centre,
      minus optional area and `p_speaking` bonuses (lower is better).
    - Acquisition (no target yet): the best candidate must stay best for
      `acquire_s`. When the two best candidates' costs are within `tie_margin`
      (equally central), the one with the higher mean `p_speaking` over the last
      1.5 s wins (speaking alone beats silent); never applied to a lock or a switch.
    - Switching: a challenger must beat the current target's cost by `margin`
      on every frame for `dwell_s`.
    - Hold: a missing target keeps its id (with decaying confidence) and its
      last box for `hold_s`; then it is dropped.
    - Eligibility: a track may become target only if it was seen speaking
      (`p_speaking > speak_thresh`) within `speech_memory_s`; the current
      target is never dropped for being silent. `speech_memory_s=None`
      disables the gate (pure centre-of-view).
    - `lock(track_id)` pins the target until `unlock()` (T-107): no challenger, no hold_s
      drop. A missing locked face is "reacquiring" for `hold_lock_s` (the state keeps its id) and
      is re-attached to a new track whose box centre is within `reattach_dist` of its last box
      (re-detection with a new id, tracker reset after a vision stall); a track that was visible
      next to the locked face in its last frame is someone else and never inherits the lock.
      After `hold_lock_s` the lock is lost (`lock_lost`, state track_id None, locked True) and the
      selector keeps searching for that face (its position or id), never free selection.
    - `forget_ids()` (tracker ids restart) keeps a lock, matched by position only from then on.
    """

    def __init__(
        self,
        *,
        dwell_s: float = 0.5,
        margin: float = 0.04,
        hold_s: float = ATTENTION_DEFAULTS["hold_s"],
        acquire_s: float = 0.1,
        area_weight: float = 0.0,
        speaking_bonus: float = 0.0,
        speak_thresh: float = 0.5,
        speech_memory_s: float | None = 10.0,
        conf_dist: float = 0.5,
        tie_margin: float = ATTENTION_DEFAULTS["tie_margin"],
        hold_lock_s: float = ATTENTION_DEFAULTS["hold_lock_s"],
        reattach_dist: float = REATTACH_DIST,
    ) -> None:
        self.hold_lock_s = hold_lock_s
        self.reattach_dist = reattach_dist
        self.tie_margin = tie_margin
        self.dwell_s = dwell_s
        self.margin = margin
        self.hold_s = hold_s
        self.acquire_s = acquire_s
        self.area_weight = area_weight
        self.speaking_bonus = speaking_bonus
        self.speak_thresh = speak_thresh
        self.speech_memory_s = speech_memory_s
        self.conf_dist = conf_dist
        self.reset()

    def reset(self) -> None:
        self._target_id: int | None = None
        self._target_track: FaceTrack | None = None  # last seen box of the target
        self._target_seen_ts = 0.0
        self._since_ts = 0.0
        self._locked = False
        self._lock_lost = False  # explicit lock whose face has been gone for hold_lock_s
        self._ids_stale = False  # tracker ids restarted: the locked id may now name a stranger
        self._neighbours: set[int] = set()  # tracks visible with the locked face in its last frame
        self._challenger_id: int | None = None
        self._challenger_start = 0.0
        self._last_spoke: dict[int, float] = {}
        self._speak_hist: dict[int, deque[tuple[float, float]]] = {}
        self._first_seen: dict[int, float] = {}
        self._ts = 0.0
        self._switch_seq = 0
        self._state = TargetState(None, 0.0, False, 0.0)

    @property
    def attention(self) -> dict[str, float]:
        return {"dwell_s": self.dwell_s, "hold_s": self.hold_s, "switch_margin": self.margin,
                "tie_margin": self.tie_margin, "hold_lock_s": self.hold_lock_s}

    def set_attention(self, **values: float) -> dict[str, float]:
        """Apply validated dwell_s / hold_s / switch_margin / tie_margin / hold_lock_s live; returns the new values."""
        v = check_attention(values)
        self.dwell_s = v.get("dwell_s", self.dwell_s)
        self.hold_s = v.get("hold_s", self.hold_s)
        self.margin = v.get("switch_margin", self.margin)
        self.tie_margin = v.get("tie_margin", self.tie_margin)
        self.hold_lock_s = v.get("hold_lock_s", self.hold_lock_s)
        return self.attention

    @property
    def state(self) -> TargetState:
        return self._state

    @property
    def candidate_id(self) -> int | None:
        """The face currently dwelling, before acquisition or a target switch."""
        return self._challenger_id

    @property
    def missing_s(self) -> float:
        """Missing duration on the video clock while a target is held."""
        if self._target_id is None and not self._lock_lost:
            return 0.0
        return max(0.0, self._ts - self._target_seen_ts)

    @property
    def locked(self) -> bool:
        """An explicit lock is set (also while its face is missing or lost)."""
        return self._locked

    @property
    def lock_lost(self) -> bool:
        """An explicit lock whose face has been missing for `hold_lock_s` (still searching for it)."""
        return self._locked and self._lock_lost

    def forget_ids(self) -> None:
        """Tracker ids restart (camera dark / vision stall). A lock is kept and re-attached by
        position; without one this is a full reset."""
        if not self._locked:
            self.reset()
            return
        self._ids_stale = True
        self._neighbours = set()
        self._challenger_id = None
        self._last_spoke.clear()
        self._speak_hist.clear()
        self._first_seen.clear()
        self._state = TargetState(None, 0.0, True, self._since_ts, self._switch_seq)

    def lock(self, track_id: int, ts: float | None = None) -> TargetState:
        """Pin `track_id` as the target (maps to `control.target.lock_track_id`)."""
        ts = self._ts if ts is None else ts
        if track_id != self._target_id or self._lock_lost or self._ids_stale:
            self._set_target(track_id, None, ts)
        self._locked = True
        self._lock_lost = False
        self._ids_stale = False
        self._challenger_id = None
        self._state = TargetState(self._target_id, self._state.confidence, True, self._since_ts, self._switch_seq)
        return self._state

    def unlock(self) -> None:
        self._locked = False
        self._lock_lost = False
        self._state = TargetState(self._state.track_id, self._state.confidence, False, self._state.since_ts,
                                  self._switch_seq)

    def cost(self, track: FaceTrack) -> float:
        x, y, w, h = track.box
        d = math.hypot(x + w / 2 - 0.5, y + h / 2 - 0.5)
        return d - self.area_weight * math.sqrt(max(w * h, 0.0)) - self.speaking_bonus * track.p_speaking

    def update(self, tracks: list[FaceTrack], ts: float) -> TargetState:
        self._ts = ts
        by_id = {t.track_id: t for t in tracks}
        self._remember_speech(tracks, ts)

        if self._locked:
            self._follow_lock(tracks, by_id, ts)
        elif self._target_id is not None:
            current = by_id.get(self._target_id)
            if current is not None:
                self._target_track = current
                self._target_seen_ts = ts
            elif ts - self._target_seen_ts >= self.hold_s - _EPS:
                self._drop_target(ts)

        progress = 0.0
        if not self._locked:  # an explicit lock never falls back to free selection (T-107)
            progress = self._challenge(tracks, ts)

        # a lost lock, or one whose id may since name a stranger (ids restarted), shows no track
        shown = None if self.lock_lost or (self._locked and self._ids_stale) else self._target_id
        self._state = TargetState(shown, self._confidence(by_id, ts, progress), self._locked, self._since_ts,
                                  self._switch_seq)
        return self._state

    def _follow_lock(self, tracks: list[FaceTrack], by_id: dict[int, FaceTrack], ts: float) -> None:
        """Keep the locked face: by id while ids are valid, else re-attach a re-detection by position."""
        current = None if self._ids_stale or self._target_id is None else by_id.get(self._target_id)
        if current is None:
            current = self._reattach(tracks)
            if current is not None and current.track_id != self._target_id:
                self._target_id = current.track_id  # same person, new id: not a switch (switch_seq kept)
        if current is not None:
            self._target_track = current
            self._target_seen_ts = ts
            self._ids_stale = False
            self._lock_lost = False
            self._neighbours = {t.track_id for t in tracks if t.track_id != current.track_id}
        elif not self._lock_lost and ts - self._target_seen_ts >= self.hold_lock_s - _EPS:
            self._lock_lost = True  # keep _target_track: the same face may come back there

    def _reattach(self, tracks: list[FaceTrack]) -> FaceTrack | None:
        """The track nearest the locked face's last box, within reattach_dist; never a face that was
        visible next to it (that is the other person), unless ids restarted since."""
        last = self._target_track
        if last is None:
            return None
        lx, ly, lw, lh = last.box
        best, best_d = None, self.reattach_dist + _EPS
        for t in tracks:
            if not self._ids_stale and t.track_id in self._neighbours:
                continue
            x, y, w, h = t.box
            d = math.hypot(x + w / 2 - lx - lw / 2, y + h / 2 - ly - lh / 2)
            if d <= best_d:
                best, best_d = t, d
        return best

    def _challenge(self, tracks: list[FaceTrack], ts: float) -> float:
        """Advance the challenger; switch when it has dwelled. Returns dwell progress."""
        candidates = [t for t in tracks if t.track_id != self._target_id and self._eligible(t.track_id, ts)]
        best = min(candidates, key=self.cost, default=None)
        if self._target_id is None and tracks:
            best, _, ready = self._tiebreak(best, candidates, tracks, ts)
            if not ready:  # equally central faces and under 1.5 s of speaking evidence: wait
                self._challenger_id = None
                return 0.0
        if best is None:
            self._challenger_id = None
            return 0.0
        if self._target_id is None:
            need = self.acquire_s  # a real tie already waited for 1.5 s of evidence in _tiebreak
        else:
            if self._target_track is None or self.cost(self._target_track) - self.cost(best) < self.margin:
                self._challenger_id = None
                return 0.0
            need = self.dwell_s
        if best.track_id != self._challenger_id:
            self._challenger_id = best.track_id
            self._challenger_start = ts
        held = ts - self._challenger_start
        if held >= need - _EPS:
            self._set_target(best.track_id, best, ts)
            return 0.0
        return held / need if need > 0 else 1.0

    def _mean_speaking(self, track_id: int, ts: float) -> float:
        hist = self._speak_hist.get(track_id)
        vals = [p for t, p in hist if ts - t <= _SPEAK_WINDOW_S + _EPS] if hist else []
        return sum(vals) / len(vals) if vals else 0.0

    def _tiebreak(self, best: FaceTrack | None, candidates: list[FaceTrack], tracks: list[FaceTrack],
                  ts: float) -> tuple[FaceTrack | None, bool, bool]:
        """Acquisition only. Among all visible faces as central as the most central one (within
        tie_margin), the eligible one talking most over the last 1.5 s, if it leads the runner-up
        by _TIE_GAP. Returns (face, decided_by_speech, ready); not ready while a real tie has
        less than 1.5 s of evidence (a face that spoke once for a frame must not win alone)."""
        base = min(self.cost(t) for t in tracks)
        tied = [t for t in tracks if self.cost(t) - base <= self.tie_margin + _EPS]
        if len(tied) < 2:
            return best, False, True
        span = min(ts - self._first_seen.get(t.track_id, ts) for t in tied)
        means = {t.track_id: self._mean_speaking(t.track_id, ts) for t in tied}
        if min(means.values()) > self.speak_thresh:
            return best, False, True  # everyone is clearly talking: nothing to wait for
        if span < _MIN_EVIDENCE_S - _EPS:
            return best, False, False
        elig = [t for t in tied if t in candidates]
        top = max(elig, key=lambda t: (means[t.track_id], -self.cost(t)), default=None)
        lead = 0.0 if top is None else means[top.track_id] - max(m for k, m in means.items() if k != top.track_id)
        decisive = lead >= _TIE_GAP
        if not decisive and span < _SPEAK_WINDOW_S - _EPS and min(means.values()) <= self.speak_thresh:
            return best, False, False  # no clear winner yet and someone may still start talking
        if not decisive:
            return best, False, True  # all talking (or none): centre-of-view decides
        return top, True, True

    def _eligible(self, track_id: int, ts: float) -> bool:
        if self.speech_memory_s is None:
            return True
        spoke = self._last_spoke.get(track_id)
        return spoke is not None and ts - spoke <= self.speech_memory_s + _EPS

    def _remember_speech(self, tracks: list[FaceTrack], ts: float) -> None:
        for t in tracks:
            hist = self._speak_hist.setdefault(t.track_id, deque())
            self._first_seen.setdefault(t.track_id, ts)
            hist.append((ts, t.p_speaking))
            while hist and ts - hist[0][0] > _SPEAK_WINDOW_S + _EPS:
                hist.popleft()
            if t.p_speaking > self.speak_thresh:
                self._last_spoke[t.track_id] = ts
        if self.speech_memory_s is not None and len(self._last_spoke) > 64:
            horizon = ts - self.speech_memory_s
            self._last_spoke = {k: v for k, v in self._last_spoke.items() if v >= horizon}
            self._speak_hist = {k: h for k, h in self._speak_hist.items() if h and h[-1][0] >= horizon}
            self._first_seen = {k: v for k, v in self._first_seen.items() if k in self._speak_hist}

    def _set_target(self, track_id: int, track: FaceTrack | None, ts: float) -> None:
        if self._target_id is not None and track_id != self._target_id:
            self._switch_seq += 1  # a deliberate switch away from a held target
        self._target_id = track_id
        self._target_track = track
        self._target_seen_ts = ts
        self._since_ts = ts
        self._challenger_id = None

    def _drop_target(self, ts: float) -> None:
        # Keep any challenger already dwelling so it can take over at once.
        self._target_id = None
        self._target_track = None
        self._locked = False
        self._since_ts = ts

    def _confidence(self, by_id: dict[int, FaceTrack], ts: float, progress: float) -> float:
        if self._target_id is None or self.lock_lost:
            return 0.0
        missing = ts - self._target_seen_ts
        seen = self._target_seen_ts == ts and self._target_id in by_id
        if self._locked:
            return 1.0 if seen else max(0.0, 1.0 - missing / self.hold_lock_s)
        presence = 1.0 if seen else max(0.0, 1.0 - missing / self.hold_s)
        track = self._target_track
        if track is None:
            centre = 0.5
        else:
            x, y, w, h = track.box
            d = math.hypot(x + w / 2 - 0.5, y + h / 2 - 0.5)
            centre = max(0.0, 1.0 - d / self.conf_dist)
        return min(1.0, max(0.0, centre * presence * (1.0 - 0.5 * progress)))
