"""Automatic enrollment for the Enrolled tier: target speech picked by active-speaker detection.

Every 10 ms chunk is kept when the camera says the target alone is talking: the target track's
``p_speaking >= TARGET_P``, every other track below ``OTHER_P``, and the chunk is voiced
(RMS above ``VOICED_RMS``, the same -40 dBFS the synth ASR's energy VAD uses, or since T-119 9 dB
above the adaptive noise floor and above -55 dBFS: :class:`VoicedGate`). The kept chunks are
concatenated until ``seconds`` of speech exist; after that the clip is :attr:`ready`.

Numbers from the T-061 association eval (``eval/association_eval.py``,
``docs/decisions/association.md``): Light-ASD gives a silent face next to a talker
``p_speaking`` of about 0.5 on average, so ``OTHER_P = 0.5`` kept the gate shut for half of the
target's solo speech and 3 s took 6-10 s to collect (16 of 22 targets never enrolled). With
``OTHER_P = 0.7`` and 2 s (``ENROLL_S``) the median lock is 3.5 s and purity stays 0.96.

The same gate mirrored collects a **negative** enrollment (``other_s``, default 2 s): the other
people talking alone. The extractor then asks "closer to the target than to the person next to
them?" instead of verifying one voice in isolation, which is what keeps the other person out of
the output while the target is silent. On a deliberate switch the old target's clip becomes the
negative.

Track ids change when a face is lost and found again (occlusion, turning away, leaving and coming
back). With ``sticky`` (default) a target change to a new track while the old track is gone keeps
serving both clips until the new collection is ready, instead of dropping to "enrolling"; a change
to a track visible next to the old one, or any switch the user made (``control.target.lock_track_id``,
a challenger that won: ``TargetState.switch_seq`` moved) is deliberate and starts over, even if the
old face is off screen (T-083).

One enroller per session (T-078). The session server puts its own :class:`AutoEnroller` on
``ctx.telemetry.extras[ENROLLER_KEY]`` and feeds it every 10 ms hop of room audio, whatever mode
plays; every enrolled ``tse`` stage it builds adopts that one instead of its own. A latency-preset
change (a new ``tiers(preset=...)`` ladder) or a Focus HQ -> Text -> Focus HQ round trip then
keeps the clips instead of collecting them again (about 27 s on the demo clip, T-072). Stages
outside a session (benches, the live loop, tests) keep their own enroller.

T-092 (``eval/enroll_eval.py``): a chunk with the target at ``RELAX_TARGET_P`` (0.4) also counts
while every other face is below ``RELAX_OTHER_P`` (0.3); demo clip 19.5 -> 13.3 s offline, scene purity
unchanged. Every chunk leaves its verdict in :attr:`AutoEnroller.reason` and the commonest
rejection of the last second becomes :attr:`AutoEnroller.hint` (none while over a quarter of it was kept,
T-119), the acquisition pill's "too quiet" (ASD says the target talks, the mic hears nothing) /
"need them to talk alone" (both talk) / "other person talking" (only the other) / "waiting for them to
talk" (nobody's lips move)).

Manual calibration (T-085, ``control.enroll``): :meth:`calibrate` opens a window of ``seconds``
(default ``CALIBRATE_S`` = 3 s) of offered audio in which EVERY voiced chunk is kept for the target,
whatever ASD says (the wearer asked the target to speak alone). When the window closes with at least
``CALIBRATE_MIN_S`` of voiced audio the clip is :attr:`calibrated`: it is ready at once, a re-detected
track (new id, no deliberate switch) is re-bound to it instead of collecting again, and only
:meth:`clear` or a deliberate target switch drops it. Too little speech in the window: back to the
automatic collection (:attr:`calibration_failed`). While calibrating, the previous clip keeps playing.

The clips are audio only and stay in memory for the session (they are voices, used as speaker
references, never written to disk or logged).

Voice profiles (T-125): ``$HEARING_PROFILES_DIR/<name>/*.wav`` (gitignored, written by
``scripts/make_profile.py``) are pre-recorded voices; ``scripts/enroll_profiles.py`` writes them versioned
(T-172), ``<name>/v<N>/*.wav`` + ``manifest.json`` with ``<name>/current`` naming the active version. :meth:`AutoEnroller.enroll_profile` puts one on
the calibrated path (:meth:`enroll_clip`) at session start, with another profile as the known
negative from t=0, so the Enrolled tier is ready before anyone speaks. The clip is the profile's
clean pieces (``PROFILE_PIECE_S``, 2-3 s) concatenated and registered here; the enrolled candidate
embeds a registered clip as the mean of the pieces' unit ECAPA embeddings, without the separator
pass a room clip gets (:func:`profile_pieces`). :attr:`AutoEnroller.source` says where the playing
clip came from: ``profile`` / ``calibrated`` / ``auto``. Off when ``HEARING_PROFILES_DIR`` is unset.
"""

from __future__ import annotations

import os
from collections import deque
from itertools import pairwise
from pathlib import Path

import numpy as np

from hearing.types import StageContext

TARGET_P = 0.7  # target track p_speaking needed for a chunk to count
OTHER_P = 0.7  # any other track at or above this: overlapping talkers, skip the chunk (T-061: was 0.5)
ENROLL_S = 1.75  # T-092: was 2.0 (T-061); 1.75 s + the relaxed gate: M/F acc 0.951 vs 0.945, ready mean 2.94 s
OTHER_S = 2.0  # negative enrollment (the other people); 1 s measured worse (acc 0.86 vs 0.95)
VOICED_RMS = 0.01  # ~ -40 dBFS (hearing.synth.asr.VOICED_RMS): always voiced at or above this
# T-119: a laptop mic at arm's length gives speech at -50..-35 dBFS over a -60..-55 dBFS room (the
# first live /judge session: 92 % of mic chunks below -40 dBFS, 398 "too quiet" ticks while the
# target talked). A chunk is also voiced FLOOR_MARGIN_DB above the adaptive noise floor (the
# FLOOR_PCTL-th percentile of the last FLOOR_WINDOW_S of chunk levels), never below VOICED_MIN_DB.
FLOOR_MARGIN_DB = 9.0
VOICED_MIN_DB = -55.0  # absolute minimum of the relative gate (a digital-silent floor cannot open it)
FLOOR_WINDOW_S = 3.0
FLOOR_PCTL = 10.0
# T-145: hysteresis. Right after a voiced chunk the relative gate stays open HOLD_DB lower (a word's
# decaying tail and soft consonants), never closer than HOLD_FLOOR_DB to the noise floor.
HOLD_DB = 6.0
HOLD_FLOOR_DB = 3.0
RELAX_TARGET_P = 0.4  # T-092: also keep a chunk with the target at/above this (0.5 -> 0.4: demo 15.8 -> 13.3 s offline) ...
RELAX_OTHER_P = 0.3  # ... while every other face is clearly quiet (below this)
HINT_S = 1.0  # the acquisition hint names the commonest rejection over this last stretch
HINT_KEPT = 0.25  # ... unless more than this share of it was kept (T-119)
# Why a chunk was not kept (``AutoEnroller.reason``; ``target_quiet`` split by whether another face
# talks) -> what the pill asks for while acquiring
HINTS = {"unvoiced": "too quiet", "other_talking": "need them to talk alone",
         "target_quiet_other": "other person talking", "target_quiet": "waiting for them to talk"}
CALIBRATE_S = 3.0  # control.enroll {"action": "start"} default window (T-085)
CALIBRATE_MIN_S = 1.0  # voiced audio a calibration window needs, else it falls back to auto
ENROLLER_KEY = "enroller"  # ctx.telemetry.extras: the session's shared AutoEnroller
PROFILES_ENV = "HEARING_PROFILES_DIR"  # voice profiles root (T-125); unset = profiles off
PROFILE_TARGET_ENV = "HEARING_VOICE_PROFILE"  # the judge profile's target profile (default: the first by name)
PROFILE_PIECE_S = 2.5  # a profile's clips are cut into pieces of 2-3 s, one ECAPA embedding each
PROFILE_MIN_PIECE_S = 1.5  # a shorter tail is merged into the previous piece (or kept alone if it is the clip)
_PROFILE_CLIPS: dict[tuple[int, int], tuple[int, ...]] = {}  # clip key -> piece lengths (T-125)
_PROFILE_NAMES: dict[tuple[int, int], str] = {}  # clip key -> profile name (T-386: the fixed-order slot)


def clip_key(clip: np.ndarray) -> tuple[int, int]:
    return len(clip), hash(np.asarray(clip, dtype=np.float32).tobytes())


def profiles_dir() -> Path | None:
    """``$HEARING_PROFILES_DIR`` when set and a directory, else None (profiles off)."""
    root = os.environ.get(PROFILES_ENV, "").strip()
    return Path(root).expanduser() if root and Path(root).expanduser().is_dir() else None


CURRENT_FILE = "current"  # profiles/<name>/current holds the active version's dir name, e.g. "v2" (T-172)


def profile_version(d: Path) -> str | None:
    """The active version of the profile dir ``d`` (``current`` names an existing ``v<N>`` dir), else None."""
    f = d / CURRENT_FILE
    if not f.is_file():
        return None
    v = f.read_text(errors="replace").strip()
    return v if v and "/" not in v and not v.startswith(".") and (d / v).is_dir() else None


def profile_audio_dir(d: Path) -> Path:
    """Where the profile dir ``d`` keeps its wavs: ``d/<current>/`` when versioned (T-172), else ``d`` (flat)."""
    v = profile_version(d)
    return d / v if v else d


def list_profiles(root: Path | None = None) -> list[str]:
    """Names of the profile directories under ``root`` (default :func:`profiles_dir`) holding a wav."""
    root = root if root is not None else profiles_dir()
    if root is None:
        return []
    return sorted(d.name for d in root.iterdir() if d.is_dir() and any(profile_audio_dir(d).glob("*.wav")))


def profile_versions(root: Path | None = None) -> dict[str, str | None]:
    """{name: active version (``"v2"``) or None for a flat, unversioned profile} for :func:`list_profiles`."""
    root = root if root is not None else profiles_dir()
    return {} if root is None else {n: profile_version(root / n) for n in list_profiles(root)}


def split_pieces(x: np.ndarray, sr: int = 16000, piece_s: float = PROFILE_PIECE_S) -> list[np.ndarray]:
    """Cut ``x`` into ``piece_s`` pieces; a tail under ``PROFILE_MIN_PIECE_S`` joins the last piece."""
    x = np.asarray(x, dtype=np.float32)
    n, short = max(1, round(piece_s * sr)), round(PROFILE_MIN_PIECE_S * sr)
    pieces = [x[i : i + n] for i in range(0, len(x), n)]
    if len(pieces) > 1 and len(pieces[-1]) < short:
        tail = pieces.pop()
        pieces[-1] = np.concatenate([pieces[-1], tail])
    return [p for p in pieces if len(p)]


def load_profile(name: str, root: Path | None = None, sr: int = 16000) -> list[np.ndarray]:
    """The pieces of profile ``name`` (every ``*.wav`` of its active version, resampled to ``sr``)."""
    from hearing.audio.io import load

    root = root if root is not None else profiles_dir()
    if root is None:
        raise ValueError(f"voice profiles are off ({PROFILES_ENV} is not set)")
    d = root / name
    if not name or "/" in name or name.startswith(".") or not d.is_dir():
        raise ValueError(f"no voice profile {name!r} in {root}")
    pieces = [p for f in sorted(profile_audio_dir(d).glob("*.wav")) for p in split_pieces(load(f, sr)[0], sr)]
    if not pieces:
        raise ValueError(f"voice profile {name!r} has no audio")
    return pieces


def profile_clip(pieces: list[np.ndarray], name: str | None = None) -> np.ndarray:
    """The pieces concatenated into one clip, registered so :func:`profile_pieces` can cut it again
    (and :func:`profile_name` can name it when ``name`` is given)."""
    pieces = [np.asarray(p, dtype=np.float32) for p in pieces if len(p)]
    clip = np.concatenate(pieces)
    key = clip_key(clip)
    _PROFILE_CLIPS[key] = tuple(len(p) for p in pieces)
    if name:
        _PROFILE_NAMES[key] = name
    return clip


def profile_name(clip: np.ndarray) -> str | None:
    """The profile name a clip was built for by ``profile_clip(pieces, name)``, None for any other clip."""
    return _PROFILE_NAMES.get(clip_key(clip)) if len(clip) else None


def profile_pieces(clip: np.ndarray) -> list[np.ndarray] | None:
    """The pieces of a clip built by :func:`profile_clip`, None for any other clip."""
    lengths = _PROFILE_CLIPS.get(clip_key(clip)) if len(clip) else None
    if lengths is None:
        return None
    bounds = np.cumsum((0, *lengths))
    return [clip[a:b] for a, b in pairwise(bounds)]


def profile_pairs(root: Path | None = None) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """``(target clip, negative clip)`` per saved profile, as :meth:`AutoEnroller.enroll_profile` builds
    them with every other profile as the negative (T-136: prepared ahead of a swap)."""
    names = list_profiles(root)
    pieces = {n: load_profile(n, root) for n in names}
    pairs = []
    for n in names:
        others = [p for o in names if o != n for p in pieces[o]]
        pairs.append((profile_clip(pieces[n], n), profile_clip(others) if others else None))
    return pairs


def level_db(x: np.ndarray) -> float:
    """RMS level of a chunk in dBFS (-120 for an empty or digital-silent chunk)."""
    if not len(x):
        return -120.0
    rms = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return 20.0 * float(np.log10(max(rms, 1e-6)))


class VoicedGate:
    """Voiced test for 10 ms chunks, relative to an adaptive noise floor (T-119).

    ``voiced(x)`` is True when the chunk's level is at or above ``VOICED_RMS`` (-40 dBFS, the old
    absolute gate, so loud clips behave as before) **or** at least ``FLOOR_MARGIN_DB`` above the
    noise floor and above ``VOICED_MIN_DB``. The floor is the ``FLOOR_PCTL``-th percentile of the
    last ``FLOOR_WINDOW_S`` of chunk levels (speech has pauses, so the low percentile stays on the
    room), recomputed every 10 chunks once 1 s has been seen; before that the gate is the absolute -40 dBFS.
    T-145: right after a voiced chunk the relative threshold drops ``HOLD_DB`` (never closer than
    ``HOLD_FLOOR_DB`` to the floor), so word tails count; the teammates' takes at -45 dBFS then enroll after
    1.95-2.47 s of talking (was 2.81-4.13 s; ``eval/enroll_eval.py --team``)."""

    def __init__(self, sr: int = 16000, chunk: int = 160):
        self._levels: deque[float] = deque(maxlen=max(1, round(FLOOR_WINDOW_S * sr / chunk)))
        self._since = 0
        self._min_n = min(self._levels.maxlen, round(sr / chunk))  # 1 s before the floor is trusted
        self.floor_db: float | None = None
        self.last_db = -120.0
        self._open = False  # the last chunk was voiced (T-145 hysteresis)

    @property
    def threshold_db(self) -> float:
        absolute = 20.0 * float(np.log10(VOICED_RMS))
        if self.floor_db is None:
            return absolute
        return min(absolute, max(VOICED_MIN_DB, self.floor_db + FLOOR_MARGIN_DB))

    def voiced(self, x: np.ndarray) -> bool:
        if not len(x):
            return False
        db = self.last_db = level_db(x)
        self._levels.append(db)
        self._since += 1
        if self._since >= 10 and len(self._levels) >= self._min_n:
            self._since = 0
            self.floor_db = float(np.percentile(np.fromiter(self._levels, float), FLOOR_PCTL))
        threshold = self.threshold_db
        if self._open and self.floor_db is not None and HOLD_DB > 0:
            threshold = min(threshold, max(threshold - HOLD_DB, self.floor_db + HOLD_FLOOR_DB))
        self._open = db >= threshold
        return self._open


class AutoEnroller:
    def __init__(self, seconds: float = ENROLL_S, sr: int = 16000, *, other_s: float = OTHER_S,
                 sticky: bool = True, target_p: float = TARGET_P, other_p: float = OTHER_P,
                 other_hold_s: float = 0.0, relax_target_p: float | None = RELAX_TARGET_P,
                 relax_other_p: float = RELAX_OTHER_P):
        if seconds <= 0:
            raise ValueError(f"enrollment length must be positive, got {seconds}")
        self.need = round(seconds * sr)
        self.other_need = round(max(0.0, other_s) * sr)
        self.sr = sr
        self.sticky = bool(sticky)
        self.target_p, self.other_p = float(target_p), float(other_p)
        self.relax_target_p = None if relax_target_p is None else float(relax_target_p)
        self.relax_other_p = float(relax_other_p)
        # last verdict: kept / ready / no_target / unvoiced / target_quiet / other_talking (T-092)
        self.reason = "no_target"
        self._recent: deque[str] = deque(maxlen=max(1, round(HINT_S * 100)))
        self._hold: deque[float] = deque(maxlen=max(1, round(other_hold_s * 100)))  # 10 ms chunks
        self._switch_seq = 0  # TargetState.switch_seq last seen
        self.offered = 0  # samples offered since construction (positions in ``spans``)
        self.calibration_failed = False  # the last calibration window had too little speech
        self.gate = VoicedGate(sr)  # T-119: survives reset(), the room does not change with the target
        # T-136: (target, negative) clips of every saved profile; the enrolled stage prepares their
        # embeddings once, so a swap (keys 1/2, a Name Call snap) plays within one window
        self.preload: list[tuple[np.ndarray, np.ndarray | None]] = []
        self.reset()

    def reset(self) -> None:
        self.track_id: int | None = None
        self.calibrated = False  # the target clip came from a manual calibration (T-085)
        self.source = "auto"  # where the clip comes from: profile / calibrated / auto (T-125)
        self.profile: str | None = None  # the voice profile's name when source == "profile"
        self._other_fixed = False  # the negative is a voice profile: no automatic replacement
        self._cal_left = 0  # samples left in an open calibration window
        self._cal_total = 0
        self._parts: list[np.ndarray] = []
        self._n = 0
        self._held: np.ndarray | None = None
        self.spans: list[tuple[int, int]] = []  # (offered position, length) of the kept chunks
        self._oparts: list[np.ndarray] = []
        self._on = 0
        self._oheld: np.ndarray | None = None
        self.other_spans: list[tuple[int, int]] = []
        self._recent.clear()

    @property
    def collected_s(self) -> float:
        return self._n / self.sr

    @property
    def ready(self) -> bool:
        return self.calibrated or (not self.calibrating and self._n >= self.need)

    @property
    def calibrating(self) -> bool:
        return self._cal_left > 0

    @property
    def calibration_progress(self) -> tuple[float, float]:
        """(seconds of the calibration window elapsed, window length)."""
        return (self._cal_total - self._cal_left) / self.sr, self._cal_total / self.sr

    def calibrate(self, seconds: float = CALIBRATE_S) -> None:
        """Start a manual calibration window: the next ``seconds`` of voiced audio are the target's."""
        if seconds <= 0:
            raise ValueError(f"calibration length must be positive, got {seconds}")
        old, old_other, track = self.clip(), self.other_clip(), self.track_id
        held_from = (self.source, self.profile, self._other_fixed) if old is not None else ("auto", None, False)
        self.reset()
        self._held, self._oheld, self.track_id = old, old_other, track
        self._held_from = held_from  # restored when the window fails
        self._cal_left = self._cal_total = round(seconds * self.sr)
        self.calibration_failed = False

    def enroll_clip(self, clip: np.ndarray, other: np.ndarray | None = None) -> None:
        """Use ``clip`` (already collected elsewhere: the Name Call utterance, T-120) as the calibrated
        target clip at once; ``other`` (if given) becomes the negative enrollment."""
        x = np.asarray(clip, dtype=np.float32)
        if not len(x):
            raise ValueError("an enrollment clip needs audio")
        track = self.track_id
        self.reset()
        self.track_id = track
        self._parts, self._n, self.calibrated = [x.copy()], len(x), True
        self._oheld = None if other is None else np.asarray(other, dtype=np.float32).copy()
        self.calibration_failed = False
        self.source = "calibrated"

    def enroll_profile(self, name: str, pieces: list[np.ndarray],
                       other: list[np.ndarray] | None = None) -> None:
        """Voice profile ``name`` (T-125) as the calibrated target clip; ``other`` (another profile's
        pieces) is the known negative and is not replaced by the automatic negative collection."""
        self.enroll_clip(profile_clip(pieces, name), profile_clip(other) if other else None)
        self.source, self.profile, self._other_fixed = "profile", name, bool(other)

    def clear(self) -> None:
        """Forget both clips (``control.enroll {"action": "clear"}``); automatic collection restarts."""
        track = self.track_id
        self.reset()
        self.track_id = track
        self.calibration_failed = False

    @property
    def hint(self) -> str:
        """What the collection is waiting for, from the commonest rejection of the last ``HINT_S``
        (``""`` when nothing is missing or chunks are being kept)."""
        if self.ready or not self._recent:
            return ""
        counts: dict[str, int] = {}
        for r in self._recent:
            counts[r] = counts.get(r, 0) + 1
        if counts.get("kept", 0) > HINT_KEPT * len(self._recent):
            return ""  # collecting: the pauses between words are not a problem to report (T-119)
        best = max(counts, key=counts.__getitem__)
        return HINTS.get(best, "")

    @property
    def profile_negative(self) -> bool:
        """The target is a voice profile and the negative is saved voices too (T-155): the closed set
        tells the pair apart, so the same-gender guard does not apply, whatever loaded it (the
        session-start pair, a key 1/2 swap with every other profile, a Name Call snap)."""
        return self.source == "profile" and self._other_fixed

    @property
    def holding(self) -> bool:
        """Serving the previous clip of a re-detected target while collecting anew (``sticky``)."""
        return not self.ready and self._held is not None

    def clip(self) -> np.ndarray | None:
        """The enrollment audio (exactly ``seconds`` long) once :attr:`ready`, the held clip
        while :attr:`holding`, else None."""
        if not self.ready:
            return self._held
        return np.concatenate(self._parts)[: None if self.calibrated else self.need]

    def other_clip(self) -> np.ndarray | None:
        """The negative enrollment (``other_s`` of the other people) once collected, else the one
        carried over from before the last target change, else None."""
        if self.other_need and self._on >= self.other_need and not self._other_fixed:
            return np.concatenate(self._oparts)[: self.other_need]
        return self._oheld

    def update(self, samples: np.ndarray, ctx: StageContext) -> bool:
        """Offer one chunk of room audio; returns True when it was kept for the target."""
        x = np.asarray(samples, dtype=np.float32)
        pos = self.offered
        self.offered += len(x)
        target = ctx.target.track_id
        if self.calibrating:
            self._switch_seq = ctx.target.switch_seq  # the wearer is facing the person they calibrate
            return self._calibration_chunk(x, pos, target)
        if target is None:
            self.reason = "no_target"
            return False
        if target != self.track_id and self.calibrated and (self.track_id is None or (
                ctx.target.switch_seq == self._switch_seq
                and all(t.track_id != self.track_id for t in ctx.tracks))):
            # a calibrated clip follows a re-detected face (old track gone); a deliberate switch, or a
            # new target while the calibrated face is still in view, drops it
            self.track_id = target
        if target != self.track_id:
            gone = all(t.track_id != self.track_id for t in ctx.tracks)
            deliberate = ctx.target.switch_seq != self._switch_seq
            # T-083: a lock / challenger win is a new person even when the old face is off screen
            redetected = self.sticky and self.track_id is not None and gone and not deliberate
            old, old_other = self.clip(), self.other_clip()
            self.reset()
            if redetected:  # same person, new track id: keep both clips until refreshed
                self._held, self._oheld = old, old_other
            elif self.other_need and old is not None:  # deliberate switch: the old target is "other"
                self._oheld = old
            self.track_id = target
        self._switch_seq = ctx.target.switch_seq
        p_target, others = 0.0, [0.0, 0.0]
        for t in ctx.tracks:
            if t.track_id == target:
                p_target = float(t.p_speaking)
            else:
                others = sorted([*others, float(t.p_speaking)], reverse=True)[:2]
        self._hold.append(others[0])
        others[0] = max(self._hold)  # the other face must have been quiet for other_hold_s
        voiced = self.gate.voiced(x)
        if (voiced and self._on < self.other_need and others[0] >= self.target_p
                and p_target < self.other_p and others[1] < self.other_p):
            self._oparts.append(x.copy())
            self._on += len(x)
            self.other_spans.append((pos, len(x)))
        if self.ready:
            self.reason = "ready"
            return False
        relaxed = (self.relax_target_p is not None and p_target >= self.relax_target_p
                   and others[0] < self.relax_other_p)
        talking = p_target >= self.target_p or relaxed
        if not voiced:  # T-119: silence while ASD says nobody talks is a pause, not a quiet mic
            self.reason = "unvoiced" if talking else "target_quiet"
        elif not talking:
            self.reason = "target_quiet"
        elif others[0] >= self.other_p:
            self.reason = "other_talking"
        else:
            self.reason = "kept"
        self._recent.append("target_quiet_other" if self.reason == "target_quiet" and others[0] >= self.other_p
                            else self.reason)
        if self.reason != "kept":
            return False
        self._parts.append(x.copy())
        self._n += len(x)
        self.spans.append((pos, len(x)))
        return True

    def _calibration_chunk(self, x: np.ndarray, pos: int, target: int | None) -> bool:
        self._cal_left = max(0, self._cal_left - len(x))
        if target is not None and self.track_id is None:
            self.track_id = target
        kept = self.gate.voiced(x)
        if kept:
            self._parts.append(x.copy())
            self._n += len(x)
            self.spans.append((pos, len(x)))
        if not self.calibrating:  # window closed
            if self._n >= round(CALIBRATE_MIN_S * self.sr):
                self.calibrated, self._held, self.source = True, None, "calibrated"
            else:  # too little speech: drop it, keep the previous clip, collect automatically again
                held, oheld, track = self._held, self._oheld, self.track_id
                held_from = getattr(self, "_held_from", ("auto", None, False))
                self.reset()
                self._held, self._oheld, self.track_id = held, oheld, track
                self.calibration_failed = True
                if held_from[0] == "profile" and held is not None:  # a failed window keeps the profile
                    self.enroll_clip(held, oheld)
                    self.source, self.profile, self._other_fixed = held_from
                    self.calibration_failed = True
        return kept
