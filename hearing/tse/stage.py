"""Focus HQ as a streaming ``Stage``: any TSE candidate over sliding windows, delayed and crossfaded.

``tse(candidate="dolphin", window_s=1.0, hop_s=0.5)`` takes 10 ms chunks and returns the target
estimate ``latency_ms`` later. Windows of ``window_s`` start every ``hop_s``; window ``k`` covers
input samples ``[k*hop, k*hop + window)`` and is extracted once that much input has arrived. The
outputs are overlap-added with the bake-off's raised-cosine weights
(:func:`hearing.tse.windowed.crossfade_window`), normalised by the summed weights, so chunked
output equals :class:`~hearing.tse.windowed.SlidingWindowExtractor` on the whole file (the
streaming-equivalence test) and an identity candidate reproduces its input exactly.

Delay: ``window`` inline (``threaded=false``: offline benches and tests; the call that completes
a window blocks for its compute) or ``window + hop`` threaded (default): each window is extracted
on a worker thread and has one hop of real time to finish. A window that is late, or skipped
because the worker is still busy (an overrun), because the target has no lip crops
(``needs`` lips) or no enrollment yet (``needs`` enroll), or because the candidate failed, is
replaced by the delayed input (the mix) for the samples it covers, with a 10 ms ramp, so the
timeline never stalls. :meth:`health` reports that, and the tier ladder
(:mod:`hearing.fusion.tiers`) drops to the next tier.

Side inputs, all taken from ``ctx``:

* lips: the session's :class:`~hearing.tse.lips.LipFeed` (``ctx.telemetry.extras["lip_feed"]``),
  frames of ``ctx.target.track_id`` placed on the audio clock by capture timestamp and resampled
  to 25 fps on the window grid (nearest frame; a window needs ``min_lips`` of its frames within
  ``LIP_GAP_S`` of a real one).
* ``p_speaking``: per video frame, from the same feed entries.
* enrollment: :class:`~hearing.tse.enroll.AutoEnroller` fed with this stage's own input while
  the target alone speaks, or the session's shared one (``ctx.telemetry.extras["enroller"]``,
  fed by the session and never reset by this stage, so a new ladder or a mode round trip keeps
  the clip, T-078); candidates whose ``needs`` include ``other`` also get its negative
  enrollment (the other people talking alone, ``None`` until collected) as ``other=``.
* ``asd`` (T-069): candidates whose ``needs`` include ``asd`` get ``asd=``, the target track's
  ``p_speaking`` from ``ctx.tracks`` per 10 ms of the window (what the enroller gates on; NaN where
  no target track is visible). It needs no lip feed and never skips a window.

Pause handling (T-069, wired through the spec by T-070): ``pause_floor``, ``pause_p``,
``pause_action`` and ``pause_max_s`` are passed to the ``enrolled`` candidate
(:class:`~hearing.tse.enrolled.EnrolledCandidate`), e.g. the "strict" preset
``tse(candidate="enrolled", pause_floor=0.25, pause_p=0.5, pause_action="duck")`` (wrong-target
audio -50 % on M/F for a few more muted windows). Unset (default) they keep the candidate's
defaults (off); any other candidate rejects them with ``ValueError``.

Output: the full-strength target. With ``residual=true`` (default) the stage also writes
``ctx.telemetry.extras["residual"] = delayed input - target`` so a following ``mixer`` plays
``target + (1 - s) * residual`` (the slider). Stats land in ``ctx.telemetry.extras["tse"]``.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import numpy as np

from hearing import device as hdevice
from hearing import registry
from hearing.fusion import stream_vad
from hearing.fusion.mixer import CONFIDENCE_KEY, OVERLAP_KEY, RESIDUAL_KEY
from hearing.tse.base import LIPS_FPS, SR, conform_output, get_candidate
from hearing.tse.enroll import ENROLL_S, ENROLLER_KEY, AutoEnroller
from hearing.tse.enrolled import loudest_enabled
from hearing.tse.gpu import TORCH_LOCK
from hearing.tse.lips import LipFrame, feed_of
from hearing.tse.windowed import crossfade_window, frame_span
from hearing.types import AudioChunk, StageContext

log = logging.getLogger(__name__)

# ``candidate="auto"``: the first of these whose weights and runtime are present, else identity.
# Order = the Focus HQ pick (T-019 bench on av_tiny, see docs/decisions/T-019.md).
AUTO_ORDER = ("av_mossformer2", "dolphin")
_MODULES = {  # candidate -> module with ``available()``
    "dolphin": "hearing.tse.dolphin",
    "av_mossformer2": "hearing.tse.av_mossformer",
    "enrolled": "hearing.tse.enrolled",
    "blind_assoc": "hearing.tse.blind",
}
MOCKS = ("identity", "oracle")
def _round(x, nd=3):
    return None if x is None else round(float(x), nd)


PAUSE_KWARGS = ("pause_floor", "pause_p", "pause_action", "pause_max_s")  # enrolled candidate only
RAMP_MS = 10.0  # fade between TSE output and the fallback mix
LIP_GAP_S = 0.08  # a window frame counts as covered when a real crop is within 2 video frames
MIN_LIPS = 0.5  # covered share of a window's frames needed to run an AV candidate
HEALTH_WINDOWS = 4  # windows the overrun/late count looks back over (2 s at the default hop)
SOFT_BAD = 2  # overrun/late windows among the last HEALTH_WINDOWS that widen the hop
SOFT_FAIL = 3  # ... that make the stage unhealthy at the widest hop (hop = window: no wider step left)
RECOVER_WINDOWS = 4  # N: consecutive clean windows that make an unhealthy stage healthy again
HOP_STEPS = (1.0, 1.5, 2.0)  # adaptive hop ladder, multiples of the configured hop (capped at the window)
HOP_DOWN_AFTER = 10  # consecutive clean windows before the hop steps back down
HOP_DOWN_FIT = 0.8  # ... and only if the recent compute p95 fits in this share of the smaller hop
COMPUTE_WINDOWS = 16  # windows the compute p50/p95 and utilisation are taken over
STATS_KEEP = 64


def _target_p(ctx: StageContext) -> float:
    """``p_speaking`` of the target track in ``ctx``, NaN without a visible target track."""
    tid = ctx.target.track_id
    if tid is None:
        return math.nan
    return next((float(t.p_speaking) for t in ctx.tracks if t.track_id == tid), math.nan)


def candidate_available(name: str) -> bool:
    """Weights and runtime of ``name`` present (mocks always are)."""
    if name in MOCKS:
        return True
    module = _MODULES.get(name)
    if module is None:
        return False
    try:
        return bool(importlib.import_module(module).available())
    except ImportError:
        return False


def resolve_candidate(name: str) -> str:
    if name != "auto":
        return name
    return next((c for c in AUTO_ORDER if candidate_available(c)), "identity")


class _Window:
    __slots__ = ("cold", "covered", "future", "hop", "k", "ms", "reason", "start", "status", "submitted")

    def __init__(self, k: int, status: str, reason: str = ""):
        self.k = k
        self.start = 0  # absolute input sample of the window's first sample
        self.hop = 0  # the hop the window was planned at (its crossfade weights)
        self.cold = False  # the first computed window (kernels, enrollment embeddings): not timed
        self.status = status  # pending | ok | skip
        self.reason = reason
        self.future: Future | None = None
        self.submitted = 0.0
        self.ms = math.nan
        self.covered = math.nan


# T-419: the enrolled tier's balanced window pair (config/tiers.yaml, latency preset "balanced") from the env,
# e.g. HEARING_TSE_WINDOW_S=0.5 HEARING_TSE_HOP_S=0.25 plays 0.75 s behind instead of 1.5 s. Unset = 1.0 / 0.5.
# Only an enrolled spec at exactly the balanced pair is rewritten: the speed/accuracy presets and the GPU box's
# own windows (remote.enrolled, 0.5 / 0.125) keep theirs. HEARING_TSE_WINDOW_S alone sets the hop to window / 2.
WINDOW_ENV, HOP_ENV = "HEARING_TSE_WINDOW_S", "HEARING_TSE_HOP_S"
BALANCED_WINDOW = (1.0, 0.5)


def balanced_window() -> tuple[float, float]:
    """``(window_s, hop_s)`` the balanced enrolled tier runs at: the env knobs, else :data:`BALANCED_WINDOW`."""
    raw_w, raw_h = os.environ.get(WINDOW_ENV, "").strip(), os.environ.get(HOP_ENV, "").strip()
    try:
        window = float(raw_w) if raw_w else BALANCED_WINDOW[0]
        hop = float(raw_h) if raw_h else (window / 2 if raw_w else BALANCED_WINDOW[1])
    except ValueError as e:
        raise ValueError(f"{WINDOW_ENV}={raw_w!r} / {HOP_ENV}={raw_h!r}: need numbers in seconds") from e
    if window <= 0 or not 0 < hop <= window:
        raise ValueError(f"{WINDOW_ENV}={window:g} / {HOP_ENV}={hop:g}: need 0 < hop <= window")
    return window, hop


def tse_window() -> dict:
    """``/health``'s ``tse_window``: the balanced enrolled tier's window, hop and threaded delay (window + hop),
    the "expected delay" the pages show, and whether the env knobs changed it from 1.0 / 0.5."""
    window, hop = balanced_window()
    return {"window_s": window, "hop_s": hop, "delay_ms": round(1000.0 * (window + hop)),
            "env": (window, hop) != BALANCED_WINDOW}


def env_window(window_s: float, hop_s: float) -> tuple[float, float]:
    """The balanced pair (1.0 / 0.5) becomes :func:`balanced_window`; any other pair is kept."""
    if math.isclose(window_s, BALANCED_WINDOW[0]) and math.isclose(hop_s, BALANCED_WINDOW[1]):
        return balanced_window()
    return window_s, hop_s


@registry.register("tse")
class FocusHQ:
    """Sliding-window target-speaker extraction with streaming state. See the module doc."""

    sample_rate = SR

    def __init__(
        self,
        candidate: str = "auto",
        window_s: float = 1.0,
        hop_s: float | None = None,
        threaded: bool = True,
        require_gpu: bool = False,
        residual: bool = True,
        min_lips: float = MIN_LIPS,
        enroll_s: float = ENROLL_S,
        pause_floor: float | None = None,
        pause_p: float | None = None,
        pause_action: str | None = None,
        pause_max_s: float | None = None,
        adapt_hop: bool = True,
        contrast: float | None = None,
        _candidate=None,  # tests inject a TSECandidate instance
    ):
        hop_s = window_s / 2 if hop_s is None else hop_s
        if candidate == "enrolled":
            window_s, hop_s = env_window(window_s, hop_s)
        if window_s <= 0 or not 0 < hop_s <= window_s:
            raise ValueError(f"need 0 < hop_s <= window_s, got window {window_s}, hop {hop_s}")
        if not 0.0 <= min_lips <= 1.0:
            raise ValueError(f"min_lips must be in [0, 1], got {min_lips}")
        self.requested = candidate
        cand_kwargs = {k: v for k, v in zip(PAUSE_KWARGS, (pause_floor, pause_p, pause_action, pause_max_s),
                                            strict=True) if v is not None}
        if contrast is not None:  # T-178: enrolled pick floor (cos(target) - cos(other), or LDA units)
            cand_kwargs["contrast"] = float(contrast)
        if _candidate is not None:
            self.cand = _candidate
        else:
            resolved = resolve_candidate(candidate)
            if cand_kwargs and resolved != "enrolled":
                raise ValueError(f"{', '.join(cand_kwargs)} only apply to candidate 'enrolled', not {resolved!r}")
            self.cand = get_candidate(resolved, **cand_kwargs)
        self.candidate_kwargs = cand_kwargs
        self.candidate = self.cand.name
        self.mock = self.candidate in MOCKS
        self.window = round(window_s * SR)
        self.threaded = bool(threaded)
        base = round(hop_s * SR)
        # Adaptive hop (threaded only): widened on overruns so a loaded machine keeps the tier.
        steps = HOP_STEPS if (self.threaded and adapt_hop) else HOP_STEPS[:1]
        self.hops = tuple(dict.fromkeys(min(self.window, round(base * f)) for f in steps))
        self.hop = base
        self.hop_pinned = False  # pin_hop(): no more widening/narrowing (kept across reset)
        self.window_s, self.hop_s = self.window / SR, self.hop / SR
        self.require_gpu = bool(require_gpu)
        self.residual = bool(residual)
        self.min_lips = float(min_lips)
        self._set_hop(0)
        self.name = f"tse({self.candidate}@{self.window_s:g}s)"
        self._weights = {h: crossfade_window(self.window, h) for h in self.hops}
        self._ramp_step = 1.0 / max(1.0, RAMP_MS * SR / 1000.0)
        self.enroller = AutoEnroller(enroll_s, SR) if "enroll" in self.cand.needs else None
        self._own_enroller = self.enroller
        self._shared_enroller: AutoEnroller | None = None  # the session's, once adopted (never reset here)
        self._want_asd = "asd" in self.cand.needs
        self._pool: ThreadPoolExecutor | None = None
        self._load: Future | None = None
        self.loaded = False
        self.error: str | None = None
        self.load_ms = math.nan
        from hearing.tse.enrolled import warm_enabled

        # T-339 (HEARING_WARM_ENROLLED=1): submit the load with the first chunk, not when the first window
        # is due, so window 0 is computed instead of skipped as ``loading`` (4 clean windows to outlive)
        self.eager_load = self.threaded and warm_enabled()

        # T-337 (HEARING_OVERLAP_FLOOR=1 at construction): record the separator's streams of each window
        # and run the per-stream VAD on them, so the mixer can tell a gated overlap window from solo speech
        self._vad = stream_vad.StreamVad() if stream_vad.enabled() else None
        self._streams = None
        sep = getattr(self.cand, "separator", None)
        if self._vad is not None and callable(getattr(sep, "separate", None)):
            real = sep.separate

            def recording(*a, **k):
                self._streams = real(*a, **k)
                return self._streams

            sep.separate = recording
        self.reset()

    # ------------------------------------------------------------ lifecycle

    def _set_hop(self, i: int) -> None:
        self._hop_i = i
        self.hop = self.hops[i]
        self.hop_s = self.hop / SR
        self.delay = self.window + (self.hop if self.threaded else 0)
        self.latency_ms = 1000.0 * self.delay / SR

    def pin_hop(self) -> None:
        """Keep the current hop from now on (T-135: the judge profile pins it once the target is
        isolated, so no later HOP_STEPS change plays as a dip). At the pinned hop
        :data:`SOFT_FAIL` overruns make the stage unhealthy, as at the widest hop."""
        if not self.hop_pinned:
            self.hop_pinned = True
            self._version += 1

    def _change_hop(self, step: int) -> None:
        """Move one step on the hop ladder. The delay (window + hop) moves with it: the output
        dips over :data:`RAMP_MS`, then stalls (longer delay) or skips (shorter) the difference."""
        old = self.delay
        self._set_hop(self._hop_i + step)
        self._jump += self.delay - old
        self.hop_changes += 1
        self._hop_clean = 0
        self._outcomes.clear()
        log.info("tse %s hop -> %.3f s (delay %.0f ms)", self.candidate, self.hop_s, self.latency_ms)

    @property
    def device(self) -> str:
        return getattr(self.cand, "device", None) or hdevice.pick()

    def load(self) -> None:
        """Load the candidate now, on this thread (benches call this before timing)."""
        if self.loaded or self.error is not None:
            return
        if self.require_gpu and not self.mock and hdevice.pick() == "cpu":
            self.error = "no GPU: Focus HQ needs CUDA/MPS locally or the remote GPU path"
            return
        t0 = time.perf_counter()
        try:
            with TORCH_LOCK:
                self.cand.load()
        except Exception as e:  # noqa: BLE001  (no weights, runtime error: the tier is unavailable)
            self.error = f"load: {type(e).__name__}: {e}"
            log.warning("tse %s failed to load: %s", self.candidate, self.error)
            return
        self.load_ms = 1000.0 * (time.perf_counter() - t0)
        self.loaded = True

    def _ensure_loaded(self) -> bool:
        if self.loaded or self.error is not None:
            return self.loaded
        if not self.threaded:
            self.load()
            return self.loaded
        if self._load is None:
            self._load = self._executor().submit(self.load)  # first job: model load off the audio path
        return False

    def _executor(self) -> ThreadPoolExecutor:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(1, thread_name_prefix=f"tse-{self.candidate}")
        return self._pool

    def reset(self) -> None:
        """Fresh streaming state; the loaded model and the load job are kept."""
        self._base = 0  # absolute sample index of _in[0] / _acc[0] / _wsum[0]
        self._in = np.zeros(0, np.float32)
        self._asd = np.zeros(0, np.float32)  # target p_speaking per input sample (``asd`` candidates)
        self._acc = np.zeros(0, np.float64)
        self._wsum = np.zeros(0, np.float64)
        self._cacc = np.zeros(0, np.float64)  # T-134: window confidence, crossfaded like _acc
        self._cwsum = np.zeros(0, np.float64)
        self._conf: float | None = None  # confidence of the last emitted chunk (None = no TSE playing)
        self._oacc = np.zeros(0, np.float64)  # T-337: window overlap (0/1), crossfaded like _cacc
        self._owsum = np.zeros(0, np.float64)
        self._ovl: float | None = None  # overlap share of the last emitted chunk (None = not measured)
        if self._vad is not None:
            self._vad.reset()
        self._n_in = 0
        self._next_k = 0
        self._set_hop(0)
        self._next_start = 0  # input sample where the next window starts
        self._out = -self.delay  # absolute position of the next output sample (negative: warm-up)
        self._stall = 0  # zeros still to emit before _out resumes (a longer delay)
        self._jump = 0  # pending change of delay in samples, applied once the output has dipped
        self._env = 1.0  # dip envelope around a delay jump
        self._windows: dict[int, _Window] = {}
        self._recent: deque[_Window] = deque(maxlen=STATS_KEEP)
        self._unsettled: deque[_Window] = deque()  # planned windows not yet counted by health
        self._outcomes: deque[str] = deque(maxlen=HEALTH_WINDOWS)  # "" = clean, else the reason
        self._ok = False
        self._bad_reason = "warming up"
        self._clean_run = self._hop_clean = 0
        self._warm = False  # a window has been computed since the reset
        self.hop_changes = 0
        self._version = 0
        self._health_key = None
        self._health_cache: dict = {}
        self._lips: deque[tuple[float, LipFrame]] = deque()  # (sample position, frame)
        self._cursor = 0
        self._feed_id = None
        self._anchor: tuple[float, int] | None = None  # (capture_ts, sample index at it)
        self._mix_gain = 0.0  # 1 = fallback mix, 0 = TSE output (ramped)
        self.overruns = self.late = 0
        self._prepared: tuple | None = None  # key of the enrollment clip the candidate has prepared
        self._prepared_clips: tuple | None = None  # (enroll, other) of that key
        self._preparing: tuple | None = None  # (key, future) of the preparation job running
        self._preloaded: list | None = None  # the enroller's ``preload`` list already queued (T-136)
        if self.enroller is not None and self.enroller is not self._shared_enroller:
            self.enroller.reset()  # a shared (session) enroller outlives this stage's state

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    # ------------------------------------------------------------ side inputs

    def _take_lips(self, chunk: AudioChunk, ctx: StageContext) -> None:
        """Place new feed frames on the input sample grid via their capture timestamps."""
        ts = float(chunk.capture_ts)
        if self._anchor is None or ts != self._anchor[0]:
            self._anchor = (ts, self._n_in)  # hops of one client frame share its capture_ts
        feed = feed_of(ctx)
        if feed is None:
            return
        if id(feed) != self._feed_id:
            self._feed_id, self._cursor = id(feed), 0
        a_ts, a_pos = self._anchor
        for f in feed.since(self._cursor):
            self._cursor = f.index + 1
            self._lips.append((a_pos + (f.ts - a_ts) * SR, f))
        horizon = self._next_start - 2 * self.window
        while self._lips and self._lips[0][0] < horizon:
            self._lips.popleft()

    def _window_lips(self, start: int, track_id: int | None) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """``(lips T x H x W, p_speaking T, covered share)`` of the target on the window's grid."""
        f0, nf = frame_span(start, self.window, SR, LIPS_FPS)
        frames = [(pos, f) for pos, f in self._lips if f.track_id == track_id] if track_id is not None else []
        if not frames or nf <= 0:
            return None, None, 0.0
        pos = np.array([p for p, _ in frames])
        order = np.argsort(pos, kind="stable")
        pos = pos[order]
        grid = (f0 + np.arange(nf)) / LIPS_FPS * SR
        right = np.clip(np.searchsorted(pos, grid), 0, len(pos) - 1)
        left = np.clip(right - 1, 0, len(pos) - 1)
        pick = np.where(np.abs(pos[left] - grid) <= np.abs(pos[right] - grid), left, right)
        covered = float(np.mean(np.abs(pos[pick] - grid) <= LIP_GAP_S * SR))
        chosen = [frames[order[i]][1] for i in pick]
        lips = np.stack([f.crop for f in chosen])
        p = np.array([f.p_speaking for f in chosen], np.float32)
        return lips, p, covered

    # ------------------------------------------------------------ windows

    def _submit_due(self, ctx: StageContext) -> None:
        while self._next_start + self.window <= self._n_in:
            k = self._next_k
            self._next_k += 1
            self._windows[k] = w = self._plan(k, ctx)
            w.start, w.hop = self._next_start, self.hop
            self._next_start += self.hop
            if w.status == "pending":
                self._run(w, ctx)
            self._recent.append(w)
            self._unsettled.append(w)

    def _plan(self, k: int, ctx: StageContext) -> _Window:
        if self.error is not None:
            return _Window(k, "skip", "error")
        if not self._ensure_loaded():
            return _Window(k, "skip", "loading")
        busy = [w for w in self._windows.values() if w.status == "pending"]
        if self.threaded and busy:
            if any(w.cold for w in busy):  # the cold first window: not the machine's steady speed
                return _Window(k, "skip", "warming up")
            self.overruns += 1
            return _Window(k, "skip", "overrun")
        return _Window(k, "pending")

    def _run(self, w: _Window, ctx: StageContext) -> None:
        start = w.start
        mix = self._in[start - self._base : start - self._base + self.window].copy()
        needs = self.cand.needs
        lips = p = None
        if "lips" in needs or "p_speaking" in needs:
            lips, p, w.covered = self._window_lips(start, ctx.target.track_id)
            if "lips" in needs and (lips is None or w.covered < self.min_lips):
                w.status, w.reason = "skip", "no lips"
                return
            if "p_speaking" in needs and p is None:
                w.status, w.reason = "skip", "no p_speaking"
                return
        enroll = None
        if self._loudest_on():  # HEARING_PICK=loudest: no voice profile needed, the loudest separated voice plays
            needs = ()
        if "enroll" in needs:
            enroll = self.enroller.clip() if self.enroller is not None else None
            if enroll is None:
                w.status, w.reason = "skip", "enrolling"
                return
        other = self.enroller.other_clip() if "other" in needs and self.enroller is not None else None
        if enroll is not None and not self._enrollment_prepared(enroll, other):
            if self._prepared_clips is None:
                w.status, w.reason = "skip", "enrolling"
                return
            # T-120: a new clip replacing a prepared one (Name Call re-target or a longer clip) keeps
            # the old one playing until its embeddings are ready, instead of dropping out of the tier
            enroll, other = self._prepared_clips
        asd = None
        if self._want_asd:
            asd = self._asd[start - self._base : start - self._base + self.window][:: SR // 100].copy()
        args = (mix, lips if "lips" in needs else None, p if "p_speaking" in needs else None, enroll, other, asd)
        w.submitted = time.perf_counter()
        w.cold, self._warm = not self._warm, True
        if self.threaded:
            w.future = self._executor().submit(self._extract, *args)
            return
        self._finish(w, *self._extract(*args))

    def _loudest_on(self) -> bool:
        """HEARING_PICK=loudest with a candidate that supports it (the enrolled candidate)."""
        return loudest_enabled() and callable(getattr(self.cand, "_loudest", None))

    def _enrollment_prepared(self, enroll: np.ndarray, other: np.ndarray | None) -> bool:
        """True once the candidate prepared ``enroll`` (``prepare()``: the enrolled candidate's
        embeddings, ~1 s on the Mac). Threaded stages prepare a new clip in its own job instead of
        inside the first window, which overran the hop and kept the tier unhealthy for several
        windows (T-088); windows due meanwhile are skipped as ``enrolling``."""
        prepare = getattr(self.cand, "prepare", None)
        if not self.threaded or not callable(prepare):
            return True
        key = (len(enroll), hash(enroll[::SR // 100].tobytes()))  # cheap; the candidate keys its cache exactly
        if key == self._prepared:
            self._prepared_clips = (enroll, other)
            return True
        if self._preparing is not None:
            running, job = self._preparing
            if not job.done():
                return False
            self._preparing = None
            if running == key:
                self._prepared = key
                self._prepared_clips = (enroll, other)
                return True
        self._preparing = (key, self._executor().submit(self._prepare, prepare, enroll, other))
        return False

    def _preload(self, pairs) -> None:
        for enroll, other in pairs:
            self._prepare(self.cand.prepare, enroll, other)

    def _prepare(self, prepare, enroll, other) -> None:
        try:
            with TORCH_LOCK:
                prepare(enroll, other, SR)
        except Exception as e:  # noqa: BLE001  (the window will compute it and report any error)
            log.debug("tse %s: preparing the enrollment failed: %s", self.candidate, e)

    def _extract(self, mix, lips, p, enroll, other=None, asd=None) -> tuple:
        """``(target or None, error, compute ms, candidate confidence or None, overlap or None)``; the time
        includes waiting for the GPU lock. ``overlap`` (T-337, knob on only): both separated streams active."""
        t0 = time.perf_counter()
        extra = {"other": other} if "other" in self.cand.needs else {}
        if self._want_asd:
            extra["asd"] = asd
        try:
            with TORCH_LOCK:
                y = self.cand.extract(mix, SR, lips=lips, lips_fps=LIPS_FPS, p_speaking=p, enroll=enroll, **extra)
                conf = getattr(self.cand, "confidence", None)  # T-134, read under the lock with its window
                streams, self._streams = self._streams, None
            y = conform_output(y, len(mix), self.candidate)
            overlap = None
            if self._vad is not None and streams is not None and len(streams) == 2:
                overlap = float(self._vad.window(mix, streams)["overlap"])
        except Exception as e:  # noqa: BLE001  (a model error skips one window, never the stream)
            return None, f"{type(e).__name__}: {e}", 1000.0 * (time.perf_counter() - t0), None
        return y, "", 1000.0 * (time.perf_counter() - t0), conf, overlap

    def _finish(self, w: _Window, y: np.ndarray | None, error: str, ms: float, conf: float | None = None,
                overlap: float | None = None) -> None:
        w.ms = ms
        if y is None:
            w.status, w.reason = "skip", error
            log.debug("tse %s window %d failed: %s", self.candidate, w.k, error)
            return
        start = w.start - self._base
        skip = max(0, -start)  # samples already played from the fallback: the window was late
        if skip:
            w.reason = "late"
            self.late += 1
        if skip < self.window:
            a = start + skip
            weights = self._weights[w.hop]
            self._acc[a : start + self.window] += (weights * y)[skip:]
            self._wsum[a : start + self.window] += weights[skip:]
            if conf is not None and math.isfinite(conf):
                self._cacc[a : start + self.window] += (weights * float(conf))[skip:]
                self._cwsum[a : start + self.window] += weights[skip:]
            if overlap is not None:
                self._oacc[a : start + self.window] += (weights * overlap)[skip:]
                self._owsum[a : start + self.window] += weights[skip:]
        w.status = "ok"

    def _collect(self) -> None:
        for w in list(self._windows.values()):
            if w.status == "pending" and w.future is not None and w.future.done():
                self._finish(w, *w.future.result())

    # ------------------------------------------------------------ Stage

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        x = np.asarray(chunk.samples, dtype=np.float32)
        n = len(x)
        if self.eager_load:
            self._ensure_loaded()
        self._take_lips(chunk, ctx)
        if self.enroller is not None:
            shared = ctx.telemetry.extras.get(ENROLLER_KEY)
            if not (isinstance(shared, AutoEnroller) and shared.need == self._own_enroller.need
                    and shared.sr == SR):
                shared = None
            if shared is not None:
                self.enroller = self._shared_enroller = shared  # the session feeds it once per hop
            else:
                if self._shared_enroller is not None and self.enroller is self._shared_enroller:
                    self.enroller = self._own_enroller
                self._shared_enroller = None
                self.enroller.update(x, ctx)
        self._in = np.concatenate([self._in, x])
        if self._want_asd:
            self._asd = np.concatenate([self._asd, np.full(n, _target_p(ctx), np.float32)])
        self._acc = np.concatenate([self._acc, np.zeros(n)])
        self._wsum = np.concatenate([self._wsum, np.zeros(n)])
        self._cacc = np.concatenate([self._cacc, np.zeros(n)])
        self._cwsum = np.concatenate([self._cwsum, np.zeros(n)])
        self._oacc = np.concatenate([self._oacc, np.zeros(n)])
        self._owsum = np.concatenate([self._owsum, np.zeros(n)])
        self._n_in += n
        self._collect()
        if self.threaded and self.loaded and self.enroller is not None and callable(getattr(self.cand, "prepare", None)):
            clip = self.enroller.clip()
            if clip is not None:  # a new clip starts preparing now, not at the next window
                self._enrollment_prepared(clip, self.enroller.other_clip() if "other" in self.cand.needs else None)
            pre = getattr(self.enroller, "preload", None)
            if pre and pre is not self._preloaded:  # T-136: every saved profile, queued behind the clip above
                self._preloaded = pre
                with_other = "other" in self.cand.needs
                self._executor().submit(self._preload, [(t, o if with_other else None) for t, o in pre])
        self._submit_due(ctx)
        self._settle()
        out, mix = self._emit_chunk(n)
        self._trim(self._out)
        if self.residual and self._loudest_on():  # loudest voice only: the room / other voice never comes back
            ctx.telemetry.extras[RESIDUAL_KEY] = np.zeros_like(out, dtype=np.float32)
            ctx.telemetry.extras[CONFIDENCE_KEY] = 1.0
        elif self.residual:
            ctx.telemetry.extras[RESIDUAL_KEY] = (mix - out).astype(np.float32)
            if self._conf is not None:
                ctx.telemetry.extras[CONFIDENCE_KEY] = self._conf
            if self._ovl is not None:
                ctx.telemetry.extras[OVERLAP_KEY] = self._ovl
        ctx.telemetry.extras["tse"] = {**self.health(), "confidence": self._conf}
        if self._vad is not None:
            ctx.telemetry.extras["tse"]["overlap"] = self._ovl
        return replace(chunk, samples=out)

    def _emit_chunk(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """``n`` output samples: stalled zeros first (a delay that grew), then the stream from
        ``_out``, under the dip envelope that hides a delay jump."""
        z = min(n, self._stall)
        self._stall -= z
        m = n - z
        out, mix = self._emit(self._out, m)
        self._out += m
        if self._jump or self._env < 1.0:
            env = self._follow(0.0 if self._jump else 1.0, m)
            out, mix = (out * env).astype(np.float32), (mix * env).astype(np.float32)
            if self._jump and self._env <= 0.0:  # silent now: jump
                if self._jump > 0:
                    self._stall += self._jump
                else:
                    self._out -= self._jump
                self._jump = 0
        if z:
            out = np.concatenate([np.zeros(z, np.float32), out])
            mix = np.concatenate([np.zeros(z, np.float32), mix])
        return out, mix

    def _follow(self, goal: float, m: int) -> np.ndarray:
        """Envelope ramping from ``_env`` toward ``goal`` over :data:`RAMP_MS`, carried across chunks."""
        s = self._ramp_step if goal > self._env else -self._ramp_step
        env = np.clip(self._env + s * np.arange(1, m + 1), 0.0, 1.0)
        env = np.minimum(env, goal) if s > 0 else np.maximum(env, goal)
        if m:
            self._env = float(env[-1])
        return env

    def _emit(self, o0: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Output and delayed input for absolute positions ``[o0, o0 + n)`` (negative: warm-up)."""
        pos = np.arange(o0, o0 + n)
        live = pos >= 0
        idx = np.clip(pos - self._base, 0, None)
        mix = np.where(live, self._in[idx] if len(self._in) else 0.0, 0.0).astype(np.float32)
        tse = np.zeros(n, np.float64)
        wsum = self._wsum[idx] if len(self._wsum) else np.zeros(n)
        ok = live & (wsum > 0) & self._final(pos)
        tse[ok] = self._acc[idx[ok]] / wsum[ok]
        cw = self._cwsum[idx] if len(self._cwsum) else np.zeros(n)
        rated = ok & (cw > 0)
        self._conf = round(float(np.mean(self._cacc[idx[rated]] / cw[rated])), 3) if rated.any() else None
        ow = self._owsum[idx] if len(self._owsum) else np.zeros(n)
        seen = ok & (ow > 0)
        self._ovl = round(float(np.mean(self._oacc[idx[seen]] / ow[seen])), 3) if seen.any() else None
        goal = np.where(ok | ~live, 0.0, 1.0)  # warm-up is silence either way
        gain = self._ramp(goal)
        out = gain * mix + (1.0 - gain) * tse
        return out.astype(np.float32), mix

    def _final(self, pos: np.ndarray) -> np.ndarray:
        """True where every window covering the position is resolved (ok or skipped)."""
        out = pos < self._next_start  # every window that will ever cover pos is planned
        for w in self._windows.values():
            if w.status == "pending":
                out &= ~((w.start <= pos) & (pos < w.start + self.window))
        return out

    def _ramp(self, goal: np.ndarray) -> np.ndarray:
        """Slope-limited follower of ``goal`` (0/1 per sample), carried across chunks."""
        out = np.empty(len(goal))
        g = self._mix_gain
        i = 0
        step = self._ramp_step
        while i < len(goal):
            j = i + 1
            while j < len(goal) and goal[j] == goal[i]:
                j += 1
            target = goal[i]
            if g == target:
                out[i:j] = g
            else:
                s = step if target > g else -step
                ramp = g + s * np.arange(1, j - i + 1)
                ramp = np.minimum(ramp, target) if s > 0 else np.maximum(ramp, target)
                out[i:j] = ramp
                g = float(ramp[-1])
            i = j
        self._mix_gain = g
        return out

    def _trim(self, emitted: int) -> None:
        keep = min(emitted, self._next_start)
        cut = keep - self._base
        if cut <= 0:
            return
        self._in, self._acc, self._wsum = self._in[cut:], self._acc[cut:], self._wsum[cut:]
        self._cacc, self._cwsum = self._cacc[cut:], self._cwsum[cut:]
        self._oacc, self._owsum = self._oacc[cut:], self._owsum[cut:]
        if self._want_asd:
            self._asd = self._asd[cut:]
        self._base = keep
        # Windows whose samples are all played are only kept in _recent for the stats; a pending
        # one stays until it resolves (late), so health counts it.
        for k in [k for k, w in self._windows.items()
                  if w.start + self.window <= emitted and w.status != "pending"]:
            del self._windows[k]

    # ------------------------------------------------------------ health

    def _settle(self) -> None:
        """Count resolved windows in plan order: the health verdict and the adaptive hop."""
        while self._unsettled and self._unsettled[0].status != "pending":
            self._account(self._unsettled.popleft())

    def _account(self, w: _Window) -> None:
        self._version += 1
        soft = w.reason in ("overrun", "late") and not w.cold  # the machine is too slow for this hop
        clean = w.status == "ok" and (not w.reason or w.cold)
        self._outcomes.append("" if clean else w.reason)
        if clean:
            self._clean_run += 1
            self._hop_clean += 1
        else:
            self._clean_run = self._hop_clean = 0
            if not soft:  # no lips / enrolling / loading / a model error: not usable now
                self._ok, self._bad_reason = False, w.reason or "skipped"
            elif not self._ok:
                self._bad_reason = w.reason
        n_soft = sum(1 for o in self._outcomes if o in ("overrun", "late"))
        widest = self.hop_pinned or self._hop_i + 1 == len(self.hops)
        if soft and not widest and n_soft >= SOFT_BAD:
            self._change_hop(+1)  # keeps the verdict: the wider hop gives the window time
        elif soft and widest and n_soft >= SOFT_FAIL:
            self._ok, self._bad_reason = False, w.reason
        # fresh state, or the enrollment arrived since its skips (T-088): no failure to outlive
        need = 1 if self._bad_reason in ("warming up", "enrolling") else RECOVER_WINDOWS
        if not self._ok and self._clean_run >= need:
            self._ok, self._bad_reason = True, ""
        if self._hop_i > 0 and not self.hop_pinned and self._hop_clean >= HOP_DOWN_AFTER:
            p95 = self._compute()[1]
            if p95 is not None and p95 < HOP_DOWN_FIT * 1000.0 * self.hops[self._hop_i - 1] / SR:
                self._change_hop(-1)

    def _compute(self) -> tuple[float | None, float | None]:
        """Compute p50/p95 in ms over the last :data:`COMPUTE_WINDOWS` timed windows."""
        times = [w.ms for w in list(self._recent)[-COMPUTE_WINDOWS:] if math.isfinite(w.ms) and not w.cold]
        if not times:
            return None, None
        p50, p95 = np.percentile(times, [50, 95]).tolist()
        return p50, p95

    def health(self) -> dict:
        """JSON-safe status for telemetry and the tier ladder (cached until a window resolves).

        ``ok``: loaded, no error, enrolled/lips present, and not overrunning. One overrun or late
        window among the last :data:`HEALTH_WINDOWS` is tolerated (its neighbours cover it);
        :data:`SOFT_BAD` of them widen the hop (0.5 -> 0.75 -> 1.0 s at the default 1 s window);
        at the widest hop :data:`SOFT_FAIL` of them make the stage unhealthy. The first computed
        window after a reset is cold (kernels, enrollment embeddings): windows it blocks read
        ``warming up`` and it is left out of the compute stats. Any other skipped window (``no
        lips``, ``enrolling``, ``loading``, a model error) makes it unhealthy at once. An unhealthy
        stage recovers after :data:`RECOVER_WINDOWS` consecutive clean windows (a fresh one, with
        no failure yet, or one that was only waiting for the enrollment, after its first clean window,
T-088). ``reason`` says
        why not (``loading``, ``no lips``, ``enrolling``, ``overrun``, ``late``, ``error``,
        ``warming up``). ``utilisation`` = compute p50 / current hop.
        """
        key = (self._version, self._next_k, self.loaded, self.error, self._hop_i)
        if key == self._health_key:
            return self._health_cache
        p50, p95 = self._compute()
        if self.error is not None:
            ok, reason = False, self.error
        elif not self.loaded:
            ok, reason = False, "loading"
        else:
            ok, reason = self._ok, ("" if self._ok else self._bad_reason)
        resolved = [w for w in self._recent if w.status != "pending"][-HEALTH_WINDOWS:]
        covered = [w.covered for w in resolved if math.isfinite(w.covered)]
        hop_ms = 1000.0 * self.hop_s
        self._health_cache = {
            "candidate": self.candidate,
            "ok": ok,
            "reason": reason,
            "mock": self.mock,
            "device": self.device if self.loaded else None,
            "window_s": self.window_s,
            "hop_s": self.hop_s,
            "hop_base_s": self.hops[0] / SR,
            "hop_changes": self.hop_changes,
            "hop_pinned": self.hop_pinned,
            "delay_ms": self.latency_ms,
            "compute_p50_ms": p50,
            "compute_p95_ms": p95,
            "rtf": (p50 / hop_ms) if p50 is not None else None,
            "utilisation": round(p50 / hop_ms, 3) if p50 is not None else None,
            "windows": len(self._recent),
            "overruns": self.overruns,
            "late": self.late,
            "lips": covered[-1] if covered else None,
            "enrolled_s": round(self.enroller.collected_s, 2) if self.enroller is not None else None,
            # T-178: the enrolled pick's contrast floor and cos(target, negative) (None: no negative / other candidate)
            "contrast": getattr(self.cand, "contrast", None),
            "pair_cos": _round(getattr(self.cand, "pair_cos", None)),
            "negative_active": getattr(self.cand, "negative_active", None),
            # T-382: the last window's gate state (enrolled pick only), for the UI's gate pill
            "gated": getattr(self.cand, "gated", None),
            "soft_gated": getattr(self.cand, "soft_gated", None),
            # T-465: "stock" / "team", the separator of the last enrolled window (None: other candidate / none yet)
            "separator": getattr(self.cand, "separator_active", None),
            # T-487: the team slot is locked (played every window, no identity gate) and its partner-dominance duck
            "locked": getattr(self.cand, "locked", None),
            "lock_duck_db": _round(getattr(self.cand, "lock_duck_db", None)),
            # solo pass-through (telemetry): True = one voice, False = overlap; the locked target slot plays either way
            "solo": getattr(self.cand, "solo", None),
            "solo_source": getattr(self.cand, "solo_source", None),
            # HEARING_PICK=loudest: "loudest" (the UI shows ISOLATED · loudest voice), else None
            "pick": "loudest" if self._loudest_on() else None,
            "loud_switches": getattr(self.cand, "loud_switches", None),
        }
        self._health_key = key
        return self._health_cache
