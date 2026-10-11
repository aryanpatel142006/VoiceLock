"""Camera-free "Name Call" mode (T-120): whoever says the wearer's name becomes the target.

``control.mode = "name_call"`` with ``control.name = {"text": "Aryan", "aliases": [...]}``. The session
plays :data:`NAME_CALL_SPEC`, a ``tiers`` ladder of the Enrolled tier over Natural noise suppression
(``dfn_ll``): until someone says the name the enrolled tier has no clip and the ladder plays
``dfn_ll``; once a clip arrives the enrolled tier becomes healthy and the ladder crossfades to it.

:class:`NameCaller` runs in a background thread (:meth:`NameCaller.feed` only enqueues):

1. The room audio is cleaned by its own ``dfn_ll`` and cut into phrases by an energy VAD relative to
   the noise floor (:class:`~hearing.tse.enroll.VoicedGate`): a phrase ends after ``PAUSE_S`` of
   silence or at ``MAX_PHRASE_S``.
2. Each phrase goes to a word-timestamped transcriber (faster-whisper ``word_timestamps=True``;
   tests inject their own). :func:`find_name` fuzzy-matches the name and aliases: case and
   punctuation are ignored, a word matches at edit distance <= 1 (names of 4+ letters), or with the
   same phonetic key (:func:`phonetic`, Soundex with every initial vowel alike) at distance <= 2.
3. Enrollment from that utterance: the raw (uncleaned) voiced audio from the phrase start (at most
   ``PRE_S`` before the name) to the phrase end, at least ``MIN_CLIP_S`` of it (else it waits for
   audio after the name, up to ``EXTEND_S``). Its ECAPA embedding is compared with the current
   target's: below ``SAME_VOICE`` it is a new voice (re-target, the old clip becomes the negative
   enrollment), else the same person (the clips are merged). The session hands the clip to the shared
   :class:`~hearing.tse.enroll.AutoEnroller` as a calibrated clip (:meth:`AutoEnroller.enroll_clip`,
   the T-085 path), so the Enrolled tier plays as soon as its embedding is prepared. Overlapping
   voices in the clip are handled by the enrolled candidate's ``clean_enroll`` (MossFormer2
   separation, dominant stream).
4. ``EXTEND_S`` after the name the clip is extended with the voiced audio that followed it when that
   audio matches the clip (cosine >= ``KEEP_SIM``), then every further ``SEGMENT_S`` of voiced audio
   that matches is appended until the clip holds ``MAX_CLIP_S`` (audio-only purity gate). Segments at
   or above ``HEARD_SIM`` count as the target talking: silent for ``QUIET_S`` -> "Target quiet", for
   ``TIMEOUT_S`` -> released (back to Natural). ``control.name_call = "release"`` releases at once.

The clips are voices used as speaker references: in memory only, never written or logged.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from hearing.audio.io import PIPELINE_SR
from hearing.tse.enroll import VoicedGate
from hearing.types import AudioChunk, StageContext

log = logging.getLogger(__name__)

# Natural (dfn_ll) until a clip exists, then the Enrolled tier; a fast ladder so the first window
# after the clip is prepared plays (upgrade 0.5 s = the enrolled hop) and a re-target is not held back.
NAME_CALL_SPEC = 'tiers(only="enrolled,audio-only", upgrade_after_s=0.5, min_switch_s=0)'
HOP = PIPELINE_SR // 100
PAUSE_S = 0.3  # a phrase ends after this much silence ...
MAX_PHRASE_S = 2.5  # ... or at this length (bounds the detection latency in continuous speech)
MIN_PHRASE_S = 0.25  # shorter voiced bursts are not transcribed
OVERLAP_S = 1.5  # a phrase cut inside speech restarts this far back: every word lands inside some window, not on its edge
PREROLL_S = 0.15
RING_S = 20.0  # raw audio kept for enrollment
PRE_S = 0.5  # enrollment starts this long before the name (earlier audio is often the previous talker)
MIN_CLIP_S = 1.0  # voiced audio needed before the first enrollment
EXTEND_S = 2.5  # the clip runs to this long after the name
MAX_CLIP_S = 5.0  # strengthening stops here
SEGMENT_S = 1.5  # voiced audio per tracking/strengthening check
SAME_VOICE = 0.45  # ECAPA cosine of a new name utterance to the target: below = a different voice
KEEP_SIM = 0.55  # a later segment this close to the clip is added to it (purity gate)
HEARD_SIM = 0.4  # a segment this close counts as the target talking
TSE_HEARD_SIM = 0.25  # ... or an enrolled window whose chosen stream is this close (T-120 scene: 0.27-0.45 while the target talks, 0.09-0.22 in its gaps)
QUIET_S = 6.0  # target not heard this long: "Target quiet"
TIMEOUT_S = 60.0  # ... this long: released, back to Natural
RETARGET_WAIT_S = 2.0  # a re-target waits this long after the name for the enrolled tier's verdict (it plays ~1.5 s late)
MAX_BACKLOG_S = 2.0  # the worker this far behind the mic skips windows until it has caught up
COOLDOWN_S = 1.0  # a second detection inside this (the same phrase re-cut) is ignored


@dataclass
class Word:
    text: str
    start: float  # seconds from the start of the transcribed audio
    end: float
    probability: float = 1.0  # Whisper's word probability (T-131 guard)
    avg_logprob: float = 0.0  # ... and its segment's average token log-probability
    no_speech_prob: float = 0.0  # ... and the segment's no-speech probability


Transcriber = Callable[[np.ndarray], list[Word]]
Embedder = Callable[[np.ndarray], np.ndarray]
DEFAULT_PROMPT = False  # T-123 matrix: prompting the name made Whisper hallucinate it (7 false triggers / 5 min)
NAME_PROMPT_ENV = "HEARING_NAMECALL_PROMPT"  # "1": the session's caller prompts the name, with the guard below
# Hit guard (T-131), on by default with the prompt: the T-123 prompt-on false triggers were bare "Aryan." windows.
GUARD_OTHER_WORDS = 2  # the window holds at least this many words besides the name
GUARD_WORD_PROB = 0.1  # every word of the name at least this probable (T-131 matrix: hallucinated "Aryan." 0.00-0.01,
# real prompted names 0.05-0.86; 0.5 kept 3/16)
GUARD_AVG_LOGPROB = -1.0  # its segment's average log-probability above this
GUARD_NO_SPEECH = 0.6  # ... and its no-speech probability below this
GUARD_PROFILE_SIM = 0.45  # with voice profiles loaded (T-125): the caller's ECAPA cosine to one of them
# T-136: the name's voice this close to a saved profile (and SNAP_MARGIN closer than to any other) enrolls that
# profile's clean clip (already prepared by the enrolled stage) instead of the mixed utterance; tried from
# SNAP_MIN_S of voiced audio around the name, again every SNAP_EVERY_S until the normal clip is due
SNAP_SIM = 0.45
SNAP_MARGIN = 0.1
SNAP_MIN_S = 0.5
SNAP_EVERY_S = 0.25


# ------------------------------------------------------------------ name matching

def normalize(word: str) -> str:
    """Lower case, letters only (``"Aryan's,"`` -> ``"aryans"``; possessive handled by the matcher)."""
    return re.sub(r"[^a-z]", "", word.lower())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


_CODES = {**dict.fromkeys("bfpv", "1"), **dict.fromkeys("cgjkqsxz", "2"), **dict.fromkeys("dt", "3"),
          "l": "4", **dict.fromkeys("mn", "5"), "r": "6"}


def phonetic(word: str) -> str:
    """Soundex with every initial vowel (and y) alike: Aryan, Arian, Arion -> ``A65``."""
    w = normalize(word)
    if not w:
        return ""
    head = "A" if w[0] in "aeiouy" else w[0].upper()
    out, last = [], _CODES.get(w[0], "")
    for c in w[1:]:
        code = _CODES.get(c, "")
        if code and code != last:
            out.append(code)
        if c not in "hw":
            last = code
    return head + "".join(out)[:3]


MATCH_LEVELS = ("strict", "normal", "loose")
DEFAULT_MATCH = "normal"  # T-123: chosen from eval.namecall_eval matrix (docs/BENCHMARKS.md)


def word_matches(word: str, name: str, match: str = "loose") -> bool:
    """``strict``: the name exactly (or its possessive); ``loose``: also edit distance <= 1 (names of 4+
    letters) or the same :func:`phonetic` key at distance <= 2. ``normal`` matches like ``loose`` here;
    it differs in :func:`find_name`, which checks the exclude list first."""
    w, n = normalize(word), normalize(name)
    if not w or not n:
        return False
    if w in (n, n + "s"):
        return True
    if match == "strict":
        return False
    d = edit_distance(w, n)
    if len(n) >= 4 and d <= 1:
        return True
    return len(n) >= 3 and d <= 2 and phonetic(w) == phonetic(n)


def excluded(word: str, exclude: list[str]) -> bool:
    w = normalize(word)
    return any(w in (e, e + "s") for e in (normalize(x) for x in exclude) if e)


def find_name(words: list[Word], names: list[str], match: str = "loose",
              exclude: list[str] | None = None) -> tuple[int, int] | None:
    """Index range ``(i, j)`` of the first words matching one of ``names`` (multi-word names match
    consecutive words), else None. ``match`` is ``strict`` (exact name or alias), ``normal`` (fuzzy,
    but never a word in ``exclude`` or in :data:`CONFUSABLE` for the name, unless it is itself a name or
    alias) or ``loose`` (fuzzy, ``exclude`` only)."""
    exclude = list(exclude or [])
    if match == "normal":
        exclude += [c for n in names for c in CONFUSABLE.get(normalize(n), ())]
    own = {normalize(n) for n in names}
    exclude = [e for e in exclude if normalize(e) not in own]
    for i in range(len(words)):
        for name in names:
            parts = [p for p in name.split() if normalize(p)]
            if parts and i + len(parts) <= len(words) and all(
                    not excluded(words[i + k].text, exclude) and word_matches(words[i + k].text, p, match)
                    for k, p in enumerate(parts)):
                return i, i + len(parts)
    return None


def guard_reject(words: list[Word], hit: tuple[int, int]) -> str | None:
    """Why the hit ``words[i:j]`` fails the T-131 transcript guard (:data:`GUARD_OTHER_WORDS` etc.), or None."""
    i, j = hit
    if sum(1 for k, w in enumerate(words) if not i <= k < j and normalize(w.text)) < GUARD_OTHER_WORDS:
        return "bare name"
    for w in words[i:j]:
        if w.probability < GUARD_WORD_PROB:
            return f"word p {w.probability:.2f}"
        if w.avg_logprob <= GUARD_AVG_LOGPROB:
            return f"avg_logprob {w.avg_logprob:.2f}"
        if w.no_speech_prob >= GUARD_NO_SPEECH:
            return f"no_speech {w.no_speech_prob:.2f}"
    return None


# Common real names a fuzzy match would take for the wearer's (Q-313/Q-315: "Ryan" for "Aryan");
# ``normal`` excludes them, ``loose`` does not. Extend with ``control.name.exclude``.
CONFUSABLE: dict[str, tuple[str, ...]] = {
    "aryan": ("ryan", "brian", "bryan", "orion", "arya"),
}


# ------------------------------------------------------------------ engines

def whisper_prompt(names: list[str]) -> str:
    """The ``initial_prompt`` that spells the name for Whisper (T-123): ``"Aryan, Ary."``."""
    return ", ".join(n for n in names if n.strip()) + "." if any(n.strip() for n in names) else ""


def whisper_transcriber(names: Callable[[], list[str]] | None = None) -> Transcriber | None:
    """faster-whisper with word timestamps (the synth ASR's model and loader), or None when it is
    not installed/cached. With ``names`` (the caller's live name + aliases) each window is decoded
    with ``initial_prompt`` = the names and, on faster-whisper >= 1.1, ``hotwords`` = the same, so the
    name is spelled right (``Aryan``, not ``Ariane``: 16/16 vs 7/16 exact). Off by default
    (:data:`DEFAULT_PROMPT`): ``eval.namecall_eval matrix`` (T-123) measured 7 false triggers in 5 min of speech
    without the name ("Aryan." hallucinated) against 0 without the prompt."""
    import inspect

    from hearing import models
    from hearing.synth import asr

    if not asr.local_available():
        return None
    key = asr._whisper_loader(asr.whisper_model_name())
    hotwords_ok: list[bool] = []

    def transcribe(x: np.ndarray) -> list[Word]:
        model = models.load(key)
        extra: dict = {}
        prompt = whisper_prompt(names()) if names is not None else ""
        if prompt:
            if not hotwords_ok:
                hotwords_ok.append("hotwords" in inspect.signature(model.transcribe).parameters)
            extra["initial_prompt"] = prompt
            if hotwords_ok[0]:
                extra["hotwords"] = prompt
        segments, _ = model.transcribe(x, language="en", beam_size=1, vad_filter=False,
                                       condition_on_previous_text=False, word_timestamps=True,
                                       max_new_tokens=40, temperature=0.0, **extra)  # no fallback re-decodes on babble
        return [Word(w.word.strip(), float(w.start), float(w.end), float(w.probability), float(s.avg_logprob),
                     float(s.no_speech_prob)) for s in segments for w in (s.words or [])]

    return transcribe


def ecapa_embedder() -> Embedder | None:
    """Unit ECAPA embeddings (the enrolled candidate's model, under the session's torch lock), or
    None when the weights are absent."""
    from hearing import models
    from hearing.tse import enrolled
    from hearing.tse.gpu import TORCH_LOCK

    if not enrolled.available():
        return None

    def embed(x: np.ndarray) -> np.ndarray:
        with TORCH_LOCK:
            return enrolled._unit_embedding(models.load(enrolled.MODEL), np.asarray(x, np.float32))

    return embed


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d > 1e-12 else 0.0


# ------------------------------------------------------------------ the caller

@dataclass
class Detection:
    text: str  # the phrase transcript
    word: str  # the words that matched
    t_name: float  # audio clock (s since the caller started) of the name's start
    t_end: float  # ... and end
    latency_ms: float  # wall time from the name's last audio arriving to the detection
    new_voice: bool = True
    similarity: float | None = None  # to the previous target (None: there was none)
    profile: str | None = None  # T-136: the saved profile the voice snapped to
    profile_sim: float | None = None  # ... and its ECAPA cosine


@dataclass
class _Pending:
    detection: Detection
    start: int  # enrollment start (sample position)
    name_end: int
    snapped_at: int = 0  # voiced samples at the last profile-snap try (T-136)


@dataclass
class NameEvent:
    kind: str  # "enroll" | "release"
    clip: np.ndarray | None = None
    other: np.ndarray | None = None
    reason: str = ""
    detection: Detection | None = None
    retarget: bool = False
    profile: str | None = None  # T-136: enroll this saved profile (``pieces``/``other_pieces``), not ``clip``
    pieces: list[np.ndarray] | None = None
    other_pieces: list[np.ndarray] | None = None


@dataclass
class _Target:
    clip: np.ndarray
    emb: np.ndarray
    last_heard: int  # sample position
    since: int
    extend_at: int | None = None  # position after which the post-name audio is merged
    extend_from: int = 0
    seg: list[np.ndarray] = field(default_factory=list)
    profile: str | None = None  # a saved profile (T-136): its clean clip is never grown with room audio


class NameCaller:
    """See the module doc. ``transcriber``/``embedder``/``cleaner`` default to faster-whisper, ECAPA
    and ``dfn_ll``; without Whisper nothing is ever detected (``engine`` says ``none``)."""

    @property
    def quiet_s(self) -> float:
        return float(QUIET_S if self._quiet_s is None else self._quiet_s)

    @quiet_s.setter
    def quiet_s(self, value: float | None) -> None:
        self._quiet_s = value

    @property
    def timeout_s(self) -> float:
        return float(TIMEOUT_S if self._timeout_s is None else self._timeout_s)

    @timeout_s.setter
    def timeout_s(self, value: float | None) -> None:
        self._timeout_s = value

    def __init__(self, name: str = "", aliases: list[str] | None = None, *,
                 transcriber: Transcriber | None = None, embedder: Embedder | None = None,
                 cleaner: str | object | None = "dfn_ll", threaded: bool = True, sr: int = PIPELINE_SR,
                 match: str = DEFAULT_MATCH, exclude: list[str] | None = None, prompt: bool = DEFAULT_PROMPT,
                 guard: bool | None = None, profiles: list[np.ndarray] | None = None,
                 saved: dict[str, list[np.ndarray]] | None = None,
                 quiet_s: float | None = None, timeout_s: float | None = None):
        self.sr = sr
        self._quiet_s, self._timeout_s = quiet_s, timeout_s  # the team preset: 10 s / 180 s (T-130); None: QUIET_S / TIMEOUT_S
        self.name, self.aliases = name.strip(), [a.strip() for a in (aliases or []) if a.strip()]
        self._lock = threading.Lock()
        self.match, self.exclude = DEFAULT_MATCH, []
        self.set_match(match, exclude)
        self.prompt = prompt
        self.guard = prompt if guard is None else guard  # T-131: the prompt mode's hit guard
        self._profile_clips = [np.asarray(p, np.float32) for p in (profiles or []) if len(p)]
        self._profile_embs: list[np.ndarray] | None = None  # embedded on the worker at first use
        # T-136: saved voice profiles (name -> clean pieces) the name's voice may snap to
        self._saved = {n: [np.asarray(p, np.float32) for p in ps if len(p)] for n, ps in (saved or {}).items()}
        self._saved = {n: ps for n, ps in self._saved.items() if ps}
        self._saved_embs: dict[str, np.ndarray] | None = None
        self.guarded = 0  # hits the guard dropped
        self.guard_reasons: list[str] = []
        self.transcriber = transcriber if transcriber is not None else whisper_transcriber(
            (lambda: self.names) if prompt else None)
        self.engine = "whisper" if transcriber is None and self.transcriber is not None else (
            "custom" if transcriber is not None else "none")
        self.embedder = embedder if embedder is not None else (ecapa_embedder() or _spectral_embedder)
        self.cleaner = _build_cleaner(cleaner)
        self._ctx = StageContext.default()
        self._ctx.suppression = 1.0
        self._events: list[NameEvent] = []
        self.detections: list[Detection] = []
        self.ignored = 0  # name heard again from the current target (not a re-target)
        self.skipped = 0  # windows not transcribed because the worker was behind the mic
        self._q: queue.Queue | None = None
        self.reset()
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        if threaded:
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._loop, name="namecall", daemon=True)
            self._thread.start()

    # -------------------------------------------------- main thread
    def set_name(self, name: str, aliases: list[str] | None = None) -> None:
        with self._lock:
            self.name, self.aliases = name.strip(), [a.strip() for a in (aliases or []) if a.strip()]

    def set_match(self, match: str | None = None, exclude: list[str] | None = None) -> None:
        """``match`` strict|normal|loose and the ``exclude`` words (never the name), each optional."""
        with self._lock:
            if match is not None:
                if match not in MATCH_LEVELS:
                    raise ValueError(f"match must be one of {MATCH_LEVELS}, not {match!r}")
                self.match = match
            if exclude is not None:
                self.exclude = [e.strip() for e in exclude if e.strip()]

    @property
    def names(self) -> list[str]:
        return [n for n in [self.name, *self.aliases] if n]

    def feed(self, samples: np.ndarray) -> None:
        x = np.asarray(samples, np.float32).copy()
        if self._q is not None:
            self._q.put(("audio", x, time.monotonic()))
        else:
            self._step(x, time.monotonic())

    def release(self, reason: str = "released") -> None:
        if self._q is not None:
            self._q.put(("release", reason, 0.0))
        else:
            self._release(reason)

    def heard(self) -> None:
        """The target was heard now (the session's enrolled tier picked a stream close to it): keeps
        the pill off "Target quiet" and the release timeout away while the target talks over others."""
        with self._lock:
            if self.target is not None:
                self.target.last_heard = self.pos
                if not self._heard_marks or self.pos - self._heard_marks[-1] >= self.sr // 10:
                    self._heard_marks.append(self.pos)

    def poll(self) -> list[NameEvent]:
        with self._lock:
            events, self._events = self._events, []
        return events

    def drain(self, timeout: float = 5.0) -> None:
        """Wait until the worker has processed everything fed so far (tests, offline evals)."""
        if self._q is None:
            return
        done = threading.Event()
        self._q.put(("sync", done, 0.0))
        done.wait(timeout)

    def close(self) -> None:
        if self._q is not None and self._thread is not None:
            self._q.put(("stop", None, 0.0))
            self._thread.join(timeout=2.0)
            self._q = self._thread = None

    def status(self) -> dict:
        with self._lock:
            target, last = self.target, (self.detections[-1] if self.detections else None)
            pos = self.pos
            pending = self._pending is not None
        heard_s = None if target is None else round((pos - target.last_heard) / self.sr, 2)
        return {
            "name": self.name, "aliases": list(self.aliases), "engine": self.engine,
            "match": self.match, "exclude": list(self.exclude), "prompt": self.prompt,
            "target": target is not None, "pending": pending,
            "clip_s": round(len(target.clip) / self.sr, 2) if target is not None else 0.0,
            "since_heard_s": heard_s, "quiet": heard_s is not None and heard_s >= self.quiet_s,
            "detections": len(self.detections), "skipped": self.skipped,
            "guard": self.guard, "guarded": self.guarded, "profiles": len(self._profile_clips),
            "saved_profiles": sorted(self._saved), "target_profile": None if target is None else target.profile,
            "last": None if last is None else {
                "word": last.word, "text": last.text, "latency_ms": round(last.latency_ms, 1),
                "new_voice": last.new_voice, "profile": last.profile,
                "profile_sim": None if last.profile_sim is None else round(last.profile_sim, 3),
                "similarity": None if last.similarity is None else round(last.similarity, 3)},
        }

    # -------------------------------------------------- worker
    def reset(self) -> None:
        self.pos = 0
        self.target: _Target | None = None
        self._pending: _Pending | None = None
        self._chunks: deque[tuple[int, np.ndarray, np.ndarray]] = deque()  # (position, raw, cleaned)
        self._ring_base = 0
        self._heard_marks: deque[int] = deque(maxlen=round(RING_S * 10))
        self._voiced: deque[tuple[int, bool]] = deque()  # (position, voiced) per 10 ms of the ring
        self._walls: deque[tuple[int, float]] = deque(maxlen=round(RING_S * 100))
        self._gate = VoicedGate(self.sr)
        self._phrase_start: int | None = None
        self._phrase_last = 0  # end of the last voiced hop of the phrase
        self._silent = 0
        self._last_detect = -10 ** 9

    def _loop(self) -> None:
        if self.transcriber is not None:
            try:  # load and warm the models before the first phrase (the first call took 2 s)
                self.transcriber(np.zeros(self.sr, np.float32))
                self.embedder(np.random.default_rng(0).normal(0, 0.01, self.sr).astype(np.float32))
                self._saved_embeddings()  # T-136: the profiles' embeddings before the first name
            except Exception:
                log.exception("name call: warming the transcriber failed")
        while True:
            kind, a, wall = self._q.get()
            try:
                if kind == "stop":
                    return
                if kind == "audio":
                    self._step(a, wall)
                elif kind == "release":
                    self._release(a)
                elif kind == "sync":
                    a.set()
            except Exception:
                log.exception("name call worker failed on one step")

    def _step(self, x: np.ndarray, wall: float) -> None:
        clean = x
        if self.cleaner is not None:
            clean = np.asarray(self.cleaner.process(AudioChunk(x, self.sr, self.pos // HOP, 0.0), self._ctx).samples,
                               np.float32)
            if len(clean) != len(x):
                clean = np.resize(clean, len(x)) if len(clean) else np.zeros_like(x)
        start = self.pos
        self._walls.append((start + len(x), wall))
        self._chunks.append((start, x, clean))
        voiced = self._gate.voiced(x)
        self._voiced.append((start, voiced))
        self.pos += len(x)
        while self.pos - self._chunks[0][0] > RING_S * self.sr:
            self._chunks.popleft()
            self._ring_base = self._chunks[0][0]
            while self._voiced and self._voiced[0][0] < self._ring_base:
                self._voiced.popleft()
        self._segment(voiced, start)
        self._track(x, voiced)
        self._advance_pending()

    def _segment(self, voiced: bool, start: int) -> None:
        if self._phrase_start is None:
            if voiced:
                self._phrase_start = max(self._ring_base, start - round(PREROLL_S * self.sr))
                self._phrase_last, self._silent = self.pos, 0
            return
        if voiced:
            self._phrase_last, self._silent = self.pos, 0
        else:
            self._silent += 1
        long = self.pos - self._phrase_start >= MAX_PHRASE_S * self.sr
        if self._silent * HOP >= PAUSE_S * self.sr or long:
            p0, p1 = self._phrase_start, self._phrase_last
            self._phrase_start = None
            if long and voiced:  # cut inside speech: the next phrase overlaps this one by OVERLAP_S
                self._phrase_start = max(self._ring_base, self.pos - round(OVERLAP_S * self.sr))
                self._phrase_last = self.pos
            if p1 - p0 >= MIN_PHRASE_S * self.sr:
                wall = self._walls[-1][1] if self._walls else None
                if self._q is not None and wall is not None and time.monotonic() - wall > MAX_BACKLOG_S:
                    self.skipped += 1  # behind the mic (a slow window, a cold model): catch up first
                else:
                    self._transcribe(p0, p1)

    def _slice(self, a: int, b: int, clean: bool = False) -> np.ndarray:
        k = 2 if clean else 1
        parts = [c[k][max(0, a - c[0]) : max(0, b - c[0])] for c in self._chunks if c[0] < b and c[0] + len(c[1]) > a]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _voiced_audio(self, a: int, b: int) -> np.ndarray:
        parts = [c[1] for c, (_, v) in zip(self._chunks, self._voiced, strict=False) if v and a <= c[0] < b]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _wall_at(self, pos: int) -> float | None:
        return next((w for p, w in self._walls if p >= pos), None)

    def _transcribe(self, p0: int, p1: int) -> None:
        names = self.names
        if not names or self.transcriber is None:
            return
        words = self.transcriber(self._slice(p0, p1, clean=True))
        hit = find_name(words, names, self.match, self.exclude)
        if hit is None:
            return
        if self.guard and (why := guard_reject(words, hit)) is not None:
            self._guarded(why)
            return
        i, j = hit
        t_name, t_end = p0 + round(words[i].start * self.sr), p0 + round(words[j - 1].end * self.sr)
        if t_name - self._last_detect < COOLDOWN_S * self.sr:
            return
        self._last_detect = t_name
        wall = self._wall_at(min(t_end, p1))
        latency = (time.monotonic() - wall) * 1000.0 if wall is not None else float("nan")
        det = Detection(" ".join(w.text for w in words).strip(), " ".join(w.text for w in words[i:j]).strip(),
                        t_name / self.sr, t_end / self.sr, latency)
        with self._lock:
            self.detections.append(det)
            self._pending = _Pending(det, max(p0, t_name - round(PRE_S * self.sr)), t_end)
        self._advance_pending()

    def _advance_pending(self) -> None:
        pending = self._pending
        if pending is None:
            return
        clip = self._voiced_audio(pending.start, self.pos)
        late = self.pos >= pending.name_end + EXTEND_S * self.sr
        due = len(clip) >= MIN_CLIP_S * self.sr or late
        if (self._saved and len(clip) >= SNAP_MIN_S * self.sr
                and (due or len(clip) - pending.snapped_at >= SNAP_EVERY_S * self.sr)):
            pending.snapped_at = len(clip)
            if self._snap(pending, clip):
                return
        if not due:
            return  # wait for more of the voice after the name
        if len(clip) < 0.5 * self.sr:
            with self._lock:
                self._pending = None
            return  # a name with almost no voiced audio around it: ignore
        old = self.target
        if old is not None and self.pos < pending.name_end + RETARGET_WAIT_S * self.sr:
            return  # wait for the enrolled tier's verdict on the name's audio
        emb = self.embedder(clip)
        det = pending.detection
        if self.guard and self._profile_clips:
            if self._profile_embs is None:
                self._profile_embs = [self.embedder(c) for c in self._profile_clips]
            best = max(_cos(emb, p) for p in self._profile_embs)
            if best < GUARD_PROFILE_SIM:
                with self._lock:
                    self._pending = None
                self._guarded(f"profile cos {best:.2f}")
                return
        if old is not None:
            det.similarity = _cos(emb, old.emb)
            # the playing enrolled tier heard the current target during the name (its marks lag ~1.5 s)
            heard = any(det.t_name * self.sr <= m <= pending.name_end + RETARGET_WAIT_S * self.sr
                        for m in self._heard_marks)
            det.new_voice = det.similarity < SAME_VOICE and not heard
        if old is not None and not det.new_voice:  # the same person again: no re-target, clip unchanged
            with self._lock:
                self._pending = None
                old.last_heard = self.pos
            self.ignored += 1  # (a mixed clip from an overlap would only dilute the voice print)
            return
        other = old.clip if old is not None else self._other_before(pending.start, emb)
        self._set_target(clip, other, retarget=old is not None, detection=det, last_heard=self.pos,
                         extend_from=self.pos, extend_at=pending.name_end + round(EXTEND_S * self.sr), emb=emb)

    def _saved_embeddings(self) -> dict[str, np.ndarray]:
        if self._saved_embs is None:  # the mean of the pieces' unit embeddings, as the enrolled candidate does
            embs = {}
            for n, pieces in self._saved.items():
                mean = np.mean([_unit(self.embedder(p)) for p in pieces], axis=0)
                embs[n] = _unit(mean)
            self._saved_embs = embs
        return self._saved_embs

    def nearest_profile(self, clip: np.ndarray) -> tuple[str | None, float, np.ndarray]:
        """``(profile, cosine, embedding of clip)``: the saved profile ``clip`` snaps to, or None when the
        best is under ``SNAP_SIM`` or not ``SNAP_MARGIN`` ahead of the runner-up."""
        emb = self.embedder(clip)
        scores = sorted(((_cos(emb, e), n) for n, e in self._saved_embeddings().items()), reverse=True)
        if not scores:
            return None, 0.0, emb
        best, name = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        return (name if best >= SNAP_SIM and best - second >= SNAP_MARGIN else None), best, emb

    def _snap(self, pending: _Pending, clip: np.ndarray) -> bool:
        """T-136: the name's voice is a saved profile: enroll its clean pieces at once (or, when it is
        already the target, keep it). False: not close enough to any profile yet, the clip path goes on."""
        name, sim, _ = self.nearest_profile(clip)
        if name is None:
            return False
        det, old = pending.detection, self.target
        det.profile, det.profile_sim = name, sim
        if old is not None:
            det.similarity = sim if old.profile == name else None
            det.new_voice = old.profile != name
            if not det.new_voice:  # the saved target said the name again
                with self._lock:
                    self._pending = None
                    old.last_heard = self.pos
                self.ignored += 1
                return True
        pieces = self._saved[name]
        others = [p for n, ps in self._saved.items() if n != name for p in ps]
        with self._lock:
            self.target = _Target(np.concatenate(pieces), self._saved_embeddings()[name], self.pos, self.pos,
                                  profile=name)
            self._pending = None
            self._events.append(NameEvent("enroll", self.target.clip.copy(), None, detection=det,
                                          retarget=old is not None, profile=name, pieces=list(pieces),
                                          other_pieces=others or None))
        return True

    def _guarded(self, why: str) -> None:
        self.guarded += 1
        self.guard_reasons = [*self.guard_reasons[-9:], why]
        log.info("name call: hit dropped by the guard (%s)", why)

    def _other_before(self, start: int, emb: np.ndarray) -> np.ndarray | None:
        """Negative enrollment: up to 2 s of voiced audio from the 6 s before the utterance, when it
        is not the same voice (the enrolled candidate ignores a negative too close to the target)."""
        before = self._voiced_audio(start - round(6.0 * self.sr), start)[-round(2.0 * self.sr):]
        if len(before) < 1.0 * self.sr:
            return None
        return before if _cos(self.embedder(before), emb) < SAME_VOICE else None

    def _set_target(self, clip: np.ndarray, other: np.ndarray | None, *, retarget: bool, detection: Detection | None,
                    last_heard: int, extend_from: int = 0, extend_at: int | None = None,
                    emb: np.ndarray | None = None) -> None:
        emb = self.embedder(clip) if emb is None else emb
        with self._lock:
            old = self.target
            self.target = _Target(clip, emb, last_heard, old.since if old is not None and not retarget else self.pos,
                                  extend_at, extend_from)
            self._pending = None  # cleared only now: the pill says "locking" while the clip is embedded
            self._events.append(NameEvent("enroll", clip.copy(), None if other is None else other.copy(),
                                          detection=detection, retarget=retarget))

    def _track(self, x: np.ndarray, voiced: bool) -> None:
        t = self.target
        if t is None:
            return
        if t.extend_at is not None:
            if self.pos >= t.extend_at:
                ext = self._voiced_audio(t.extend_from, self.pos)
                t.extend_at = None
                if len(ext) >= 0.5 * self.sr and _cos(self.embedder(ext), t.emb) >= KEEP_SIM:
                    self._set_target(np.concatenate([t.clip, ext])[-round(MAX_CLIP_S * self.sr):], None,
                                     retarget=False, detection=None, last_heard=self.pos)
            return  # the post-name audio is the extension's, not a tracking segment
        if voiced and self._pending is None:
            t.seg.append(x)
        if sum(len(s) for s in t.seg) >= SEGMENT_S * self.sr:
            seg, t.seg = np.concatenate(t.seg), []
            sim = _cos(self.embedder(seg), t.emb)
            if sim >= HEARD_SIM:
                t.last_heard = self.pos
            if sim >= KEEP_SIM and len(t.clip) < MAX_CLIP_S * self.sr and t.profile is None:
                self._set_target(np.concatenate([t.clip, seg])[-round(MAX_CLIP_S * self.sr):], None,
                                 retarget=False, detection=None, last_heard=self.pos)
        if self.pos - t.last_heard >= self.timeout_s * self.sr:
            self._release(f"target silent for {self.timeout_s:g} s")

    def _release(self, reason: str) -> None:
        with self._lock:
            had = self.target is not None or self._pending is not None
            self.target, self._pending = None, None
            if had:
                self._events.append(NameEvent("release", reason=reason))


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _spectral_embedder(x: np.ndarray) -> np.ndarray:
    """Fallback without ECAPA weights: the enrolled candidate's mock backend (not speaker recognition)."""
    from hearing.tse.enrolled import MockBackend

    v = np.asarray(MockBackend().embed(np.asarray(x, np.float32)), np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _build_cleaner(cleaner):
    if cleaner is None or not isinstance(cleaner, str):
        return cleaner
    import importlib

    from hearing import registry

    try:
        importlib.import_module("hearing.enhance.dfn")  # registers dfn / dfn_ll
        return registry.build_pipeline(cleaner)
    except Exception as e:  # noqa: BLE001  (missing weights: transcribe the raw room)
        log.info("name call: cleaner %r unavailable (%s), using the raw room", cleaner, e)
        return None
