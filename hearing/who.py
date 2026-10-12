"""Who's speaking (the ``/who`` page): which of the two team voices is talking right now.

The research has two halves. Live Focus mode separates voices with the stock separator and plays the loudest.
Knowing *whose* voice each one is took a model fine-tuned on Aryan's and Joseph's voices: the fixed-slot team
separator (``models/T-374/v2_t387.pt``, T-387; slot 0 = aryan, slot 1 = joseph, :func:`hearing.tse.blind.fixed_slots`).
Fusing both into one model needs a training run the hackathon had no time or compute for, so this module shows the
second half on its own: per window the fine-tuned model splits the audio into one stream per teammate and the share
of energy that lands in each slot says who is talking. ECAPA cosine against the saved voice profiles (the stock
speaker encoder of :mod:`hearing.tse.enrolled`) is the second opinion; the two are averaged. :class:`WhoSmoother`
turns that per-window evidence into a label that does not flicker: a short EMA, hysteresis (the other person must be
clearly ahead for ``switch_s``) and a hold through short pauses. No audio is produced anywhere here.

Measured offline by ``eval/who_eval.py`` (numbers in its report and in the /who commit message).
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from hearing import models
from hearing.tse import blind, enrolled
from hearing.tse.enroll import PROFILES_ENV, load_profile, profiles_dir

SR = 16000
WINDOW_S = 1.0  # separator window: the length the T-374 checkpoints were trained on
EMBED_S = 1.5  # ECAPA window
LEVEL_S = 0.4  # the speech flag looks at this much of the newest audio (shorter = quicker pauses)
HOP_S = 0.25  # analysis hop the server aims for (compute-bound on the Mac: ~0.4 s per analysis)
SPEECH_FLOOR_DB = -55.0  # never call anything under this speech
SPEECH_MARGIN_DB = 10.0  # ... or anything less than this above the running noise floor
NOISE_WINDOW_S = 10.0  # the noise floor is the 10th percentile of the last 10 s of level readings
NOISE_PERCENTILE = 10
NOISE_CAP_DB = -40.0  # the floor never rises above this: continuous talk from Start must not become the 'noise'
ECAPA_SCALE = 6.0  # cosine difference -> [-1, 1] evidence (profiles sit ~0.1-0.2 apart on the demo mics)
MATCH_COS = (0.2, 0.8)  # ECAPA cosine -> page "match": 0.2 (an unrelated voice on the demo mics) = 0, 0.8 (same voice) = 1
RUN_GAP_S = 0.75  # a run of agreeing windows may start this much before the hysteresis span and still count
WHO_MODEL = f"{blind.MODEL}@who"  # hearing.models name of the checkpoint loaded here
CKPT_ENV = "HEARING_WHO_CKPT"
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "models" / "T-374" / "v2_t387.pt"
DEFAULT_PROFILES = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "generated" / "profiles"
METHOD = "fixed-slot energy share (fine-tuned T-387 separator) + ECAPA cosine to the saved profiles, averaged"


def who_checkpoint() -> Path | None:
    """The fine-tuned checkpoint to run: ``$HEARING_WHO_CKPT``, else the server's ``$HEARING_SEP_TEAM_CKPT``, else
    ``$HEARING_SEP_CKPT`` under ``HEARING_PICK=fixed``, else ``models/T-374/v2_t387.pt`` when present, else None."""
    for raw in (os.environ.get(CKPT_ENV, ""), os.environ.get(blind.TEAM_CKPT_ENV, ""),
                os.environ.get(blind.SEP_CKPT_ENV, "") if blind.pick_fixed() else ""):
        raw = raw.strip()
        if raw:
            path = Path(raw).expanduser()
            if not path.is_file():
                raise RuntimeError(f"who checkpoint is not a file: {path}")
            return path
    return DEFAULT_CKPT if DEFAULT_CKPT.is_file() else None


def who_profiles_dir() -> Path | None:
    """``$HEARING_PROFILES_DIR`` (the server's profiles) else the repo's ``eval/fixtures/generated/profiles``."""
    root = profiles_dir()
    if root is not None:
        return root
    return DEFAULT_PROFILES if DEFAULT_PROFILES.is_dir() else None


def available() -> bool:
    try:
        ckpt, root = who_checkpoint(), who_profiles_dir()
    except RuntimeError:
        return False
    if ckpt is None or root is None or not blind.available() or not enrolled.available():
        return False
    return all((root / n).is_dir() for n in blind.fixed_slots())


def level_db(x) -> float:
    """RMS level in dBFS (-120 for silence)."""
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return -120.0
    return float(10.0 * math.log10(max(float(np.mean(x * x)), 1e-12)))


def slot_evidence(streams) -> float:
    """Energy share of slot 0 against slot 1 as evidence in [-1, 1]: +1 = everything in slot 0 (the first name)."""
    e = [max(float(np.mean(np.asarray(s, np.float64) ** 2)), 0.0) for s in streams]
    total = e[0] + e[1]
    return float((e[0] - e[1]) / total) if total > 1e-14 else 0.0


def ecapa_evidence(cosines) -> float:
    """Cosine lead of profile 0 over profile 1, scaled to [-1, 1]."""
    return float(np.clip((cosines[0] - cosines[1]) * ECAPA_SCALE, -1.0, 1.0))


def match_of(cos: float) -> float:
    """One profile's ECAPA cosine as the /who page's match in [0, 1]: linear from :data:`MATCH_COS` (0.2 -> 0, 0.8 -> 1),
    clipped. web/console/who-view.js ``matchOfCos`` maps raw.cos the same way for older servers."""
    lo, hi = MATCH_COS
    return float(np.clip((float(cos) - lo) / (hi - lo), 0.0, 1.0))


def combine(slot: float | None, ecapa: float | None, slot_weight: float = 0.5) -> float | None:
    """The averaged evidence (None when there is none)."""
    if slot is None and ecapa is None:
        return None
    if slot is None:
        return ecapa
    if ecapa is None:
        return slot
    return float(np.clip(slot_weight * slot + (1.0 - slot_weight) * ecapa, -1.0, 1.0))


class NoiseFloor:
    """Running noise floor: a low percentile of the recent level readings (seconds, not hops, so a change of hop keeps it)."""

    def __init__(self, window_s: float = NOISE_WINDOW_S, percentile: float = NOISE_PERCENTILE):
        self.window_s, self.percentile = window_s, percentile
        self._levels: list[tuple[float, float]] = []

    def update(self, t: float, level: float) -> float:
        self._levels.append((t, level))
        while self._levels and self._levels[0][0] < t - self.window_s:
            self._levels.pop(0)
        return self.floor

    @property
    def floor(self) -> float | None:
        if not self._levels:
            return None
        return min(float(np.percentile([lv for _, lv in self._levels], self.percentile)), NOISE_CAP_DB)


def is_speech(level: float, floor: float | None) -> bool:
    if level < SPEECH_FLOOR_DB:
        return False
    return floor is None or level >= floor + SPEECH_MARGIN_DB


class WhoModel:
    """The fine-tuned separator plus ECAPA against the saved profiles; :meth:`score` one window at a time.

    Inject ``separator`` (anything with ``separate(mix)`` -> two streams), ``embedder`` (``embed(audio)`` -> vector) and
    ``profile_embeddings`` (two unit vectors) for offline tests; the real loaders run under :data:`TORCH_LOCK`.
    """

    def __init__(self, separator=None, embedder=None, profile_embeddings=None, names=None, checkpoint=None,
                 profiles=None, slot_weight: float = 0.5):
        self.separator, self.embedder = separator, embedder
        self.profile_embeddings = None if profile_embeddings is None else [np.asarray(v, np.float64) for v in profile_embeddings]
        self.names = tuple(names) if names else blind.fixed_slots()
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.profiles = Path(profiles) if profiles else None
        self.slot_weight = slot_weight
        self.loaded = False
        self.device = "cpu"

    def load(self) -> None:
        if self.loaded:
            return
        from hearing.tse.gpu import TORCH_LOCK

        with TORCH_LOCK:
            if self.separator is None:
                ckpt = self.checkpoint or who_checkpoint()
                if ckpt is None:
                    raise RuntimeError(f"no fine-tuned checkpoint: set {CKPT_ENV} or put models/T-374/v2_t387.pt in place")
                self.checkpoint = ckpt
                team = blind.team_checkpoint()
                if team is not None and team.resolve() == ckpt.resolve():
                    backend = models.load(blind.TEAM_MODEL)  # the live engine's instance, not a second 223 MB copy
                else:
                    if WHO_MODEL not in models._loaders:
                        models.register_loader(WHO_MODEL)(lambda: blind._load_mossformer(who_checkpoint() or ckpt))
                    backend = models.load(WHO_MODEL)
                self.separator = blind.BlindSeparator(backend=backend)
                self.separator.load()
            if self.embedder is None:
                self.embedder = models.load(enrolled.MODEL)
            if self.profile_embeddings is None:
                root = self.profiles or who_profiles_dir()
                if root is None:
                    raise RuntimeError(f"no voice profiles: set {PROFILES_ENV}")
                self.profiles = root
                self.profile_embeddings = [self._centroid(load_profile(n, root)) for n in self.names]
        self.device = getattr(self.separator, "device", None) or getattr(getattr(self.separator, "backend", None), "device", "cpu")
        self.loaded = True

    def _centroid(self, pieces) -> np.ndarray:
        vectors = [enrolled._unit_embedding(self.embedder, np.asarray(p, np.float32)) for p in pieces]
        mean = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-12 else mean

    def score(self, audio, *, use_slot: bool = True, use_ecapa: bool = True) -> dict:
        """Evidence for the newest audio (16 kHz float, at least :data:`LEVEL_S` long; longer is cut to the windows).

        ``slot`` is the fine-tuned model's energy share (+ = first name), ``ecapa`` the profile cosine lead, ``evidence``
        their average; ``level_db`` is the level of the last :data:`LEVEL_S`.
        """
        from hearing.tse.gpu import TORCH_LOCK

        self.load()
        x = np.ascontiguousarray(np.asarray(audio, np.float32))
        out: dict = {"level_db": level_db(x[-int(LEVEL_S * SR):]), "slot": None, "ecapa": None, "cos": None, "energy_db": None}
        with TORCH_LOCK:
            if use_slot:
                streams = self.separator.separate(x[-int(WINDOW_S * SR):])
                out["slot"] = slot_evidence(streams)
                out["energy_db"] = [level_db(s) for s in streams]
            if use_ecapa:
                emb = enrolled._unit_embedding(self.embedder, x[-int(EMBED_S * SR):])
                cos = [float(np.dot(emb, c)) for c in self.profile_embeddings]
                out["cos"] = cos
                out["ecapa"] = ecapa_evidence(cos)
        out["evidence"] = combine(out["slot"], out["ecapa"], self.slot_weight)
        return out


class WhoSmoother:
    """Per-window evidence -> a calm label.

    * Hysteresis: the label moves only when every speech window of the last ``switch_s`` points at the other
      person and their mean lead is past ``margin`` (``first_s`` of agreement when no one is labelled yet).
    * The EMA over ``smooth_s`` only sets the confidence (how far the recent evidence sits behind the label).
    * Hold: the label survives pauses up to ``hold_s``; longer, and the label clears to "no one speaking".
    ``update`` returns the state the page shows: ``speaker`` (name or None), ``state`` (``speaking`` / ``hold`` /
    ``unsure`` / ``listening``), ``confidence`` in [0, 1] and ``switched`` (the label changed on this update).
    """

    def __init__(self, names=("aryan", "joseph"), *, smooth_s: float = 0.5, switch_s: float = 0.8, hold_s: float = 1.5,
                 margin: float = 0.15, first_s: float = 0.3, sure: float = 0.5):
        if len(names) != 2:
            raise ValueError("two names")
        self.names = tuple(names)
        self.smooth_s, self.switch_s, self.hold_s = smooth_s, switch_s, hold_s
        self.margin, self.first_s, self.sure = margin, first_s, sure
        self.reset()

    def reset(self) -> None:
        self.current: int | None = None
        self.last_speech_t: float | None = None
        self.last_t: float | None = None
        self.ema = 0.0
        self.switches = 0
        self._recent: list[tuple[float, float]] = []  # (t, evidence) of the speech windows inside switch_s

    def _agree(self, t: float, span: float) -> int | None:
        """The person every speech window of the last ``span`` seconds points at (past the margin), else None.
        The newest window older than ``span`` (within :data:`RUN_GAP_S`) counts too: it shows the run began before."""
        recent = [(u, e) for u, e in self._recent if u >= t - span - 1e-9]
        older = [(u, e) for u, e in self._recent if u < t - span - 1e-9]
        if older:
            recent = [older[-1]] + recent
        elif len(recent) < 2 or t - recent[0][0] < span - 1e-9:
            return None
        signs = {1 if e > 0 else -1 if e < 0 else 0 for _, e in recent}
        mean = float(np.mean([e for _, e in recent]))
        if len(signs) != 1 or abs(mean) <= self.margin:
            return None
        return 0 if mean > 0 else 1

    def update(self, t: float, speech: bool, evidence: float | None) -> dict:
        switched = False
        if speech and evidence is not None:
            dt = (t - self.last_t) if self.last_t is not None else self.smooth_s
            alpha = 1.0 - math.exp(-max(dt, 0.0) / self.smooth_s)
            self.ema += alpha * (float(evidence) - self.ema)
            self.last_speech_t = t
            keep = max(self.switch_s, self.first_s) + RUN_GAP_S
            self._recent = [(u, e) for u, e in self._recent if u >= t - keep - 1e-9] + [(t, float(evidence))]
            cand = self._agree(t, self.first_s if self.current is None else self.switch_s)
            if cand is not None and cand != self.current:
                if self.current is not None:
                    self.switches += 1
                self.current, switched = cand, True
                self.ema = float(np.mean([e for _, e in self._recent]))  # confidence follows the new person at once
        elif self.current is not None and (self.last_speech_t is None or t - self.last_speech_t > self.hold_s):
            self.current, self.ema, self._recent = None, 0.0, []
        self.last_t = t
        return self.state(t, speech=speech, switched=switched)

    def tick(self, t: float) -> dict:
        """Time passing without new audio (the hold can expire)."""
        return self.update(t, False, None)

    def state(self, t: float | None = None, *, speech: bool = False, switched: bool = False) -> dict:
        if self.current is None:
            state = "unsure" if speech else "listening"
            confidence = 0.0
        else:
            agree = self.ema if self.current == 0 else -self.ema
            confidence = float(np.clip(agree / self.sure, 0.0, 1.0))
            state = "speaking" if speech else "hold"
        return {"speaker": self.names[self.current] if self.current is not None else None, "state": state,
                "confidence": round(confidence, 3), "switched": switched, "t": None if t is None else round(float(t), 3),
                "ema": round(float(self.ema), 3)}


class WhoTracker:
    """A :class:`WhoModel` and :class:`WhoSmoother` fed with audio: :meth:`push` 16 kHz samples, :meth:`analyse` the newest
    window, :meth:`tick` the clock. The server and the offline eval share it."""

    def __init__(self, model: WhoModel, smoother: WhoSmoother | None = None, *, hop_s: float = HOP_S,
                 use_slot: bool = True, use_ecapa: bool = True):
        self.model = model
        self.smoother = smoother or WhoSmoother(model.names)
        self.hop_s, self.use_slot, self.use_ecapa = hop_s, use_slot, use_ecapa
        self.noise = NoiseFloor()
        self.keep = int(max(WINDOW_S, EMBED_S) * SR)
        self.buffer = np.zeros(0, np.float32)
        self.received = 0  # samples pushed so far
        self.analysed_at = -1.0  # audio time of the last analysis
        self.last: dict | None = None

    @property
    def audio_t(self) -> float:
        return self.received / SR

    def push(self, samples) -> None:
        x = np.asarray(samples, np.float32).reshape(-1)
        self.received += len(x)
        self.buffer = np.concatenate([self.buffer, x])[-self.keep:]

    def due(self) -> bool:
        return len(self.buffer) >= int(LEVEL_S * SR) and self.audio_t - self.analysed_at >= self.hop_s

    def analyse(self, t: float | None = None) -> dict:
        """Score the newest window at time ``t`` (default: the audio clock) and feed the smoother."""
        t = self.audio_t if t is None else t
        self.analysed_at = self.audio_t
        return self.apply(t, self.model.score(self.buffer, use_slot=self.use_slot, use_ecapa=self.use_ecapa))

    def apply(self, t: float, score: dict) -> dict:
        """Feed one :meth:`WhoModel.score` result (taken at time ``t``) to the VAD and the smoother."""
        floor = self.noise.update(t, score["level_db"])
        speech = is_speech(score["level_db"], floor)
        state = self.smoother.update(t, speech, score["evidence"] if speech else None)
        self.last = {**state, "speech": speech, "level_db": round(score["level_db"], 1), "noise_db": round(floor, 1),
                     "raw": {"slot": None if score["slot"] is None else round(score["slot"], 3),
                             "ecapa": None if score["ecapa"] is None else round(score["ecapa"], 3),
                             "cos": None if score["cos"] is None else [round(c, 3) for c in score["cos"]]},
                     "match": None if score["cos"] is None else {n: round(match_of(c), 3) for n, c in zip(self.model.names, score["cos"])}}
        return self.last

    def tick(self, t: float) -> dict:
        state = self.smoother.tick(t)
        base = self.last or {"speech": False, "level_db": None, "noise_db": None, "raw": {"slot": None, "ecapa": None, "cos": None},
                             "match": None}
        self.last = {**base, **state}
        return self.last
