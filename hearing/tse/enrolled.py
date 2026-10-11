"""Audio-only extraction by ECAPA speaker-embedding cosine matching."""

from __future__ import annotations

import math
import os
from importlib.util import find_spec

import numpy as np

from hearing import device, models
from hearing.fusion.stream_vad import StreamVad
from hearing.tse import fixed_gate
from hearing.tse.base import SR, register_candidate
from hearing.tse.blind import (
    SEP_CKPT_ENV,
    TEAM_CKPT_ENV,
    TEAM_MODEL,
    BlindSeparator,
    fixed_slot,
    team_slot,
    validate_audio,
)
from hearing.tse.enroll import profile_name, profile_pieces
from hearing.tse.team_head import from_env as team_head_from_env
from hearing.tse.team_head import stream_features

MODEL = "spkrec-ecapa-voxceleb"
REPO = f"speechbrain/{MODEL}"
FILES = (
    "hyperparams.yaml",
    "embedding_model.ckpt",
    "mean_var_norm_emb.ckpt",
    "classifier.ckpt",
    "label_encoder.txt",
)


def download_model():
    """Explicit download only; saved underneath HEARING_MODELS_DIR."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=REPO, local_dir=models.cache_dir() / MODEL)


def _local_source():
    root = models.cache_dir() / MODEL
    if not all((root / name).is_file() for name in FILES):
        raise RuntimeError(
            "ECAPA weights absent; call hearing.tse.enrolled.download_model()"
        )
    return root


def available():
    try:
        _local_source()
    except RuntimeError:
        return False
    return find_spec("speechbrain") is not None


CACHE_ENTRIES = 16  # cached enrollment embeddings (both slots): the saved profiles and the last clips

class MockBackend:
    """Deterministic spectral features for offline wiring, not speaker recognition."""

    device = "cpu"

    def embed(self, audio):
        spectrum = np.abs(np.fft.rfft(audio, n=2048))
        return np.array([np.linalg.norm(band) for band in np.array_split(spectrum, 8)])


class _ECAPABackend:
    def __init__(self, encoder, dev):
        self.encoder, self.device = encoder, dev

    def embed(self, audio):
        import torch

        with torch.inference_mode():
            x = torch.from_numpy(audio).unsqueeze(0).to(self.device)
            return self.encoder.encode_batch(x).detach().cpu().numpy().reshape(-1)


@models.register_loader(MODEL)
def _load_encoder():
    root = _local_source()
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise RuntimeError("ECAPA requires speechbrain from the ml extra") from exc
    dev = device.pick()
    # Override the YAML's remote pretrained_path: all referenced files must already exist.
    encoder = EncoderClassifier.from_hparams(
        source=str(root),
        savedir=str(root),
        overrides={"pretrained_path": str(root)},
        run_opts={"device": dev},
    )
    return _ECAPABackend(encoder, dev)


def _unit_embedding(backend, audio):
    embedding = np.asarray(backend.embed(audio), dtype=np.float64).reshape(-1)
    if not embedding.size or not np.isfinite(embedding).all():
        raise ValueError("speaker encoder returned an invalid embedding")
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 1e-12 else np.zeros_like(embedding)


def _overlap(prev, mix):
    """Offset ``d`` with ``prev[d:] == mix[:len(prev) - d]`` (consecutive sliding windows), else None."""
    for d in np.flatnonzero(prev == mix[0])[:MAX_OFFSETS]:
        n = len(prev) - int(d)
        if MIN_OVERLAP <= n <= len(mix) and np.array_equal(prev[d:], mix[:n]):
            return int(d)
    return None


def _correlation(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


MIN_OVERLAP = 1600  # 100 ms of shared samples needed to follow a stream across windows
MAX_OFFSETS = 64  # sample-equality candidates tried when locating the previous window
PURITY_SEGMENT_S = 0.5  # enrollment is checked in pieces of this length
MARGIN = 0.1  # T-061: cosine-score lead a stream needs to take over from the one being followed
GATED_PLAY_ENV = "HEARING_GATED_PLAY"  # T-180 K5: "stream" = a gated two-profile window plays its best stream
SAME_VOICE = 0.7  # a negative enrollment this close to the target's is not a different person
# T-370 soft gate (knob, default off): a gated two-profile window whose streams overlap (both speech-active by
# hearing.fusion.stream_vad, or score margin < the hold margin) plays its best stream at -(1 - p) * D dB instead
# of silence (Cohen & Berdugo's G = G1^p * Gmin^(1 - p) in dB). Run it with HEARING_GATED_FLOOR_DB=20 so the
# mixer ducks the room 20 dB in those windows (confidence stays 0).
SOFT_ENV = "HEARING_GATED_SOFT"  # "1" = on; unset / "0" = off (hard gate as before)
SOFT_DB_ENV = "HEARING_GATED_SOFT_DB"  # D: the stream's attenuation at p = 0 in dB (Gmin = 10^(-D/20))
SOFT_DB = 12.0
SOFT_SPAN = 0.5  # p = 1 + (score - contrast) / SOFT_SPAN, clipped to [0, 1] (hand-set: false-gate picks sit at -0.01..-0.45)
# T-484 slot soft cap (knob, default off): with a fixed-order / team slot the slot itself is reliable, so a gated
# window whose target slot carries speech plays that slot instead of silence: at 0 dB, or SLOT_DUCK dB down when the
# partner slot is louder by more than SLOT_PARTNER_DB (the other teammate talking). Silence in the slot stays muted.
# The window keeps confidence 0, so the mixer still ducks the room (HEARING_GATED_FLOOR_DB). Live test Oct 11: the
# 20 dB gated drop on every flip was the loud/quiet pumping.
SLOT_DUCK_ENV = "HEARING_SLOT_DUCK_DB"  # e.g. 6: the most a gated slot with speech is ducked; unset = off (mute)
SLOT_PARTNER_DB = 3.0  # partner slot louder than the target slot by this much = partner dominates
SLOT_ACTIVE_DB = -20.0  # target slot RMS at least this far under the window mix RMS = it has speech energy
SLOT_MIN_RMS = 1e-4  # and at least -80 dBFS
# T-487 locked mode (default on): with the team separator and a saved teammate in HEARING_FIXED_SLOTS the slot IS
# that person, so every window plays the target slot (silence included) with no per-window ECAPA gate, confidence 1
# (the mixer ducks the room and the partner by the room dial). Only a partner slot louder than the target slot by
# SLOT_PARTNER_DB for more than LOCK_PARTNER_S ducks the target, at most LOCK_DUCK_DB, moving LOCK_DUCK_RATE dB/s.
LOCK_ENV = "HEARING_TEAM_LOCK"  # "0" = off (the T-386/T-484 gated fixed slot as before)
LOCK_PARTNER_S = 3.0
LOCK_DUCK_DB = 6.0
LOCK_DUCK_RATE = 6.0  # dB per second of hops, so the full duck (or release) takes 1 s
CONTRAST_ENV = "HEARING_ENROLLED_CONTRAST"  # T-178: default ``contrast`` when the pipeline spec sets none
WARM_ENV = "HEARING_WARM_ENROLLED"  # T-339: "1" = saved-profile embeddings computed once at server start, shared
# T-339: (embedder id, clip key) -> (embedding, profile pieces info), filled by hearing.server.warm (knob on only)
WARM_PROFILES: dict[tuple, tuple] = {}


def soft_db() -> float | None:
    """``D`` (dB) when ``$HEARING_GATED_SOFT`` is on (T-370), else None."""
    if os.environ.get(SOFT_ENV, "").strip() not in ("1", "on", "true"):
        return None
    try:
        d = float(os.environ.get(SOFT_DB_ENV, "") or SOFT_DB)
    except ValueError:
        d = SOFT_DB
    return d if math.isfinite(d) and d >= 0 else SOFT_DB


def slot_duck_db() -> float | None:
    """The slot soft cap in dB when ``$HEARING_SLOT_DUCK_DB`` is set (T-484), else None."""
    raw = os.environ.get(SLOT_DUCK_ENV, "").strip()
    if not raw:
        return None
    try:
        d = float(raw)
    except ValueError:
        return None
    return d if math.isfinite(d) and d >= 0 else None


def lock_enabled() -> bool:
    """T-487 locked mode for team slots: on unless ``$HEARING_TEAM_LOCK`` is 0/off/false."""
    return os.environ.get(LOCK_ENV, "").strip().lower() not in ("0", "off", "false")


def partner_dominates(slot, partner, mix) -> bool:
    """The partner slot carries speech energy and is louder than the target slot by more than SLOT_PARTNER_DB."""
    def rms(x):
        return float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64))))) if len(x) else 0.0

    rs, rp, rm = rms(slot), rms(partner), rms(mix)
    if rp < max(SLOT_MIN_RMS, rm * 10.0 ** (SLOT_ACTIVE_DB / 20.0)):
        return False
    return rp > rs * 10.0 ** (SLOT_PARTNER_DB / 20.0)


def slot_gain_db(slot, partner, mix, cap_db: float) -> float | None:
    """Gain (dB, <= 0) for a gated fixed slot under the T-484 soft cap: None when the slot has no speech energy
    (keep it muted), ``-cap_db`` when the partner slot is louder by more than SLOT_PARTNER_DB, else 0."""
    def rms(x):
        return float(np.sqrt(np.mean(np.square(np.asarray(x, np.float64))))) if len(x) else 0.0

    rs, rp, rm = rms(slot), rms(partner), rms(mix)
    if rs < max(SLOT_MIN_RMS, rm * 10.0 ** (SLOT_ACTIVE_DB / 20.0)):
        return None
    return -float(cap_db) if rp > rs * 10.0 ** (SLOT_PARTNER_DB / 20.0) else 0.0


def soft_gain_db(score: float, contrast: float, d: float, p: float | None = None) -> float:
    """Gain (dB, <= 0) of the best stream in a soft-gated window: ``-(1 - p) * d``; ``p`` from the team head when
    given, else ``1 + (score - contrast) / SOFT_SPAN`` (a gated pick scores under ``contrast``)."""
    if p is None:
        p = 1.0 + (float(score) - float(contrast)) / SOFT_SPAN
    p = min(max(float(p), 0.0), 1.0) if math.isfinite(p) else 0.0
    return -(1.0 - p) * float(d)


def fixed_target(enroll, n_streams: int) -> int | None:
    """T-386: the stream index to play when ``$HEARING_PICK=fixed`` and ``$HEARING_SEP_CKPT`` (a fixed-order
    fine-tuned separator, T-374) are both set and ``enroll`` is a saved profile with a slot; else None."""
    if not os.environ.get(SEP_CKPT_ENV, "").strip():
        return None
    slot = fixed_slot(profile_name(enroll))
    return slot if slot is not None and slot < n_streams else None


def warm_enabled() -> bool:
    """``$HEARING_WARM_ENROLLED`` is ``1`` (T-339); unset keeps per-candidate profile embeddings."""
    return os.environ.get(WARM_ENV, "").strip() == "1"


PAUSE_ACTIONS = ("hold", "duck", "loud")
# T-134 per-window confidence (``extract`` sets ``confidence``; the tse stage passes it to the mixer):
# CONF_MARGIN*: fitted by eval/confidence_fit.py (97 % of picks right above it) on the male-male GRID stand-in
# pp-mm (s1 vs s2, Q-348), not on the teammates' takes yet: refit with --team when they land.
CONF_MARGIN = 0.54  # score lead (cosine units) for full confidence; 0 lead -> 0
CONF_MARGIN_LDA = 0.58  # same, in closed-set LDA units (target centroid +1, other centroid -1)
LDA_HOLD_SCALE = 4.0  # MARGIN (cosine) -> the hold margin in LDA units
LDA_SHRINK = 0.5  # pooled within-class covariance shrunk this far toward its mean variance
F0_LO, F0_HI = 60.0, 400.0  # Hz searched by the autocorrelation pitch tracker
F0_VOICED = 0.5  # a stream (or clip) needs this share of voiced frames for a median F0
F0_VOICING = 0.5  # normalised autocorrelation peak for a voiced 40 ms frame


def f0_frames(audio, sr=SR, frame_s=0.04, hop_s=0.02, loud_only=False):
    """Per-frame F0 in Hz by normalised autocorrelation (NaN = unvoiced or silent; ``loud_only``:
    silent frames dropped, so NaN = an unvoiced loud frame)."""
    x = np.asarray(audio, np.float64)
    n, h = round(frame_s * sr), round(hop_s * sr)
    if len(x) < n:
        return np.zeros(0)
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::h]
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = np.sum(frames**2, axis=1)
    loud = energy > max(1e-8 * n, 1e-3 * float(np.max(energy, initial=0.0)))
    spec = np.fft.rfft(frames * np.hanning(n), n=2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    lo, hi = int(sr / F0_HI), min(int(sr / F0_LO), n - 1)
    norm = np.maximum(ac[:, :1], 1e-12)
    seg = ac[:, lo:hi] / norm
    lag = lo + np.argmax(seg, axis=1)
    peak = seg[np.arange(len(seg)), lag - lo]
    f0 = np.where(loud & (peak >= F0_VOICING), sr / lag, np.nan)
    return f0[loud] if loud_only else f0


def median_f0(audio, sr=SR):
    """Median F0 of a stream with at least ``F0_VOICED`` voiced frames among its loud ones, else None."""
    f0 = f0_frames(audio, sr, loud_only=True)
    voiced = f0[np.isfinite(f0)]
    if len(voiced) < 5 or len(voiced) < F0_VOICED * len(f0):
        return None
    return float(np.median(voiced))


def f0_range(pieces, sr=SR):
    """``(p10, p90)`` of a profile's voiced F0 frames, None with fewer than 20 voiced frames."""
    f0 = np.concatenate([f0_frames(p, sr) for p in pieces]) if pieces else np.zeros(0)
    f0 = f0[np.isfinite(f0)]
    return None if len(f0) < 20 else (float(np.percentile(f0, 10)), float(np.percentile(f0, 90)))


def _inside(f0, rng):
    return f0 is not None and rng is not None and rng[0] <= f0 <= rng[1]


def fit_lda(target_embs, other_embs, shrink=LDA_SHRINK):
    """2-class LDA over profile segment embeddings: ``(w, b)`` with ``w @ e - b`` = +1 at the target
    centroid and -1 at the other's (pooled within-class covariance, shrunk toward its mean variance)."""
    t, o = np.asarray(target_embs, np.float64), np.asarray(other_embs, np.float64)
    mt, mo = t.mean(axis=0), o.mean(axis=0)
    centred = np.concatenate([t - mt, o - mo])
    cov = centred.T @ centred / max(len(centred) - 2, 1)
    d = cov.shape[0]
    var = float(np.trace(cov)) / d if np.trace(cov) > 1e-12 else 1e-3
    cov = (1.0 - shrink) * cov + shrink * var * np.eye(d)
    w = np.linalg.solve(cov, mt - mo)
    half = float(w @ (mt - mo)) / 2.0
    if half <= 1e-12:
        return None
    w = w / half
    return w, float(w @ (mt + mo)) / 2.0


@register_candidate("enrolled")
class EnrolledCandidate:
    """Separate the window into two streams and play the one whose ECAPA embedding matches the enrollment.

    Association guards (T-061, ``docs/decisions/association.md``), all absolute cosine numbers.
    Defaults are the measured pick: ``clean_enroll`` on, ``margin`` 0.1, ``contrast`` 0 with the
    stage's negative enrollment; ``floor`` and ``purify`` off (they cost accuracy / gained nothing).

    * ``margin``: when the best stream beats the other by less than this, keep following the stream
      played in the previous window instead (matched by waveform correlation over the windows'
      shared samples), so an ambiguous window never flips the talker.
    * ``floor``: when even the best stream is below this similarity, the target is not talking in
      the window: return silence (the mixer then plays only the attenuated residual).
    * ``purify``: before embedding the enrollment, cut it into ``PURITY_SEGMENT_S`` pieces and drop
      every piece whose embedding has a cosine below this with the mean of the other pieces (speech
      of someone else that slipped past the ASD gate). Cached per enrollment clip.
    * ``clean_enroll``: run the enrollment clip through the same separator and embed its louder
      stream, so the reference and the window streams are embedded under the same conditions (the
      live clip is cut from the noisy room mix, the windows' streams are separated).
    * ``other`` (the stage's negative enrollment, when the enroller collects one): streams are
      ranked by ``cos(target) - cos(other)`` instead of ``cos(target)``, and a window whose chosen
      stream is not at least ``contrast`` closer to the target than to the other people returns
      silence. A negative that is itself too close to the target (``cos > same_voice``, default
      ``SAME_VOICE``: a polluted clip) is ignored. ``contrast`` None reads ``$HEARING_ENROLLED_CONTRAST``
      (else 0; T-178 sweep, the team preset's ``enrolled_contrast``); ``pair_cos`` / ``negative_active``
      log cos(target, negative) and whether the negative was used.
    * pause handling (T-069): a window is a **pause** when no stream matches (best
      ``cos(target) < pause_floor``) and, with ``pause_p`` set, the camera agrees the target is quiet
      (the target track's mean ``p_speaking`` over the window from ``asd=`` is below ``pause_p``, or
      no target face is visible). In a pause, ``pause_action="hold"`` keeps playing the stream that
      continues the previous window's choice (like ``margin``, but whatever the scores say) for at
      most ``pause_max_s`` of consecutive pause hops; ``"duck"`` returns silence so the mixer plays
      only the attenuated room, as the gate tier does; ``"loud"`` plays the stream with the most energy
      (the separator's speech stream when the other person is silent; no ``pause_max_s`` cap). A window
      that matches resets the count, and the ``margin`` rule still decides confident switches.
    """

    name = "enrolled"
    licence = "Apache-2.0"
    needs = frozenset({"enroll", "other"})

    def __init__(self, separator=None, embedder=None, margin=MARGIN, floor=None, purify=None,
                 clean_enroll=True, contrast=None, pause_floor=None, pause_p=None, pause_action="hold",
                 pause_max_s=2.0, same_voice=SAME_VOICE, closed_set=True, f0_veto=True, team_separator=None):
        self.separator = separator if separator is not None else BlindSeparator()
        # T-465: the fine-tuned fixed-order separator for saved team profiles ($HEARING_SEP_TEAM_CKPT; None = off)
        if team_separator is None and os.environ.get(TEAM_CKPT_ENV, "").strip():
            team_separator = BlindSeparator(model=TEAM_MODEL)
        self.team_separator = team_separator
        self.embedder = embedder
        self.device = self.separator.device
        self.margin = float(margin)
        self.floor = None if floor is None else float(floor)
        self.purify = None if purify is None else float(purify)
        self.clean_enroll = bool(clean_enroll)
        if contrast is None:  # T-178: the sweep sets it per run; unset = 0 (the measured default)
            contrast = os.environ.get(CONTRAST_ENV) or 0.0
        self.contrast = float(contrast)
        self.same_voice = float(same_voice)
        if pause_action not in PAUSE_ACTIONS:
            raise ValueError(f"pause_action must be one of {PAUSE_ACTIONS}, got {pause_action!r}")
        self.pause_floor = None if pause_floor is None else float(pause_floor)
        self.pause_p = None if pause_p is None else float(pause_p)
        self.pause_action = pause_action
        self.pause_max_s = float(pause_max_s)
        if self.pause_floor is not None and self.pause_p is not None:
            self.needs = type(self).needs | {"asd"}  # the stage then passes asd= (no lip feed needed)
        # (slot, clip key) -> (embedding, kept share, profile pieces or None); several entries (T-136): the
        # saved profiles are prepared at hello, so swapping between them is a cache hit, not a ~1 s ECAPA job
        self._cache: dict[tuple, tuple] = {}
        self._pieces: dict[str, tuple] = {}  # slot -> (clip key, profile piece embeddings, F0 range)
        self._lda: tuple | None = None  # ((target key, other key), (w, b) or None)
        self.closed_set = closed_set
        self.f0_veto = f0_veto
        self.team_head = team_head_from_env()  # T-336: None unless $HEARING_TEAM_HEAD names saved weights
        self.kept_share = None  # share of enrollment pieces kept by the purity filter
        self.reset()

    def reset(self):
        self.similarity = None
        self.similarities = None
        self.selected_stream = None
        self.held = False  # last window followed the previous stream (margin)
        self.gated = False  # last window returned silence (floor / contrast)
        self.scores = None  # per stream: cos(target) - cos(other), or cos(target) without a negative
        self.paused = False  # last window was a pause (pause_floor) and was held or ducked
        self.confidence = None  # T-134: 0 (unsure: play the mix) .. 1 (sure: duck the residual deep)
        self.margin_score = None  # scores[0] - scores[1] of the last window
        self.vetoed = False  # last window's pick was overturned by the F0 cue
        self.stream_f0 = None  # median F0 per stream (None = under-voiced)
        self.scoring = "cosine"  # or "lda" (closed set over two saved profiles)
        self.pair_cos = None  # T-178: cos(target, negative) of the enrollments (None without a negative)
        self.negative_active = False  # the negative was used (pair_cos <= same_voice)
        self.head_p = None  # T-336: the team head's p(target dominates) per stream (None when it is off)
        self._pause_s = 0.0  # consecutive pause time held so far
        self._prev = None  # (mix, chosen stream) of the previous window
        self.soft_gated = False  # T-370: last gated window played its best stream at soft_gain_db
        self.soft_gain_db = None
        self.slot_soft_db = None  # T-484: gain of the last gated window's slot under HEARING_SLOT_DUCK_DB (None = muted/off)
        self._soft_vad = StreamVad()  # running noise floor for the soft gate's overlap cue
        self.fixed_pick = False  # T-386: last window played the fine-tuned checkpoint's fixed slot
        self.separator_active = None  # T-465: "stock" / "team", the separator of the last window (None before one)
        self._fixed_gate = None  # T-402: hearing.tse.fixed_gate.FixedGate while HEARING_FIXED_GATE=1
        self.fixed_gate_probs = None  # T-402: last window's p(target, partner, none) of the fixed-slot gate
        self.fixed_gate_stranger = False  # T-402: last window's target slot matched neither profile
        self.fixed_gate_swapped = False  # T-402: last window played the partner slot (swap rescue)
        self.locked = False  # T-487: last window played the team slot in locked mode (no identity gate)
        self.lock_duck_db = 0.0  # T-487: the locked slot's gain (0 .. -LOCK_DUCK_DB), down while the partner dominates
        self._partner_s = 0.0  # T-487: consecutive time the partner slot has dominated
        self._lock_key = None  # T-487: the target the lock belongs to (a new pick resets the duck)
        self._lock_hop = None  # T-487: the last measured hop in samples

    def load(self):
        self.separator.load()
        if self.team_separator is not None:  # T-465: both separators load at start, so a switch never waits
            self.team_separator.load()
        if self.embedder is None:
            self.embedder = models.load(MODEL)
        self.device = self.separator.device

    def prepare(self, enroll, other=None, sr=SR):
        """Compute and cache the embeddings :meth:`extract` needs for this enrollment (the ``tse``
        stage calls it in its own job when the clip changes, T-088), exactly as extract derives them."""
        self.load()
        self.target_embedding(validate_audio(enroll, sr))
        if other is not None:
            self.other_embedding(validate_audio(other, sr))

    def target_embedding(self, enroll):
        """Unit ECAPA embedding of the enrollment (separated / purity-filtered as configured)."""
        return self._clip_embedding(enroll, "target")

    def other_embedding(self, other):
        """Unit ECAPA embedding of the negative enrollment, same treatment as the target's."""
        return self._clip_embedding(other, "other")

    def _clip_embedding(self, clip, slot):
        key = (len(clip), hash(clip.tobytes()))
        cached = self._cache.get((slot, key))
        if cached is not None:
            if slot == "target":
                self.kept_share = cached[1]
            if cached[2] is not None:  # a saved profile: its pieces drive the closed set and the F0 cue
                self._pieces[slot] = cached[2]
            else:
                self._pieces.pop(slot, None)
            return cached[0]
        pieces = profile_pieces(clip)
        if pieces is not None:  # a voice profile (T-125): clean pieces, mean of unit embeddings, no separator
            shared = (id(self.embedder), key) if warm_enabled() else None
            if shared in WARM_PROFILES:  # T-339: embedded once at server start by this same encoder
                embedding, info = WARM_PROFILES[shared]
                self._pieces[slot] = info
            else:
                units = [_unit_embedding(self.embedder, p) for p in pieces]
                self._pieces[slot] = info = (key, np.stack(units), f0_range(pieces))
                mean = np.mean(units, axis=0)
                norm = np.linalg.norm(mean)
                embedding = mean / norm if norm > 1e-12 else mean
                if shared is not None:
                    WARM_PROFILES[shared] = (embedding, info)
            if slot == "target":
                self.kept_share = 1.0
            self._remember(slot, key, embedding, 1.0, info)
            return embedding
        self._pieces.pop(slot, None)
        if self.clean_enroll:
            streams = self.separator.separate(clip, SR)
            clip = streams[int(np.argmax([np.sum(np.square(y, dtype=np.float64)) for y in streams]))]
        seg = round(PURITY_SEGMENT_S * SR)
        n = len(clip) // seg
        kept = 1.0
        if self.purify is not None and n >= 3:
            pieces = [clip[i * seg : (i + 1) * seg] for i in range(n)]
            emb = np.stack([_unit_embedding(self.embedder, p) for p in pieces])
            total = emb.sum(axis=0)
            keep = []
            for i in range(n):
                rest = total - emb[i]
                norm = np.linalg.norm(rest)
                keep.append(norm < 1e-12 or float(np.dot(emb[i], rest / norm)) >= self.purify)
            if sum(keep) >= 2:
                clip = np.concatenate([p for p, k in zip(pieces, keep, strict=True) if k])
                kept = sum(keep) / n
        if slot == "target":
            self.kept_share = kept
        embedding = _unit_embedding(self.embedder, clip)
        self._remember(slot, key, embedding, kept)
        return embedding

    def _remember(self, slot, key, embedding, kept, pieces=None) -> None:
        self._cache[(slot, key)] = (embedding, kept, pieces)
        while len(self._cache) > CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))

    def _closed_set(self):
        """``(w, b)`` of the LDA over both saved profiles' piece embeddings, None otherwise."""
        t, o = self._pieces.get("target"), self._pieces.get("other")
        if not self.closed_set or t is None or o is None or len(t[1]) < 2 or len(o[1]) < 2:
            return None
        key = (t[0], o[0])
        if self._lda is None or self._lda[0] != key:
            self._lda = (key, fit_lda(t[1], o[1]))
        return self._lda[1]

    def _f0_pick(self, streams, sr):
        """``(index, strong)`` of the one stream whose median F0 lies in the target profile's range
        (not the other's) while the other stream's does not; ``strong`` when that one lies in the
        negative profile's range. None when the cue is off or undecided. Sets ``stream_f0``."""
        t, o = self._pieces.get("target"), self._pieces.get("other")
        self.stream_f0 = None
        if not self.f0_veto or t is None or o is None or t[2] is None or o[2] is None or len(streams) != 2:
            return None
        rt, ro = t[2], o[2]
        if not (rt[1] < ro[0] or ro[1] < rt[0]):
            return None  # overlapping pitch ranges: the cue cannot tell them apart
        self.stream_f0 = f = [median_f0(y, sr) for y in streams]
        for i in (0, 1):
            if _inside(f[i], rt) and not _inside(f[i], ro) and not _inside(f[1 - i], rt):
                strong = _inside(f[1 - i], ro)  # both streams placed: the other voice is the other profile's
                return i, strong
        return None

    def _follow(self, mix, streams):
        """Index of the stream continuing the previous window's choice, or None."""
        found = self._follow_hop(mix, streams)
        return None if found is None else found[0]

    def _follow_hop(self, mix, streams):
        """``(stream index, hop in samples)`` continuing the previous window's choice, or None."""
        if self._prev is None:
            return None
        prev_mix, prev_out = self._prev
        d = _overlap(prev_mix, mix)
        if d is None:
            return None
        n = len(prev_mix) - d
        return int(np.argmax([_correlation(prev_out[d:], y[:n]) for y in streams])), d

    def team_target(self, enroll, other=None) -> int | None:
        """T-465: the team checkpoint's slot for this window's target, or None (stock separator, ECAPA pick):
        on only with a team separator, for a saved profile named in ``$HEARING_FIXED_SLOTS`` whose partner,
        when it is a saved profile too, is the other slot (:func:`hearing.tse.blind.team_slot`)."""
        if self.team_separator is None:
            return None
        return team_slot(profile_name(enroll), None if other is None else profile_name(other))

    def _is_pause(self, similarities, asd):
        if self.pause_floor is None or max(similarities) >= self.pause_floor:
            return False
        if self.pause_p is None:
            return True
        if asd is None:
            return False  # pause_p set but no camera evidence passed: treat as unsure, not quiet
        p = np.asarray(asd, np.float64)
        p = p[np.isfinite(p)]
        return not p.size or float(p.mean()) < self.pause_p

    def extract(
        self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None, other=None, asd=None
    ):
        mix = validate_audio(mix, sr)
        if enroll is None:
            raise ValueError("enrolled needs enrollment audio")
        enroll = validate_audio(enroll, sr)
        self.load()
        target = self.target_embedding(enroll)
        negative = None
        if other is not None:
            negative = self.other_embedding(validate_audio(other, sr))
            self.pair_cos = float(np.dot(target, negative))
            if self.pair_cos > self.same_voice:
                negative = None
        else:
            self.pair_cos = None
        self.negative_active = negative is not None
        team = self.team_target(enroll, other)
        self.separator_active = "stock" if team is None else "team"
        streams = (self.separator if team is None else self.team_separator).separate(mix, sr)
        embeddings = [_unit_embedding(self.embedder, y) for y in streams]
        similarities = [float(np.dot(target, e)) for e in embeddings]
        scores = similarities if negative is None else [
            s - float(np.dot(negative, e)) for s, e in zip(similarities, embeddings, strict=True)
        ]
        lda = self._closed_set() if negative is not None else None
        self.scoring = "lda" if lda is not None else "cosine"
        if lda is not None:  # closed set: both voices are saved profiles
            w, b = lda
            scores = [float(w @ e) - b for e in embeddings]
        head_p = None
        if lda is not None and self.team_head is not None:  # T-336: scores become head log-odds vs threshold
            x = stream_features(scores, similarities, [float(np.dot(negative, e)) for e in embeddings],
                                [float(np.sum(np.square(y, dtype=np.float64))) for y in streams])
            head_p = self.team_head.prob(x)
            scores = [float(v) for v in self.team_head.scores(x)]
        self.head_p = None if head_p is None else [float(v) for v in head_p]
        self.similarities, self.scores = similarities, scores
        choice = int(np.argmax(scores))
        ranked = sorted(scores, reverse=True)
        margin = ranked[0] - ranked[1] if len(ranked) > 1 else math.inf
        full = CONF_MARGIN_LDA if lda is not None else CONF_MARGIN
        hold = self.margin * (LDA_HOLD_SCALE if lda is not None else 1.0)
        self.margin_score = None if math.isinf(margin) else margin
        conf = 1.0 if math.isinf(margin) else min(max(margin / full, 0.0), 1.0)
        self.held = self.vetoed = False
        if team is not None and team < len(streams) and lock_enabled():
            return self._locked(mix, sr, streams, team, similarities, enroll)
        self.locked, self._lock_key = False, None
        fixed = team if team is not None and team < len(streams) else fixed_target(enroll, len(streams))
        self.fixed_pick = fixed is not None
        if fixed is not None:  # T-386: the slot is the identity; ECAPA only sets the gate and confidence
            choice = fixed
            rest = [v for i, v in enumerate(scores) if i != fixed]
            lead = scores[fixed] - max(rest) if rest else math.inf
            conf = 1.0 if math.isinf(lead) else min(max(lead / full, 0.0), 1.0)
        pick = None if fixed is not None else self._f0_pick(streams, sr)
        f0 = None if pick is None else pick[0]
        if f0 is not None and f0 != choice and (pick[1] or margin < hold):  # veto, or tiebreak in the margin
            choice, self.vetoed, conf = f0, True, min(conf, 0.5)
        elif f0 is not None and f0 == choice:
            conf = min(1.0, conf + 0.25)  # pitch agrees with the embeddings
        elif fixed is None and hold > 0 and margin < hold:
            follow = self._follow(mix, streams)
            if follow is not None and lda is not None and scores[follow] < self.contrast:
                follow = None  # closed set: the followed stream sits on the other profile's side, not held
            if follow is not None:
                self.held = follow != choice
                choice = follow
        self.paused = duck = False
        if fixed is not None:
            self._pause_s = 0.0
        elif self._is_pause(similarities, asd) and self.pause_action == "loud":
            loud = int(np.argmax([float(np.mean(np.square(y, dtype=np.float64))) for y in streams]))
            self.paused, self.held = True, self.held or loud != choice
            choice = loud
        elif self._is_pause(similarities, asd):
            found = self._follow_hop(mix, streams)
            hop_s = (found[1] if found is not None else len(mix)) / sr
            if self._pause_s + hop_s <= self.pause_max_s + 1e-9:
                if self.pause_action == "duck":
                    self.paused = duck = True
                elif found is not None:
                    self.paused = True
                    self.held = self.held or found[0] != choice
                    choice = found[0]
            self._pause_s += hop_s
        else:
            self._pause_s = 0.0
        self.selected_stream = choice
        self.similarity = similarities[choice]
        self._prev = (mix, streams[choice])
        self.gated = duck or (self.floor is not None and max(similarities) < self.floor) or (
            negative is not None and scores[choice] < self.contrast
        )
        if head_p is not None:
            conf = float(head_p[choice])  # T-336: the calibrated probability replaces the margin confidence
        if fixed is not None and negative is not None and len(streams) == 2 and fixed_gate.enabled():
            # T-402: an active target slot is played even when ECAPA scores it under the contrast (Joseph's slot was
            # muted 4.6-8.2 s per scene); a slot matching neither profile (stranger) stays muted
            if self._fixed_gate is None:
                self._fixed_gate = fixed_gate.FixedGate()
            play, gate_conf = self._fixed_gate.step(streams[fixed], streams[1 - fixed], mix, similarities[fixed],
                                                    float(np.dot(negative, embeddings[fixed])))
            partner_lead = scores[1 - fixed] - scores[fixed]
            if self.gated and play and not duck and partner_lead <= fixed_gate.RESCUE_MAX_PARTNER_LEAD:
                # rescue only: the slot un-mutes what the contrast gate muted, unless the partner slot sounds more like
                # the target than the target slot does (the other teammate is talking)
                self.gated, conf = False, gate_conf
            elif self.gated and not duck and self._fixed_gate.swap(scores[1 - fixed]):
                choice = 1 - fixed  # swap rescue: the net routed the target into the partner slot this window
                self.selected_stream, self.similarity, self._prev = choice, similarities[choice], (mix, streams[choice])
                self.gated, conf = False, fixed_gate.MIN_CONF
            self.fixed_gate_probs = self._fixed_gate.probs
            self.fixed_gate_stranger = self._fixed_gate.stranger
            self.fixed_gate_swapped = self._fixed_gate.swapped
        self.confidence = 0.0 if self.gated else conf
        self.soft_gated, self.soft_gain_db = False, None
        self.slot_soft_db = None
        cap = slot_duck_db()
        if cap is not None and self.gated and not duck and fixed is not None and len(streams) == 2:
            g = slot_gain_db(streams[fixed], streams[1 - fixed], mix, cap)
            if g is not None:
                self.slot_soft_db = g
                return (streams[fixed] * np.float32(10.0 ** (g / 20.0))).astype(np.float32)
        d = soft_db()
        if d is not None and negative is not None and len(streams) == 2:
            overlap = bool(self._soft_vad.window(mix, streams)["overlap"])  # always fed: keeps the noise floor running
            if self.gated and not duck and (overlap or margin < hold):  # cannot-link: one active stream is the target
                p = None if head_p is None else float(head_p[choice])
                self.soft_gated = True
                self.soft_gain_db = soft_gain_db(scores[choice], self.contrast, d, p)
                return (streams[choice] * np.float32(10.0 ** (self.soft_gain_db / 20.0))).astype(np.float32)
        if self.gated and not duck and negative is not None and os.environ.get(GATED_PLAY_ENV, "").strip() == "stream":
            return streams[choice].copy()  # K5 (T-180): two-profile mode plays the best stream, confidence 0
        if self.gated:
            return np.zeros_like(streams[choice])
        return streams[choice].copy()

    def _locked(self, mix, sr, streams, slot, similarities, enroll):
        """T-487: play the team slot every window (pauses too, no ECAPA gate), confidence 1; duck it smoothly by at
        most LOCK_DUCK_DB only after the partner slot has dominated for more than LOCK_PARTNER_S."""
        key = (slot, len(enroll), profile_name(enroll))
        if key != self._lock_key:  # a new pick (or the first window): fresh lock, no carried duck
            self._lock_key, self._partner_s, self.lock_duck_db = key, 0.0, 0.0
        found = self._follow_hop(mix, streams) if self._prev is not None else None
        if found is not None and found[1] > 0:  # the hop is measured from the sample-equal overlap with the last window
            self._lock_hop = found[1]
        hop_s = (self._lock_hop or len(mix) // 2) / sr  # unknown (first window, identical windows): half a window
        partner = streams[1 - slot] if len(streams) == 2 else np.zeros_like(streams[slot])
        self._partner_s = self._partner_s + hop_s if partner_dominates(streams[slot], partner, mix) else 0.0
        goal = -LOCK_DUCK_DB if self._partner_s > LOCK_PARTNER_S else 0.0
        step = LOCK_DUCK_RATE * hop_s
        self.lock_duck_db = float(min(max(goal, self.lock_duck_db - step), self.lock_duck_db + step))
        self.locked = self.fixed_pick = True
        self.gated = self.paused = self.soft_gated = False
        self.soft_gain_db = self.slot_soft_db = None
        self._pause_s = 0.0
        self.confidence = 1.0
        self.selected_stream = slot
        self.similarity = similarities[slot]
        self._prev = (mix, streams[slot])
        out = streams[slot]
        if self.lock_duck_db < 0.0:
            return (out * np.float32(10.0 ** (self.lock_duck_db / 20.0))).astype(np.float32)
        return out.copy()
