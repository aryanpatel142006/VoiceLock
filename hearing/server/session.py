"""One live session, independent of the transport: bytes/JSON in, bytes/JSON out.

The WebSocket handler in ``hearing.server.app`` feeds client messages to
:meth:`Session.handle_text` / :meth:`Session.handle_binary`, sends whatever they
return, and calls :meth:`Session.telemetry` at ~10 Hz.

Audio path per client frame: PCM16 at the client rate -> float32 -> (stream resample
to 16 kHz) -> 10 ms hops -> the pipeline of the current mode -> (resample back) ->
PCM16 frame with the input frame's ``seq`` and ``capture_ts`` so the client can
correlate round-trip latency.

Synthetic and Text modes end in the ``synth`` stage: after every audio message its
captions are drained into ``caption`` events. The session owns a
:class:`~hearing.synth.voice_match.VoiceMatcher`: the enhanced target audio (what enters
``synth``) feeds the voice estimator while a target track is set, ``on_face`` feeds the
face estimator, and every target change or ``voice_override`` re-picks the synth voice.
Estimates stay inside the matcher; only the chosen bucket reaches telemetry.

Quiet (T-149) plays digital silence and captions only the selected person: it runs the very
pipeline Focus HQ plays (same spec, same instance, so the target, voice profile, enrollment and the
ladder's isolating tier survive a Focus <-> Quiet toggle untouched) and taps its output into a
session-owned captions-only ``synth(speak=false)`` (:meth:`Session._apply_quiet`). The output gain
ramps to 0 over :data:`QUIET_FADE_MS`; the tap is fed only while the playing tier isolates the target
(:data:`ISOLATING_TIERS`), else silence, and telemetry ``quiet`` says which (``feed`` focus|paused).
``HEARING_PIPELINES="quiet=..."`` keeps the old behaviour (that spec plays, no captions).

Video frames go to a :class:`~hearing.server.vision.VisionEngine` (decode and detection on
a worker thread). Finished frames are applied between audio messages: they set
``ctx.tracks`` (with ``p_speaking``) and ``ctx.target``, feed ``on_face`` and AND the
voice estimator's energy VAD with the target's ``p_speaking``. The engine also cuts every
visible face's mouth crop into the session's :class:`~hearing.tse.lips.LipFeed`
(``ctx.telemetry.extras["lip_feed"]``), which Focus HQ's ``tse`` stages read.

Focus HQ (the ``focus_hq`` mode) is the ``tiers`` stage by default: the fallback ladder of
:mod:`hearing.fusion.tiers` over ``config/tiers.yaml`` (AV-TSE -> enrolled -> gate ->
audio-only). ``tier`` in telemetry is what the active pipeline reports (the ladder or a
``remote`` stage set ``ctx.telemetry.tier``); for other pipelines it is ``audio-only`` without
live video, ``gate`` when the pipeline has a gate stage, else ``av``. The ladder's state goes out
as ``tiers`` (:meth:`hearing.fusion.tiers.TierLadder.status`).

Where Focus HQ runs is ``hq_backend`` (``control.hq_backend``, ``ServerConfig.hq_backend``):
``local`` plays the configured ``focus_hq`` pipeline; ``remote`` plays
:attr:`ServerConfig.hq_remote`, a ``remote`` stage (``HEARING_REMOTE_URL``) whose fallback is
that same local pipeline, so a dead or stalled remote degrades to local inside the stage within
``stall_ms`` (500 ms); ``auto`` plays the remote while its ``/health`` is green and the RTT is
below 150 ms (:func:`hearing.server.remote.auto_rule`, checked every hop against a background
:class:`~hearing.server.remote.HealthMonitor`), else local. A backend change is an ordinary
pipeline switch (below). Camera frames go to the remote session while it plays. Telemetry:
``hq_backend`` (requested), ``hq_active`` (what Focus HQ plays now), ``hq_reason`` and
``remote_rtt_ms`` (stream RTT p50 while the remote plays, else the ``/health`` RTT).

Pipeline switches (mode changes, Auto, HQ backend, latency preset) are make-before-break: the old
pipeline keeps playing while the new one fills its algorithmic delay (1.5 s for Focus HQ), then
the output fades over :data:`SWITCH_FADE_MS` the way the tier ladder switches tiers
(:func:`hearing.fusion.tiers.fade_kind`): equal delays crossfade, a delay jump up to 500 ms is an
equal-power splice, a larger one dips (old out, then new in) so two different moments never play
on top of each other. A new pipeline holding a tier ladder also waits, up to
:data:`SWITCH_MATCH_S`, until it isolates the target (``enrolled`` / ``focus_hq``) whenever the old
one did, so a preset click never drops to the gate. ``transition`` in telemetry shows it. Leaving
Synthetic/Text and degrading after a failure switch at once, as before.

Enrollment belongs to the session (T-078): one :class:`~hearing.tse.enroll.AutoEnroller` on
``ctx.telemetry.extras["enroller"]``, fed with every hop's room audio whatever mode plays, is the
one every enrolled ``tse`` stage uses, so new ladders (latency presets, the Synthetic/Text focus
feed) and mode round trips keep the target's voice clip.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from hearing import registry
from hearing.audio.agc import Agc, floor_rule_kwargs
from hearing.audio.io import HOP, PIPELINE_SR, float_to_pcm16, pcm16_to_float
from hearing.audio.stream import Chunker
from hearing.context.features import ContextTracker
from hearing.context.prefs import (
    AUTO_MODES,
    Advisor,
    AdvisorConfig,
    PrefLog,
    Recommender,
)
from hearing.fusion.mixer import ALIGNED_MIX_KEY, RESIDUAL_KEY
from hearing.fusion.tiers import fade_gains, fade_kind
from hearing.latency_preset import ISOLATIONS, apply_isolation, apply_preset
from hearing.registry import Pipeline
from hearing.server import stages as _stages  # noqa: F401  (registers "silence")
from hearing.server import team as team_preset
from hearing.server.judge import (
    JUDGE_AGC,
    JUDGE_FIELDS,
    JUDGE_LATENCY,
    JUDGE_NATURAL_SUPPRESSION,
    JUDGE_PIN_HOP_ON_ISOLATED,
    AutoCalibrator,
    judge_spec,
)
from hearing.server.namecall import DEFAULT_MATCH as NAME_DEFAULT_MATCH
from hearing.server.namecall import NAME_CALL_SPEC, NAME_PROMPT_ENV, NameCaller
from hearing.server.namecall import TSE_HEARD_SIM as NAME_TSE_HEARD_SIM
from hearing.server.protocol import (
    KIND_AUDIO_IN,
    KIND_AUDIO_OUT,
    KIND_VIDEO_IN,
    Control,
    Hello,
    ProtocolError,
    error_message,
    pack_frame,
    parse_message,
    unpack_frame,
)
from hearing.server.remote import (
    AUTO_MAX_RTT_MS,
    AUTO_UPGRADE_S,
    ENV_URL,
    HQ_BACKENDS,
    AutoBackend,
    HealthMonitor,
    RemoteStage,
    auto_rule,
)
from hearing.server.resample import StreamResampler
from hearing.server.vision import DETECTORS, VisionEngine
from hearing.synth import voice_match
from hearing.synth.apparent_voice import RollingVoiceEstimator
from hearing.synth.asr import VOICED_RMS
from hearing.synth.pipeline import SynthStage
from hearing.synth.voice_match import (
    SAME_BUCKET_KEY,
    ApparentEstimate,
    SameBucket,
    VoiceChoice,
    VoiceMatcher,
)
from hearing.tse.enroll import (
    ENROLL_S,
    ENROLLER_KEY,
    PROFILE_TARGET_ENV,
    AutoEnroller,
    list_profiles,
    load_profile,
    profile_pairs,
    profiles_dir,
)
from hearing.tse.gpu import TORCH_LOCK
from hearing.tse.lips import FEED_KEY, LipFeed
from hearing.types import AudioChunk, Mode, StageContext, TargetState

log = logging.getLogger(__name__)

# T-133: the mic/output the latest page reported in its hello (``GET /health`` ``client_devices``), so
# scripts/judge_check.py can assert the demo mic is the one in use. Process-wide: the last hello with a label wins.
LAST_CLIENT_DEVICES: dict = {}

PIPELINE_MODES = ("natural", "focus_hq", "synthetic", "text", "quiet", "name_call")  # "auto" picks one
# Natural fast path: DeepFilterNet3 low-latency (13 ms algorithmic vs 33 ms for "dfn", and
# higher SI-SDRi on noise-only clips, T-010/T-050) with the slider. The visual gate chain
# (``natural_gate``) scored worse on demo-mf-easy (s=50: SI-SDRi -3.06 dB, DNSMOS 2.12 vs plain
# dfn_ll at s=80: -0.23 dB, 2.92; Q-240/T-103); the other talker is Focus HQ's job. Missing
# weights degrade to the fallback.
DEFAULT_PIPELINES = {
    "natural": "dfn_ll",
    "focus_hq": "tiers",  # the fallback ladder over config/tiers.yaml (hearing.fusion.tiers)
    "synthetic": "passthrough>synth(speak=true)",  # natural chain > synth, see ServerConfig
    "text": "passthrough>synth(speak=false)",
    "quiet": "silence",
    "name_call": NAME_CALL_SPEC,  # dfn_ll until someone says the wearer's name, then Enrolled (T-120)
}
# Named pipelines a mode may ask for by name (``HEARING_PIPELINES="natural=natural_gate"``):
# kept selectable so the gate chain can be re-evaluated when gate improvements land (T-103).
NAMED_PIPELINES = {"natural_gate": "dfn_ll>gate>mixer"}
SYNTH_MODES = {"synthetic": True, "text": False}  # mode -> synth(speak=...)
# The ``mixture`` feed: the raw room, also when a natural chain is configured (T-074/T-079 bench:
# DFN in front of Whisper gave no WER gain and doubled phrase latency, 3100 vs 1390 ms p50).
SYNTH_BASE = "passthrough"
SYNTH_STAGE = "synth"
LOCK_LOST_DETAIL = "lost the person you locked — look at them again"  # acquisition detail (T-107)
# T-329: the Enrolled tier's ladder reasons that mean "still starting" (no verdict yet, the tse stage's
# model loading or first cold window, or eligible and waiting out upgrade_after_s), not "held out"
PROFILE_LOADING_REASONS = frozenset({"", "loading", "warming up"})
# What feeds Synthetic/Text ASR (``control.synth_feed``): the raw mixture, the natural-path
# enhancer, or the target-isolating tier ladder. ``auto``: mixture, focus with 2+ talkers.
SYNTH_FEEDS = ("auto", "mixture", "natural", "focus")
AUTO_FEED = "mixture"  # what auto plays with one talker (and while focus cannot build)
AUTO_FOCUS_SPEAKERS = 2  # auto: this many simultaneous talkers (context.speakers) switch to focus
AUTO_FOCUS_ATTACK_S = 2  # ... once seen for this many consecutive seconds (a 1 s blip does not flip)
AUTO_FOCUS_RELEASE_S = 5  # ... and it returns to mixture after this many seconds below that
VOICE_EVERY_S = 2.0  # at most one voice estimate per 2 s of target audio
DEFAULT_TIER = "audio-only"
QUIET_FADE_MS = 20.0  # T-149: Quiet ramps the output to digital silence (and back in Focus) over this
SWITCH_FADE_MS = 100.0  # crossfade from the old pipeline once the new one is primed
SWITCH_MATCH_S = 6.0  # a new ladder may take this long to reach the old one's isolating tier
ISOLATING_TIERS = frozenset({"enrolled", "focus_hq"})  # tiers that play the target, not the room
SPEAKING_P = 0.5  # target p_speaking that lets its audio into the voice estimate
HQ_STALL_MS = 500.0  # hq_backend=remote: a silent remote hands over to local within this
HQ_DELAY_MS = 200.0  # remote playout buffer: >= p95 RTT (auto keeps RTT < 150 ms)
_RAW = ""  # identity: what a failing fallback degrades to
# Packages whose stages the server can use; imported best-effort so a missing package
# just means its stages show as "unavailable" in /health.
STAGE_MODULES = (
    "hearing.audio.basic",
    "hearing.audio.agc",
    "hearing.enhance.dfn",
    "hearing.enhance.gtcrn",
    "hearing.fusion",
    "hearing.fusion.tiers",
    "hearing.vision",
    "hearing.tse",
    "hearing.synth",
    "hearing.server.remote",
)


def _clear_side_channels(ctx: StageContext) -> None:
    """A residual left by one pipeline must never reach another pipeline's mixer."""
    for key in (RESIDUAL_KEY, ALIGNED_MIX_KEY):
        ctx.telemetry.extras.pop(key, None)


def load_stage_modules() -> None:
    for module in STAGE_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as e:
            if e.name != module:  # the package exists but one of its imports is missing
                raise


def parse_pipelines(text: str) -> dict[str, str]:
    """``"natural=dfn>gate>mixer; quiet=silence"`` -> ``{mode: pipeline spec}``.

    Separators inside quoted stage arguments are preserved.
    """
    result = {}
    for part in registry._split(text, ";"):
        if not part:
            continue
        mode, equals, spec = part.partition("=")
        mode, spec = mode.strip(), spec.strip()
        if not equals or not spec:
            raise ValueError(f"expected mode=pipeline, got {part!r}")
        if mode not in PIPELINE_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {PIPELINE_MODES}")
        result[mode] = spec
    return result


def synth_spec(speak: bool, asr: str | None = None, tts: str | None = None) -> str:
    """``synth(speak=...)``; engines left out are picked by ``get_asr()``/``get_tts()``
    (``HEARING_ASR``/``HEARING_TTS``, else ElevenLabs with a key, else local, else mock)."""
    args = [f"speak={'true' if speak else 'false'}"]
    args += [f"{k}={json.dumps(v)}" for k, v in (("asr", asr), ("tts", tts)) if v]
    return f"{SYNTH_STAGE}({','.join(args)})"


def default_prefs_log() -> Path | None:
    """``HEARING_PREFS_LOG`` if set; otherwise choices are learned in memory only."""
    path = os.environ.get("HEARING_PREFS_LOG")
    return Path(path).expanduser() if path else None


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def default_console_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "web" / "console"


@dataclass
class ServerConfig:
    """``pipelines`` overrides :data:`DEFAULT_PIPELINES` per mode; ``prefs_log`` is the
    JSONL file of the user's mode choices (``None``: learn in memory for the session).

    Unless given explicitly, ``synthetic``/``text`` are the configured natural pipeline
    (``passthrough`` when natural is left at its default) followed by ``synth``; ``asr``/``tts`` name its engines (``None``: picked from the environment).

    ``faces``/``asd`` pick the face detector and active-speaker backend
    (``HEARING_FACES``/``HEARING_ASD``, default ``auto``; see :mod:`hearing.server.vision`).
    ``vision_threaded=False`` runs detection inline (deterministic tests).

    ``hq_backend`` is where Focus HQ starts (``HEARING_HQ_BACKEND``, default ``auto``);
    ``remote_url`` the GPU session server (``HEARING_REMOTE_URL``). ``hq_remote`` is the
    remote Focus HQ pipeline, by default ``remote(...)`` with the local ``focus_hq`` pipeline
    as its fallback; ``None`` when there is no url or ``focus_hq`` is itself remote
    (``hq_unavailable`` says why). ``hq_auto_upgrade_s``: how long ``auto`` waits for a good
    remote before moving Focus HQ to it.
    """

    pipelines: dict[str, str] = field(default_factory=dict)
    fallback: str = "passthrough"
    telemetry_hz: float = 10.0
    console_dir: Path | None = None
    replay_dir: Path = field(default_factory=lambda: Path(os.environ.get("HEARING_REPLAY_DIR")
                                                          or Path(__file__).resolve().parents[2] / "eval/fixtures/demo"))
    prefs_log: Path | None = field(default_factory=default_prefs_log)
    advisor: AdvisorConfig = field(default_factory=AdvisorConfig)
    asr: str | None = None
    tts: str | None = None
    faces: str = field(default_factory=lambda: _env("HEARING_FACES", "auto"))
    asd: str = field(default_factory=lambda: _env("HEARING_ASD", "auto"))
    vision_threaded: bool = True
    hq_backend: str = field(default_factory=lambda: _env("HEARING_HQ_BACKEND", "auto"))
    remote_url: str | None = field(default_factory=lambda: os.environ.get(ENV_URL) or None)
    hq_remote: str | None = None
    hq_auto_upgrade_s: float = AUTO_UPGRADE_S
    hq_unavailable: str = field(default="", init=False)
    synth_feed: str = field(default_factory=lambda: _env("HEARING_SYNTH_FEED", "auto"))  # initial feed
    synth_explicit: set[str] = field(default_factory=set, init=False)
    synth_base: str = field(default=SYNTH_BASE, init=False)
    quiet_explicit: bool = field(default=False, init=False)  # quiet= given: that spec plays, no captions

    def __post_init__(self):
        unknown = set(self.pipelines) - set(PIPELINE_MODES)
        if unknown:
            raise ValueError(f"unknown modes {sorted(unknown)}; expected {PIPELINE_MODES}")
        given = {mode: NAMED_PIPELINES.get(spec.strip(), spec) for mode, spec in self.pipelines.items()}
        self.pipelines = {**DEFAULT_PIPELINES, **given}
        base = self.synth_base  # the "mixture" feed: always the raw room
        self.synth_explicit = {m for m in SYNTH_MODES if m in given}  # their feed is not selectable
        self.quiet_explicit = "quiet" in given
        for mode in SYNTH_MODES:
            if mode not in given:
                self.pipelines[mode] = f"{base}>{self.synth_only(mode)}"
        if self.telemetry_hz <= 0:
            raise ValueError("telemetry_hz must be positive")
        if self.faces not in DETECTORS:
            raise ValueError(f"unknown face detector {self.faces!r}; expected one of {DETECTORS}")
        self.console_dir = Path(self.console_dir) if self.console_dir else default_console_dir()
        if self.synth_feed not in SYNTH_FEEDS:
            raise ValueError(f"unknown synth_feed {self.synth_feed!r}; expected one of {SYNTH_FEEDS}")
        if self.hq_backend not in HQ_BACKENDS:
            raise ValueError(f"unknown hq_backend {self.hq_backend!r}; expected one of {HQ_BACKENDS}")
        local = self.pipelines["focus_hq"]
        if self.hq_remote is None and self.remote_url:
            if "remote" in _top_names(local):
                self.hq_unavailable = "the focus_hq pipeline is itself remote"
            else:
                self.hq_remote = (f"remote(url={json.dumps(self.remote_url)}, "
                                  f"fallback={json.dumps(local)}, delay_ms={HQ_DELAY_MS:g}, "
                                  f'stall_ms={HQ_STALL_MS:g}, fallback_tier="")')
        if self.hq_remote is None and not self.hq_unavailable:
            self.hq_unavailable = f"no remote GPU url (set {ENV_URL})"

    def synth_only(self, mode: str) -> str:
        """The bare synth spec of a Synthetic/Text mode: its fallback when the chain fails."""
        return synth_spec(SYNTH_MODES[mode], self.asr, self.tts)

    def stage_names(self) -> set[str]:
        """Stage names referenced by the configured pipelines (ignores unparsable specs)."""
        names = set()
        for spec in [*self.pipelines.values(), self.fallback, self.hq_remote or ""]:
            names |= _top_names(spec)
        return names


def _top_names(spec: str) -> set[str]:
    """The top-level stage names of a pipeline spec (unparsable parts ignored)."""
    names = set()
    for part in registry._split(spec, ">"):
        try:
            names.add(registry._parse_stage(part)[0])
        except ValueError:
            pass
    return names


def _stage_keys(pipeline: Pipeline) -> list[str]:
    """Telemetry keys for each stage; repeated names get ``#2``, ``#3``…"""
    keys, seen = [], {}
    for stage in pipeline.stages:
        seen[stage.name] = seen.get(stage.name, 0) + 1
        keys.append(stage.name if seen[stage.name] == 1 else f"{stage.name}#{seen[stage.name]}")
    return keys


def _synth_stages(pipeline: Pipeline | None) -> list[SynthStage]:
    return [s for s in pipeline.stages if isinstance(s, SynthStage)] if pipeline else []


def _nested(pipeline: Pipeline | None) -> list:
    """Every stage of ``pipeline``, including those inside a tier ladder or a remote fallback."""
    from hearing.fusion.tiers import iter_stages

    return list(iter_stages(pipeline))[1:] if pipeline is not None else []


def _ladder(pipeline: Pipeline | None):
    from hearing.fusion.tiers import TierLadder

    return next((s for s in _nested(pipeline) if isinstance(s, TierLadder)), None)


class _VoiceWorker:
    """Voice estimates off the audio path: the ``VoiceEstimator`` a
    :class:`RollingVoiceEstimator` calls, running one estimate at a time on its own thread
    (the real model loads for seconds, then takes tens of ms). A window that is ready
    while the previous estimate still runs is skipped. Results go back via :meth:`poll`,
    tagged with the track that was the target when its audio was captured.
    """

    def __init__(self, estimator: voice_match.VoiceEstimator):
        self.estimator = estimator
        self.track: int | None = None  # the session sets this before each feed
        self._pool: ThreadPoolExecutor | None = None
        self._job: tuple[int, Future] | None = None

    def estimate(self, samples: np.ndarray) -> None:
        if self._job is None and self.track is not None:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(1, thread_name_prefix="voice-estimate")
            self._job = (self.track, self._pool.submit(self._estimate, samples))

    def _estimate(self, samples: np.ndarray) -> ApparentEstimate | None:
        with TORCH_LOCK:  # the wav2vec2 model shares the GPU with Focus HQ and the ASD
            return self.estimator.estimate(samples)

    def poll(self, timeout: float = 0.0) -> tuple[int, ApparentEstimate | None] | None:
        """The finished estimate as ``(track, estimate)``; re-raises the estimator's error."""
        if self._job is None:
            return None
        track, job = self._job
        try:
            estimate = job.result(timeout)
        except FutureTimeout:
            return None
        finally:
            if job.done():
                self._job = None
        return track, estimate

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        self._job = None


class Session:
    def __init__(self, config: ServerConfig, device: str):
        self.config = config
        self.device = device
        self.ctx = StageContext.default()
        self.ctx.telemetry.device = device
        self.ctx.telemetry.tier = DEFAULT_TIER
        self.mode: Mode = "natural"
        self.voice_override: str | None = None
        self.ab: str = "processed"  # client-side A/B toggle, echoed into telemetry only
        self.hello = Hello(type="hello")
        self.video_frames = 0
        self.video_dropped = 0  # T-168: camera frames ignored because the team preset is audio only
        self._substitute: dict[str, str] = {}  # failed spec -> spec used instead
        self._built: dict[str, Pipeline] = {}
        self._keys: dict[str, list[str]] = {}
        self._active_spec: str | None = None
        self._chunker = Chunker(HOP)
        self._to_16k = self._from_16k = None
        self._hop_seq = 0
        self._last_seq: int | None = None
        self._acquisition_key: tuple | None = None
        self._acquisition_since = time.monotonic()
        self._enrollment_previous: tuple | None = None
        self._enrollment_restarted = False
        self._last_ts = 0.0
        self._stage_sum: dict[str, float] = {}
        self._stage_n: dict[str, int] = {}
        self._server_sum = 0.0
        self._server_n = 0
        self._latency: dict[str, float] = {}
        self._server_ms = 0.0
        self._frame_ms = self.hello.frame_ms
        self._resample_ms = 0.0
        self.context = ContextTracker()
        self.advisor = Advisor(Recommender(PrefLog(config.prefs_log)), config.advisor)
        self._prefs_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefs")
        self._prefs_jobs: list[Future] = []
        self._context_s = 0  # audio clock: one context vector per second
        # Voice match: estimates live only in the matcher (memory, per track), never on the wire.
        self.voice_matcher = VoiceMatcher()
        self.same_bucket = SameBucket()  # tier ladder guard: target + another face, same voice group
        self.face_estimator = voice_match.get_face_estimator()
        estimator = voice_match.get_voice_estimator()
        self._voice_worker = _VoiceWorker(estimator) if estimator is not None else None
        self._voice_feed = (RollingVoiceEstimator(self._voice_worker, every_s=VOICE_EVERY_S)
                            if self._voice_worker is not None else None)
        self._voice_track: int | None = None
        self._voice_choice = VoiceChoice("neutral", "default", 0.0)
        self.vision = VisionEngine(config.faces, config.asd, threaded=config.vision_threaded,
                                   on_face=self.on_face, warmup=self._warm_face_estimator)
        self._vision_pending: list[dict] = []
        self.lip_feed = LipFeed()  # mouth crops of every visible face, read by Focus HQ
        self.ctx.telemetry.extras[FEED_KEY] = self.lip_feed
        self.enroller = AutoEnroller(ENROLL_S, PIPELINE_SR)  # shared by every enrolled tse (T-078)
        self.ctx.telemetry.extras[ENROLLER_KEY] = self.enroller
        self._pipeline_tier: str | None = None  # tier the active pipeline reported
        self._outgoing: tuple[str, Pipeline] | None = None  # still playing while the new one primes
        self._outgoing_tier = ""  # the tier the outgoing pipeline reported on its last hop
        self._incoming_n = 0  # samples the new pipeline has processed since the switch
        self._fade_pos = 0
        self._switch_fade: str | None = None  # fade_kind once the new pipeline is primed
        self._fade_n = round(SWITCH_FADE_MS * PIPELINE_SR / 1000.0)
        self.hq_backend: str = config.hq_backend
        self.synth_feed_request: str = config.synth_feed  # control.synth_feed: auto|mixture|natural|focus
        self._auto_feed = AUTO_FEED  # what auto resolved to (updated at 1 Hz from context.speakers)
        self._auto_high_s = self._auto_low_s = 0
        self.latency_preset: str = "balanced"  # Focus HQ latency/quality preset: speed|balanced|accuracy
        self.ctx.telemetry.extras["latency_preset"] = self.latency_preset
        self.isolation: str = "standard"  # control.isolation: enrolled-tier strictness (T-085)
        self.profile: str | None = None  # control.profile: "judge" locks the demo defaults (T-088)
        self.auto_calibrator = AutoCalibrator()  # on with the judge profile
        self.out_agc = Agc(PIPELINE_SR, **floor_rule_kwargs())  # T-135 output AGC + limiter (T-179 floor rule opt-in)
        self.agc_enabled = True  # control.agc (T-169): false on the remote leg, the laptop levels once
        # T-135: the tap's out_pre_agc.wav (16 kHz sessions): separation is scored before output leveling
        self.pre_agc_sink: Callable[[bytes], None] | None = None
        self._isolated_once = False  # judge profile: tse hops pinned from the first isolated tick on
        self.preset: dict | None = None  # the team demo preset on top of the judge profile (T-130)
        self._preset_pins: dict | None = None
        self._hq_reprobe: team_preset.Reprobe | None = None  # T-154: a GPU-caused local pin, re-probed
        self._quiet_s = 0.0  # T-154: input below VOICED_RMS this long (a pause for the backend switch)
        self._isolated = False  # T-154: the last acquisition state was locked (the pill's Isolated)
        self._same_bucket_guard = True  # off only under the team preset with a verified M/F pair
        self._lock_on_start = False  # team preset: lock the first face attention picks
        self._namecall_timers: dict = {}  # team preset: quiet_s / timeout_s for the NameCaller
        self._quiet_synth: SynthStage | None = None  # T-149: Quiet's captions-only synth, fed by the Focus chain
        self._quiet_synth_failed = False  # no ASR engine could be built: Quiet stays captionless
        self._quiet_on = False  # the last hop ran in Quiet
        self._quiet_gain = 1.0  # output gain: ramps to 0 in Quiet, back to 1 outside
        self._quiet_feed: dict | None = None  # telemetry ``quiet``: feed focus|paused and why
        self._caption_tap: np.ndarray | None = None  # the active pipeline's output this hop (Quiet's ASR input)
        self.namecall: NameCaller | None = None  # Name Call mode (T-120), built on first use
        self._name: tuple[str, list[str]] = ("", [])
        self._name_match, self._name_exclude = NAME_DEFAULT_MATCH, []  # T-123 false-trigger guards
        self._namecall_events: list[dict] = []
        self._namecall_sim: float | None = None
        self._hq_auto = AutoBackend(self.config.hq_auto_upgrade_s)
        self._hq_monitor: HealthMonitor | None = None
        self._hq_rule = (False, "")  # last auto_rule verdict
        self._ensure_monitor()

    # ------------------------------------------------------------ HQ backend

    def _ensure_monitor(self) -> None:
        """Watch the remote's /health while the user might want it (remote/auto)."""
        wanted = self.hq_backend != "local" and self.config.hq_remote is not None
        if wanted and self._hq_monitor is None:
            self._hq_monitor = HealthMonitor(self.config.remote_url).start()
        elif not wanted and self._hq_monitor is not None:
            self._hq_monitor.close()
            self._hq_monitor = None

    def _hq_remote_selected(self) -> bool:
        if self.config.hq_remote is None:
            return False
        return self.hq_backend == "remote" or (self.hq_backend == "auto" and self._hq_auto.remote)

    def _remote_stage(self, playing: bool = True) -> RemoteStage | None:
        """The remote HQ pipeline's ``remote`` stage (only while it plays, by default)."""
        spec = self.config.hq_remote
        if spec is None or (playing and self._active_spec != spec):
            return None
        return next((s for s in _nested(self._built.get(spec)) if isinstance(s, RemoteStage)), None)

    def remote_rtt_ms(self) -> float | None:
        """Stream RTT p50 while the remote HQ pipeline plays and measures, else /health RTT."""
        stage = self._remote_stage()
        rtt = stage.rtt_ms() if stage is not None else None
        if rtt is None and self._hq_monitor is not None:
            rtt = self._hq_monitor.snapshot()["rtt_ms"]
        return rtt

    def _update_hq(self) -> None:
        """``auto``: re-apply the rule (with hysteresis); the next hop switches pipelines."""
        if self.hq_backend != "auto" or self._hq_monitor is None:
            return
        self._hq_rule = auto_rule(self._hq_monitor.snapshot()["ok"], self.remote_rtt_ms(),
                                  AUTO_MAX_RTT_MS)
        self._hq_auto.update(self._hq_rule[0], time.monotonic())

    def set_hq_backend(self, backend: str) -> list[dict]:
        """Pick where Focus HQ runs; returns an ``error`` event when remote is impossible."""
        if backend not in HQ_BACKENDS:
            return [error_message("hq_backend", f"unknown hq_backend {backend!r}")]
        if backend == self.hq_backend:
            return []
        was_remote = self._hq_remote_selected()
        self.hq_backend = backend
        self._ensure_monitor()
        # Entering auto from a healthy remote keeps it (no remote -> local -> remote pumping).
        self._hq_auto = AutoBackend(self.config.hq_auto_upgrade_s)
        if backend == "auto" and was_remote and self._hq_monitor is not None:
            self._update_hq()
            self._hq_auto.remote = self._hq_rule[0]
        if backend != "local" and self.config.hq_remote is None:
            return [error_message("hq_backend", f"Focus HQ cannot run remotely: "
                                  f"{self.config.hq_unavailable}", degraded_to="local")]
        return []

    def set_latency_preset(self, preset: str) -> list[dict]:
        """Set Focus HQ latency/quality preset; returns an ``error`` event if invalid."""
        if preset not in ("speed", "balanced", "accuracy"):
            return [error_message("latency_preset", f"unknown preset {preset!r}; expected speed, balanced, or accuracy")]
        if preset == self.latency_preset:
            return []
        self.latency_preset = preset  # active_spec now names tiers(preset=...): a make-before-break switch
        self.ctx.telemetry.extras["latency_preset"] = preset
        return []

    def set_isolation(self, isolation: str) -> list[dict]:
        """``standard`` | ``strict`` (T-085): a make-before-break switch to the ladder with or without
        the enrolled tier's pause-duck rule; the shared enroller keeps its clips."""
        if isolation not in ISOLATIONS:
            return [error_message("isolation", f"unknown isolation {isolation!r}; expected standard or strict")]
        self.isolation = isolation
        self.ctx.telemetry.extras["isolation"] = isolation  # the remote stage forwards it (T-110)
        return []

    def set_enroll(self, action: str, seconds: float, name: str | None = None,
                   other: str | None = None) -> list[dict]:
        """Manual voice calibration (``control.enroll``): ``start`` / ``clear`` on the shared enroller;
        ``profile`` loads a voice profile (T-125)."""
        if action == "start":
            self.enroller.calibrate(seconds)
        elif action == "clear":
            self.enroller.clear()
            self.ctx.telemetry.extras.pop("voice_profile", None)  # the remote stage forwards the clear
            self.auto_calibrator.enabled = False  # T-177: Release; no re-enrolling whoever talks next
        elif action == "profile":
            errors = self.load_voice_profile(name, other)
            if not errors and self.profile == "judge":
                self.auto_calibrator.enabled = True  # T-177: a chosen person ends the Release
            return errors
        else:
            return [error_message("enroll", f"unknown enroll action {action!r}; expected start, clear or profile")]
        return []

    def load_voice_profile(self, name: str | None, other: str | None = None) -> list[dict]:
        """Enroll voice profile ``name`` from ``$HEARING_PROFILES_DIR`` (T-125) as the calibrated target,
        ``other`` (default: every other profile there) as the known negative from t=0."""
        root = profiles_dir()
        if root is None:
            return [error_message("enroll", "voice profiles are off (HEARING_PROFILES_DIR is not set)")]
        names = list_profiles(root)
        if not name or name not in names:
            return [error_message("enroll", f"no voice profile {name!r}; have {names}")]
        if other is not None and other not in names:
            return [error_message("enroll", f"no voice profile {other!r}; have {names}")]
        try:
            pieces = load_profile(name, root)
            others = [p for o in ([other] if other else [n for n in names if n != name]) for p in load_profile(o, root)]
        except (OSError, ValueError) as e:
            return [error_message("enroll", f"voice profile {name!r}: {e}")]
        self._preload_profiles()
        self.enroller.enroll_profile(name, pieces, others or None)
        self.ctx.telemetry.extras["voice_profile"] = {"name": name, **({"other": other} if other else {})}
        return []

    def set_profile(self, profile: str) -> list[dict]:
        """``control.profile`` (T-088): ``judge`` applies :data:`~hearing.server.judge.JUDGE_FIELDS`, the
        backend's latency preset, and automatic calibration, as if the client had sent them."""
        preset_name = "team" if profile == "team" else team_preset.active_preset()
        profile = "judge"  # the team preset runs on top of the judge profile (T-130)
        fields = dict(JUDGE_FIELDS)
        if self.config.hq_remote is None:
            fields.pop("hq_backend")  # no GPU configured: auto would only answer an error
        self.profile = profile
        self.ctx.telemetry.extras["profile"] = profile  # the remote stage forwards it (T-110)
        self.auto_calibrator.enabled = True
        errors = self._apply_control(Control(type="control", **fields))
        if preset_name is not None:
            errors.extend(self._apply_preset(preset_name))
        names = list_profiles()
        if names and self.enroller.source != "profile":  # T-125: a known voice from the first window
            wanted = os.environ.get(PROFILE_TARGET_ENV, "").strip()
            pair = (self.preset or {}).get("pair") or {}
            mine, theirs = ((pair.get(k) or {}).get("profile") for k in ("target", "other"))
            if not wanted and mine in names:
                wanted = mine
            other = theirs if wanted == mine and theirs in names and theirs != mine else None
            errors.extend(self.load_voice_profile(wanted if wanted else names[0], other))
        return errors + self.set_latency_preset(JUDGE_LATENCY.get(self.hq_status()["active"], "balanced"))

    def _pin_hops(self, pipeline: Pipeline | None, acquisition: dict) -> None:
        """Judge profile (T-135, Q-341): from the first isolated tick on (acquisition ``locked``, "target
        isolation active", the pill's Isolated), every tse stage of the playing pipeline keeps its current
        hop (adaptive until then, so a loaded machine still locks)."""
        if self.profile != "judge" or not JUDGE_PIN_HOP_ON_ISOLATED:
            return
        self._isolated_once = self._isolated_once or acquisition.get("state") == "locked"
        if self._isolated_once:
            for stage in _nested(pipeline):
                if not getattr(stage, "hop_pinned", True):
                    stage.pin_hop()

    @property
    def _agc_on(self) -> bool:
        # T-129: the GPU session behind a laptop's remote stage gets profile=judge too, but the laptop's
        # own session levels the output (T-169 also sends control.agc false from the remote stage).
        return (self.profile == "judge" and JUDGE_AGC and self.agc_enabled
                and self.hello.client != "remote-stage")

    def _apply_preset(self, name: str) -> list[dict]:
        """The team demo preset (T-130, ``config/demo-team.yaml``): every pin as a control field or a
        session setting; the Focus HQ backend is pinned to the Mac (T-181), or under ``HEARING_HQ_BACKEND=remote``
        from a ``/health`` probe (no auto); a local pin caused by the GPU is re-probed and moves to remote at a
        safe point (T-154, :meth:`_update_hq_pin`)."""
        try:
            preset = team_preset.load_preset(name)
        except (OSError, ValueError, yaml.YAMLError) as e:
            return [error_message("profile", f"demo preset {name!r}: {e}")]
        self.preset = preset
        url = self.config.remote_url if self.config.hq_remote is not None else None
        backend = team_preset.pinned_backend(url, preset)
        self._same_bucket_guard = team_preset.same_bucket_guard(preset)
        self._lock_on_start = bool(preset.get("lock_on_start"))
        self._namecall_timers = {k: float(preset[f"namecall_{k}"]) for k in ("quiet_s", "timeout_s")
                                 if preset.get(f"namecall_{k}") is not None}
        if self.namecall is not None:
            self.namecall.quiet_s = self._namecall_timers.get("quiet_s", self.namecall.quiet_s)
            self.namecall.timeout_s = self._namecall_timers.get("timeout_s", self.namecall.timeout_s)
        fields = team_preset.team_fields(preset, backend[0] if url else None, self.vision.selector.attention)
        rev = team_preset.remote_rev(url)
        stale = team_preset.rev_warning(team_preset.local_rev(), rev) if url else None
        if stale:
            log.warning("team preset: %s", stale)
        self._preset_pins = team_preset.pins(preset, backend, self._same_bucket_guard, rev, stale)
        if self._hq_reprobe is not None:
            self._hq_reprobe.close()
            self._hq_reprobe = None
        if url and backend[0] == "local" and team_preset.wants_gpu(preset):  # T-154: the GPU blipped or was slow
            self._hq_reprobe = self._make_reprobe(url, preset)
            self._preset_pins["hq_reason"] += f"; re-probing every {self._hq_reprobe.every_s:g} s"
        return self._apply_control(Control(type="control", **fields))

    def _make_reprobe(self, url: str, preset: dict) -> team_preset.Reprobe:
        return team_preset.Reprobe(url, preset).start()

    def _update_hq_pin(self, samples: np.ndarray) -> None:
        """T-154: the team preset pinned local because of the GPU and a re-probe now picks remote:
        switch at the next safe point (before Isolated, or at a pause), log it, tell the drawer."""
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0
        self._quiet_s = 0.0 if rms > VOICED_RMS else self._quiet_s + len(samples) / PIPELINE_SR
        probe = self._hq_reprobe
        if probe is None or probe.ready is None or self.preset is None:
            return
        if not team_preset.safe_to_switch(self._isolated, self._quiet_s):
            return
        self._hq_reprobe = None
        probe.close()
        if self.hq_backend != "local":  # the user picked a backend since: theirs stands
            return
        reason = f"{probe.ready[1]}; switched from local at a {'pause' if self._isolated else 'safe point'}"
        log.info("team preset: Focus HQ local -> remote (%s)", reason)
        self.set_hq_backend("remote")
        if self._preset_pins is not None:
            self._preset_pins.update(hq_backend="remote", hq_reason=reason)

    def hq_status(self) -> dict:
        """``hq_active`` (``local``/``remote``: what Focus HQ plays now, or would play) and why."""
        if self.config.hq_remote is None:
            return {"active": "local", "reason": self.config.hq_unavailable}
        if not self._hq_remote_selected():
            if self.hq_backend == "local":
                return {"active": "local", "reason": "local selected"}
            return {"active": "local", "reason": self._hq_rule[1] or "checking the remote"}
        stage = self._remote_stage()
        monitor = self._hq_monitor.snapshot() if self._hq_monitor is not None else {}
        if stage is not None:
            if stage.source == "delayed":  # connected, but the reply missed the playout buffer (T-075)
                return {"active": "remote", "reason": f"remote late (RTT p50 {stage.rtt_ms() or 0:.0f} ms > "
                                                      f"{stage.remote_latency_ms:.0f} ms buffer), local fills"}
            if stage.source != "local":
                return {"active": "remote", "reason": "remote GPU playing"}
            error = stage.client.last_error if stage.client is not None else None
            return {"active": "local", "reason": f"remote down, local fallback ({error or 'no audio back'})"}
        if monitor.get("ok") and self._last_seq is None:  # T-156: nothing to send the GPU yet
            return {"active": "remote", "reason": "remote selected; no microphone audio from the client yet"}
        if monitor.get("ok"):
            return {"active": "remote", "reason": self._hq_rule[1] or "remote selected"}
        return {"active": "local", "reason": f"remote down ({monitor.get('error') or 'not checked yet'})"}

    # ------------------------------------------------------------ state

    @property
    def effective_mode(self) -> str:
        return self.advisor.auto_mode if self.mode == "auto" else self.mode

    @property
    def quiet(self) -> bool:
        """T-149: Quiet as the product mode: the Focus chain plays silently and feeds the captions."""
        return self.effective_mode == "quiet" and not self.config.quiet_explicit

    @property
    def active_spec(self) -> str:
        mode = "focus_hq" if self.quiet else self.effective_mode  # T-149: Quiet keeps Focus's pipeline
        if mode == "focus_hq" and self._hq_remote_selected():
            spec = self.config.hq_remote
        elif mode in SYNTH_MODES and mode not in self.config.synth_explicit:
            spec = self._synth_spec(mode, self.synth_feed)
        else:
            spec = self.config.pipelines[mode]
            if mode in ("focus_hq", "name_call"):
                spec = apply_isolation(apply_preset(spec, self.latency_preset), self.isolation)
                if self.profile == "judge":
                    spec = judge_spec(spec)
                if self.preset is not None:
                    spec = team_preset.team_spec(spec, self.preset)
        while spec in self._substitute:
            spec = self._substitute[spec]
        return spec

    def _synth_spec(self, mode: str, feed: str) -> str:
        """The Synthetic/Text chain whose ASR is fed by ``feed`` (see :data:`SYNTH_FEEDS`)."""
        cfg = self.config
        if feed == "focus":  # the same ladder Focus HQ plays, preset included
            base = apply_isolation(apply_preset(cfg.pipelines["focus_hq"], self.latency_preset), self.isolation)
        elif feed == "natural":
            base = cfg.pipelines["natural"]
        else:
            base = cfg.synth_base
        return f"{base}>{cfg.synth_only(mode)}"

    @property
    def synth_feed(self) -> str:
        """The feed in effect: the requested one, ``auto`` resolved by the speaker count and
        by whether the ladder built (a ladder that failed to build plays the bare synth: the mixture)."""
        if self.quiet:  # T-149: forced; whether it is fed right now is telemetry ``quiet.feed``
            return "focus"
        feed = self._auto_feed if self.synth_feed_request == "auto" else self.synth_feed_request
        if self.synth_feed_request == "auto" and self.profile == "judge":
            # T-136: pinned for the demo: the ladder feeds the captions on a CUDA server (the A100), the
            # mixture on the Mac (the local ladder there would starve the ASR and TTS)
            feed = "focus" if self.device == "cuda" else "mixture"
        mode = self.effective_mode
        if feed == "focus" and mode in SYNTH_MODES and self._synth_spec(mode, "focus") in self._substitute:
            return AUTO_FEED
        return feed

    def set_synth_feed(self, feed: str) -> list[dict]:
        if feed not in SYNTH_FEEDS:
            return [error_message("synth_feed", f"unknown synth_feed {feed!r}; expected one of {SYNTH_FEEDS}")]
        self.synth_feed_request = feed  # active_spec changes: the synth stage moves to the new chain
        return []

    def _update_auto_feed(self, speakers: int) -> None:
        """Hysteresis for ``auto`` (called at 1 Hz): focus after 2 s with 2+ talkers, back to
        the mixture after 5 s below that."""
        if speakers >= AUTO_FOCUS_SPEAKERS:
            self._auto_high_s, self._auto_low_s = self._auto_high_s + 1, 0
            if self._auto_high_s >= AUTO_FOCUS_ATTACK_S:
                self._auto_feed = "focus"
        else:
            self._auto_high_s, self._auto_low_s = 0, self._auto_low_s + 1
            if self._auto_feed == "focus" and self._auto_low_s >= AUTO_FOCUS_RELEASE_S:
                self._auto_feed, self._auto_low_s = AUTO_FEED, 0

    def _pipeline(self, errors: list[dict]) -> Pipeline | None:
        """The pipeline for the current mode, building (and degrading) on first use."""
        while True:
            spec = self.active_spec
            if spec == _RAW:
                return None
            pipeline = self._built.get(spec)
            if pipeline is None:
                try:
                    pipeline = self._built[spec] = registry.build_pipeline(spec)
                    self._keys[spec] = _stage_keys(pipeline)
                except Exception as e:  # noqa: BLE001  (unknown stage, bad args, no weights)
                    errors.append(self._degrade(spec, "pipeline", f"cannot build {spec!r}: {e}"))
                    continue
            if spec != self._active_spec:
                self._switch_to(spec, pipeline)
            return pipeline

    def _switch_to(self, spec: str, pipeline: Pipeline) -> None:
        """Make ``spec`` active. Leaving Synthetic/Text drops queued phrases and captions,
        except on a feed change of the same mode, where the synth stage (its ASR, queue and
        captions) moves to the new chain and only the upstream restarts; entering a mode starts
        from fresh streaming state, and the old pipeline keeps playing (make-before-break)
        unless it was synth or has failed."""
        old_spec = self._active_spec
        old = self._built.get(old_spec) if old_spec is not None else None
        handover = self._synth_handover(old_spec, old, spec, pipeline)
        synth_old = _synth_stages(old)  # after a handover: the fresh one the old chain got
        for stage in synth_old:
            stage.reset()
        if handover:
            for stage in pipeline.stages[:-1]:
                stage.reset()
            self._outgoing = None
        elif self._outgoing is not None and self._outgoing[0] == spec:
            # Switching back while the new pipeline primed: the old one never stopped.
            self._outgoing = None
        else:
            pipeline.reset()
            keep = old is not None and not synth_old and old_spec not in self._substitute
            self._outgoing = (old_spec, old) if keep and self._fade_n else None
        self._incoming_n = self._fade_pos = 0
        self._switch_fade = None
        self._outgoing_tier = self._pipeline_tier or ""
        self._pipeline_tier = None
        self._active_spec = spec
        self._sync_voice()

    def _synth_handover(self, old_spec: str | None, old: Pipeline | None, spec: str,
                        pipeline: Pipeline) -> bool:
        """A feed change of one Synthetic/Text mode: swap the two chains' synth stages so the
        playing one (queue, captions, ASR state) continues behind the new upstream; the old
        chain keeps the fresh one. Not after a failure (the old synth may be the culprit)."""
        if old is None or old_spec in self._substitute:
            return False
        tails = [self.config.synth_only(m) for m in SYNTH_MODES]
        tail = next((t for t in tails if spec == t or spec.endswith(f">{t}")), None)
        if tail is None or not (old_spec == tail or old_spec.endswith(f">{tail}")):
            return False
        if not (isinstance(old.stages[-1], SynthStage) and isinstance(pipeline.stages[-1], SynthStage)):
            return False
        playing, fresh = old.stages[-1], pipeline.stages[-1]
        old.stages = (*old.stages[:-1], fresh)
        pipeline.stages = (*pipeline.stages[:-1], playing)
        return True

    def _synth(self) -> SynthStage | None:
        """The ``synth`` stage of the active pipeline, if it has one."""
        stages = _synth_stages(self._built.get(self._active_spec)) if self._active_spec else []
        return stages[-1] if stages else None

    def _degrade(self, spec: str, stage: str, message: str) -> dict:
        mode = self.effective_mode
        # Synthetic/Text never fall back to room audio: bare synth, then silence.
        chain = ([self.config.synth_only(mode), "silence", _RAW] if mode in SYNTH_MODES
                 else [self.config.fallback, _RAW])
        target = chain[chain.index(spec) + 1] if spec in chain else chain[0]
        self._substitute[spec] = target
        if self._outgoing is not None and self._outgoing[0] == spec:
            self._outgoing = None
        return error_message(stage, message, degraded_to=target or "raw")

    # ------------------------------------------------------------ voice

    def _refresh_voice(self) -> None:
        """Re-pick the voice for the current target and override; push it to synth."""
        track = self.ctx.target.track_id
        if track != self._voice_track:
            self._voice_track = track
            if self._voice_feed is not None:
                self._voice_feed.reset()  # never mix two people's speech in one estimate
        self._voice_choice = self.voice_matcher.choice(track, self.voice_override)
        self._sync_voice()

    def _sync_voice(self) -> None:
        synth = self._synth()
        bucket = self._voice_choice.bucket
        if synth is not None and synth.current_voice != bucket:
            synth.set_voice(bucket)

    def _feed_voice(self, chunk: AudioChunk, out: list) -> None:
        """Enhanced target audio -> voice estimator (at most every ``VOICE_EVERY_S``)."""
        if self._voice_feed is None:
            return
        out.extend(self._poll_voice())
        track = self.ctx.target.track_id
        if self._voice_feed is None or track is None:
            return
        if track != self._voice_track:
            self._refresh_voice()
        x = np.asarray(chunk.samples, dtype=np.float32)
        speaking = len(x) > 0 and float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) > VOICED_RMS
        if speaking and self.vision.live:  # with a camera, only the target's own speech counts
            p = next((t.p_speaking for t in self.ctx.tracks if t.track_id == track), 0.0)
            speaking = p >= SPEAKING_P
        self._voice_worker.track = track
        self._voice_feed.update(chunk, speaking=speaking)  # submits to the worker, never blocks

    def _poll_voice(self, timeout: float = 0.0) -> list[dict]:
        if self._voice_worker is None:
            return []
        try:
            result = self._voice_worker.poll(timeout)
        except Exception as e:  # noqa: BLE001  (no weights, model error: keep the face/default)
            self._voice_worker.close()
            self._voice_worker = self._voice_feed = None
            return [error_message("voice", f"voice estimator failed: {type(e).__name__}: {e}")]
        if result is not None and result[1] is not None:
            self.voice_matcher.update(result[0], voice=result[1])
            self._refresh_voice()
        return []

    def wait_voice(self, timeout: float = 1.0) -> list[dict]:
        """Apply the in-flight voice estimate, waiting up to ``timeout`` (tests, offline runs)."""
        return self._poll_voice(timeout)

    def on_face(self, track_id: int, image: np.ndarray,
                box: tuple[float, float, float, float]) -> list[dict]:
        """Vision hook: one face crop of ``track_id`` (RGB frame + normalised box).

        Updates that track's smoothed face estimate and, for the current target, the voice.
        Callers throttle (a few Hz per track is plenty). Returns ``error`` events, if any.
        """
        if self.face_estimator is None:
            return []
        try:
            estimate = self.face_estimator.estimate(image, box)
        except Exception as e:  # noqa: BLE001  (no weights, bad frame: voice-only from now on)
            self.face_estimator = None
            return [error_message("voice", f"face estimator failed: {type(e).__name__}: {e}")]
        if estimate is not None:
            self.voice_matcher.update(track_id, face=estimate)
            if track_id == self.ctx.target.track_id:
                self._refresh_voice()
        return []

    # ------------------------------------------------------------ messages

    def handle_text(self, text: str) -> list[dict]:
        try:
            msg = parse_message(json.loads(text))
        except json.JSONDecodeError as e:
            return [error_message("protocol", f"invalid JSON: {e}")]
        except ProtocolError as e:
            return [error_message("protocol", str(e))]
        errors: list[dict] = []
        if isinstance(msg, Hello):
            self._apply_hello(msg)
        else:
            errors = self._apply_control(msg)
        telemetry = self.telemetry()
        return [*errors, *self.poll_pref_errors(), *self._vision_events(), telemetry]

    def _apply_hello(self, hello: Hello) -> None:
        self.hello = hello
        if hello.input_device or hello.output_device:
            LAST_CLIENT_DEVICES.clear()
            LAST_CLIENT_DEVICES.update(client=hello.client, input=hello.input_device, output=hello.output_device,
                                       missing=list(hello.devices_missing), at=time.time())
        self._frame_ms = hello.frame_ms
        self._chunker.reset()
        if hello.sample_rate == PIPELINE_SR:
            self._to_16k = self._from_16k = None
            self._resample_ms = 0.0
        else:
            self._to_16k = StreamResampler(hello.sample_rate, PIPELINE_SR)
            self._from_16k = StreamResampler(PIPELINE_SR, hello.sample_rate)
            self._resample_ms = self._to_16k.delay_ms + self._from_16k.delay_ms
        self._preload_profiles()

    def _preload_profiles(self) -> None:
        """T-136: hand every saved profile's (target, negative) clips to the shared enroller once, so the
        enrolled stage prepares their embeddings before the first swap (keys 1/2, a Name Call snap)."""
        if self.enroller.preload or not list_profiles():
            return
        try:
            self.enroller.preload = profile_pairs()
        except (OSError, ValueError) as e:
            log.warning("voice profiles not preloaded (%s)", e)

    def _apply_control(self, control: Control) -> list[dict]:
        errors: list[dict] = []
        fields = control.model_fields_set
        if "profile" in fields and control.profile is not None:  # first: explicit fields override it
            errors.extend(self.set_profile(control.profile))
        if "mode" in fields and control.mode is not None and control.mode != self.mode:
            previous, current = self.mode, self.effective_mode
            self.mode = control.mode
            if self.mode == "auto":  # Auto starts where the user was, unless that was silent
                self.advisor.auto_mode = current if current in AUTO_MODES else "natural"
            self.advisor.on_user_mode(self.mode, previous)
            self.ctx.mode = self.effective_mode
            if self.profile == "judge" and self.mode == "natural" and "suppression" not in fields:
                self.ctx.suppression = JUDGE_NATURAL_SUPPRESSION / 100.0  # the judge's Natural slider
        if "suppression" in fields and control.suppression is not None:
            self.ctx.suppression = control.suppression / 100.0
        if "target" in fields:
            old = self.ctx.target
            if self.vision.live:  # the selector owns the target; it applies on the next frame
                if control.target is None:
                    self.vision.unlock()
                    self.ctx.target = TargetState(old.track_id, old.confidence, False, old.since_ts,
                                                   old.switch_seq)
                else:
                    self.vision.lock(control.target.lock_track_id)
                    self.ctx.target = self.vision.selector.state
            elif control.target is None:
                self.vision.unlock()  # the selector must not hold a lock the wearer released while dark
                self.ctx.target = TargetState(None, 0.0, False, self._last_ts, old.switch_seq)
            else:
                self.vision.lock(control.target.lock_track_id)  # survives until frames apply (T-107)
                self.ctx.target = TargetState(control.target.lock_track_id, old.confidence, True,
                                              self._last_ts,
                                              old.switch_seq + int(old.track_id not in (None, control.target.lock_track_id)))
        if "attention" in fields and control.attention is not None:
            self._apply_attention(control.attention)
        if "hq_backend" in fields and control.hq_backend is not None:
            errors.extend(self.set_hq_backend(control.hq_backend))
        if "latency_preset" in fields and control.latency_preset is not None:
            errors.extend(self.set_latency_preset(control.latency_preset))
        if "synth_feed" in fields and control.synth_feed is not None:
            errors.extend(self.set_synth_feed(control.synth_feed))
        if "isolation" in fields and control.isolation is not None:
            errors.extend(self.set_isolation(control.isolation))
        if "enroll" in fields and control.enroll is not None:
            errors.extend(self.set_enroll(control.enroll.action, control.enroll.seconds,
                                          control.enroll.name, control.enroll.other))
        if "name" in fields and control.name is not None:
            self._name = (control.name.text, list(control.name.aliases))
            if control.name.match is not None:
                self._name_match = control.name.match
            if control.name.exclude is not None:
                self._name_exclude = list(control.name.exclude)
            if self.namecall is not None:
                self.namecall.set_name(*self._name)
                self.namecall.set_match(self._name_match, self._name_exclude)
        if "name_call" in fields and control.name_call == "release" and self.namecall is not None:
            self.namecall.release("released by the wearer")
        if self.effective_mode == "name_call":
            self._ensure_namecall()
            if "mode" in fields and self.namecall.target is None:
                self.enroller.clear()  # a camera target's clip must not play before anyone says the name
        elif self.namecall is not None and self.namecall.target is not None:
            self.namecall.release("left Name Call")  # the next hop clears its clip from the enroller
        if "ab" in fields and control.ab is not None:
            self.ab = control.ab
        if "agc" in fields and control.agc is not None:
            self.agc_enabled = control.agc
        if "voice_override" in fields:
            try:
                voice_match.check_override(control.voice_override)
            except ValueError as e:  # keep the previous voice
                errors.append(error_message("voice", str(e)))
            else:
                self.voice_override = control.voice_override
        self._refresh_voice()
        return errors

    def _ensure_namecall(self) -> None:
        if self.namecall is None:
            kw: dict = dict(self._namecall_timers)  # the team preset's quiet/release timers (T-130)
            try:  # T-136: the name's voice snaps to the nearest saved profile
                saved = {n: load_profile(n) for n in list_profiles()}
            except (OSError, ValueError) as e:
                log.warning("name call: voice profiles not loaded (%s)", e)
                saved = {}
            if saved:
                kw["saved"] = saved
            if os.environ.get(NAME_PROMPT_ENV, "").strip() == "1":  # T-131: prompt + hit guard (+ profile gate)
                kw["prompt"] = True
                if saved:
                    kw["profiles"] = [np.concatenate(p) for p in saved.values()]
            self.namecall = NameCaller(*self._name, match=self._name_match, exclude=self._name_exclude, **kw)

    def _update_namecall(self, samples: np.ndarray) -> None:
        """Feed the Name Call listener (in its mode) and apply what it found: a clip from the name's
        utterance goes to the shared enroller as a calibrated clip (T-085 path); a release clears it."""
        if self.effective_mode == "name_call":
            self.namecall.feed(samples)
            sim = self._enrolled_similarity()
            self._namecall_sim = sim
            if sim is not None and sim >= NAME_TSE_HEARD_SIM and self._pipeline_tier == "enrolled":
                self.namecall.heard()
        for event in self.namecall.poll():
            if event.kind == "enroll":
                if event.profile is not None:  # T-136: a saved profile (its embeddings prepared at hello)
                    self.enroller.enroll_profile(event.profile, event.pieces, event.other_pieces)
                    self.ctx.telemetry.extras["voice_profile"] = {"name": event.profile}
                else:
                    self.enroller.enroll_clip(event.clip, event.other)
                    self.ctx.telemetry.extras.pop("voice_profile", None)
                d = event.detection
                if d is not None:
                    self._namecall_events.append({"event": "retarget" if event.retarget else "heard", "word": d.word,
                                                  "latency_ms": round(d.latency_ms, 1), "new_voice": d.new_voice,
                                                  "profile": event.profile})
            elif event.kind == "release":
                self.enroller.clear()
                self._namecall_events.append({"event": "released", "reason": event.reason})

    def _enrolled_similarity(self) -> float | None:
        """ECAPA cosine of the stream the playing enrolled window chose to the target's clip."""
        for stage in _nested(self._built.get(self._active_spec) if self._active_spec else None):
            sim = getattr(getattr(stage, "cand", None), "similarity", None)
            if sim is not None:
                return float(sim)
        return None

    def _namecall_status(self) -> dict | None:
        """``telemetry.name_call``: the listener's state and the pill (``state``): ``listening`` (no target
        yet), ``locking`` (heard the name, the Enrolled tier is not playing yet), ``isolated``, ``quiet``
        (isolated, but the target has not been heard for QUIET_S)."""
        if self.namecall is None:
            return None
        st = self.namecall.status()
        if self.effective_mode != "name_call":
            state = "off"
        elif not st["target"]:
            state = "locking" if st["pending"] else "listening"
        elif self._pipeline_tier != "enrolled":
            state = "locking"
        else:
            state = "quiet" if st["quiet"] else "isolated"
        st["state"] = state
        st["tse_similarity"] = None if self._namecall_sim is None else round(self._namecall_sim, 3)
        st["events"], self._namecall_events = self._namecall_events[-5:], []
        return st

    def _apply_attention(self, att) -> None:
        values = {k: getattr(att, k) for k in ("dwell_s", "hold_s", "switch_margin", "tie_margin", "hold_lock_s")
                  if getattr(att, k) is not None}
        applied = self.vision.selector.set_attention(**values)
        self._prefs_jobs.append(self._prefs_pool.submit(
            self.advisor.rec.log.append_attention, dict(applied), att.preset))

    def poll_pref_errors(self) -> list[dict]:
        """Drain completed preference writes on the session thread."""
        errors = []
        pending = []
        for job in self._prefs_jobs:
            if not job.done():
                pending.append(job)
                continue
            try:
                job.result()
            except OSError as exc:
                errors.append(error_message("prefs", f"cannot save attention preferences: {exc}"))
        self._prefs_jobs = pending
        return errors

    def handle_binary(self, data: bytes) -> list[bytes | dict]:
        try:
            frame = unpack_frame(data)
            if frame.kind == KIND_VIDEO_IN:
                if self.audio_only:  # T-168: no vision, no AV tier, nothing forwarded to the GPU
                    self.video_dropped += 1
                    return []
                self.video_frames += 1
                self._poll_vision()
                self.vision.submit(frame.seq, frame.capture_ts, frame.payload)
                self._forward_video(frame.seq, frame.capture_ts, frame.payload)
                return self._vision_events()
            if frame.kind != KIND_AUDIO_IN:
                raise ProtocolError(f"clients may only send audio (1) or video (2), got {frame.kind}")
            if len(frame.payload) % 2:
                raise ProtocolError(f"PCM16 payload has odd length {len(frame.payload)}")
        except ProtocolError as e:
            return [error_message("protocol", str(e))]
        return self._process_audio(frame.seq, frame.capture_ts, frame.payload)

    @property
    def audio_only(self) -> bool:
        """T-168: the team preset's ``audio_only`` pin (``config/demo-team.yaml``): camera frames are dropped."""
        return bool((self.preset or {}).get("audio_only"))

    def _forward_video(self, seq: int, capture_ts: float, jpeg: bytes) -> None:
        """Camera frames to the remote GPU session of any playing ``remote`` stage (its own
        vision and Focus HQ need them)."""
        playing = [self._built.get(self._active_spec)] if self._active_spec else []
        if self._outgoing is not None:
            playing.append(self._outgoing[1])
        for pipeline in playing:
            for stage in _nested(pipeline):
                client = getattr(stage, "client", None)  # a RemoteStage's connection
                if client is not None and getattr(client, "connected", False):
                    client.send_video(seq, capture_ts, jpeg)

    def _process_audio(self, seq: int, capture_ts: float, payload: bytes) -> list[bytes | dict]:
        start = time.perf_counter()
        out: list[bytes | dict] = []
        x = pcm16_to_float(payload)
        self._frame_ms = 1000.0 * len(x) / self.hello.sample_rate
        if self._to_16k is not None:
            x = self._to_16k.process(x)
        self._last_seq = seq
        self._last_ts = capture_ts
        self.ctx.telemetry.seq = seq
        self.vision.push_audio(x, capture_ts)
        self._poll_vision()  # fresh tracks/target before the gate runs; re-picks the voice
        hops = [self._process_hop(h, capture_ts, out) for h in self._chunker.push(x)]
        y = np.concatenate(hops) if hops else np.zeros(0, dtype=np.float32)
        if self._agc_on and len(y):
            if self.pre_agc_sink is not None and self._from_16k is None:
                self.pre_agc_sink(float_to_pcm16(y))
            y = self.out_agc.process_array(y)
        if self._from_16k is not None:
            y = self._from_16k.process(y)
        if len(y):
            out.append(pack_frame(KIND_AUDIO_OUT, seq, capture_ts, float_to_pcm16(y)))
        out.extend(self._vision_events())  # after the audio: errors never delay the reply
        synth = self._quiet_synth if self._quiet_on else self._synth()
        if synth is not None:
            out.extend({"type": "caption", **c} for c in synth.drain_captions())
        self._server_sum += 1000.0 * (time.perf_counter() - start)
        self._server_n += 1
        return out

    def _process_hop(self, samples: np.ndarray, capture_ts: float, out: list) -> np.ndarray:
        self._update_context(samples, out)
        self._update_hq()
        self._update_hq_pin(samples)
        if self.namecall is not None:
            self._update_namecall(samples)
        if self.effective_mode != "name_call":  # Name Call owns the enroller (no camera needed)
            self.enroller.update(samples, self.ctx)  # every mode: the clip is ready when Focus HQ starts
        self.auto_calibrator.update(len(samples) / PIPELINE_SR, self.ctx, self.enroller)  # judge profile
        chunk = AudioChunk(samples, PIPELINE_SR, self._hop_seq, capture_ts)
        self._hop_seq += 1
        while True:
            pipeline = self._pipeline(out)
            if pipeline is None:
                self._outgoing = None
                self._caption_tap = None
                return self._apply_quiet(samples, out)
            before = self._run_outgoing(chunk, out)
            current, timings, target_audio = chunk, [], None
            _clear_side_channels(self.ctx)
            self.ctx.telemetry.tier = ""
            try:
                for key, stage in zip(self._keys[self.active_spec], pipeline.stages, strict=True):
                    if isinstance(stage, SynthStage) and target_audio is None:
                        target_audio = current  # the enhanced target, before synthesis
                    t0 = time.perf_counter()
                    current = stage.process(current, self.ctx)
                    timings.append((key, 1000.0 * (time.perf_counter() - t0)))
            except Exception as e:  # noqa: BLE001  (a stage crashed: degrade, keep going)
                out.append(self._degrade(self.active_spec, stage.name, f"{type(e).__name__}: {e}"))
                continue
            for key, ms in timings:
                self._stage_sum[key] = self._stage_sum.get(key, 0.0) + ms
                self._stage_n[key] = self._stage_n.get(key, 0) + 1
            if target_audio is not None:
                self._feed_voice(target_audio, out)
            self._pipeline_tier = self.ctx.telemetry.tier or None
            y = np.asarray(current.samples, dtype=np.float32)
            self._caption_tap = y
            return self._apply_quiet(self._crossfade(before, y, pipeline), out)

    def _apply_quiet(self, y: np.ndarray, out: list) -> np.ndarray:
        """T-149: Quiet's output gain ramp (digital silence within :data:`QUIET_FADE_MS`, also for
        whatever a crossfade or a priming pipeline would still play) and its captions."""
        quiet = self.quiet
        if quiet != self._quiet_on:  # entering or leaving: a phrase from before must not caption
            self._quiet_on = quiet
            if quiet and self._hop_seq <= 1:
                self._quiet_gain = 0.0  # Quiet from the start: nothing played yet, nothing to fade
            if self._quiet_synth is not None:
                self._quiet_synth.reset()
            self._quiet_feed = None
        if quiet:
            self._quiet_captions(len(y), out)
        goal = 0.0 if quiet else 1.0
        if self._quiet_gain == goal:
            return y if goal else np.zeros_like(y)
        step = 1000.0 / (QUIET_FADE_MS * PIPELINE_SR)
        ramp = self._quiet_gain + (step if goal else -step) * np.arange(1, len(y) + 1)
        gains = np.clip(ramp, 0.0, 1.0).astype(np.float32)
        self._quiet_gain = float(gains[-1]) if len(gains) else self._quiet_gain
        return (y * gains).astype(np.float32)

    def _quiet_captions(self, n: int, out: list) -> None:
        """Feed Quiet's captions-only synth with the Focus chain's output while its playing tier
        isolates the selected person; silence otherwise (never the room), and say why."""
        if self._quiet_synth is None and not self._quiet_synth_failed:
            try:
                self._quiet_synth = registry.build_pipeline(self.config.synth_only("text")).stages[-1]
            except Exception as e:  # noqa: BLE001  (no ASR engine: Quiet stays silent, captionless)
                self._quiet_synth_failed = True  # built once: the error goes out once, not every hop
                out.append(error_message("quiet", f"no captions: {e}"))
        tier = self._pipeline_tier
        if self._quiet_synth is None:
            reason = "no ASR engine: captions off"
        elif self._caption_tap is None or self._active_spec in set(self._substitute.values()):
            reason = f"Focus chain failed (playing {self.active_spec or 'raw'!r}): captions paused"
        elif tier not in ISOLATING_TIERS:
            reason = f"not isolating the selected person yet ({tier or 'no'} tier): captions paused"
        else:
            reason = ""
        feed = "paused" if reason else "focus"
        was = self._quiet_feed["feed"] if self._quiet_feed else None
        self._quiet_feed = {"feed": feed, "tier": tier, "reason": reason or f"captions from the {tier} tier"}
        if was == "focus" and feed == "paused":  # a degrade is never silent (telemetry carries it too)
            out.append({"type": "error", "stage": "quiet", "message": reason, "degraded_to": "paused"})
        if self._quiet_synth is None:
            return
        x = self._caption_tap if feed == "focus" and len(self._caption_tap) == n else np.zeros(n, np.float32)
        self._quiet_synth.process(AudioChunk(x, PIPELINE_SR, self._hop_seq, self._last_ts or 0.0), self.ctx)

    def _run_outgoing(self, chunk: AudioChunk, out: list) -> np.ndarray | None:
        """The old pipeline's output for this hop while a switch is in progress."""
        if self._outgoing is None:
            return None
        spec, old = self._outgoing
        _clear_side_channels(self.ctx)
        self.ctx.telemetry.tier = ""
        try:
            y = np.asarray(old.process(chunk, self.ctx).samples, dtype=np.float32)
            self._outgoing_tier = self.ctx.telemetry.tier
            return y
        except Exception as e:  # noqa: BLE001  (the old pipeline is going away anyway)
            log.info("outgoing pipeline %r failed during the switch: %s", spec, e)
            self._outgoing = None
            return None

    def _crossfade(self, before: np.ndarray | None, after: np.ndarray, pipeline: Pipeline) -> np.ndarray:
        """Old output until the new pipeline has filled its delay (and caught up, see
        :meth:`_caught_up`), then the ladder's fade for the jump in delay (module doc)."""
        if before is None or len(before) != len(after):
            self._outgoing = None if before is not None else self._outgoing
            return after
        n = len(after)
        if self._switch_fade is None:
            primed = (self._incoming_n >= round(pipeline.latency_ms * PIPELINE_SR / 1000.0)
                      and self._caught_up(pipeline))
            self._incoming_n += n
            if not primed:
                return before
            # Both delays as they are now: a ladder's is its playing tier's, a remote's its path's.
            delta = float(pipeline.latency_ms) - float(self._outgoing[1].latency_ms)
            self._switch_fade = fade_kind(delta)
            log.info("switch %r -> %r: %s fade (delay %+.0f ms)", self._outgoing[0],
                     self._active_spec, self._switch_fade, delta)
        old_gain, new_gain = fade_gains(self._switch_fade, self._fade_pos + np.arange(n), self._fade_n)
        self._fade_pos += n
        if self._fade_pos >= self._fade_n:
            self._outgoing = None
        return (old_gain * before + new_gain * after).astype(np.float32)

    def _caught_up(self, pipeline: Pipeline) -> bool:
        """Whether a new pipeline with a tier ladder plays as well as the old one: isolating the
        target (:data:`ISOLATING_TIERS`) if the old one was, and not mid tier fade. A latency
        preset click builds a new ladder whose enrolled tier needs one window (plus the ladder's
        ``upgrade_after_s``) before it is eligible; until then the old ladder keeps playing
        rather than handing over to the gate. Gives up after :data:`SWITCH_MATCH_S`."""
        ladder = _ladder(pipeline)
        if ladder is None or _synth_stages(pipeline) or self._incoming_n >= SWITCH_MATCH_S * PIPELINE_SR:
            return True  # nothing to wait for (Synthetic/Text never play the room anyway)
        if ladder._from is not None:  # the new ladder is still fading between its own tiers
            return False
        return self._outgoing_tier not in ISOLATING_TIERS or self._pipeline_tier in ISOLATING_TIERS

    def _transition(self) -> dict | None:
        if self._outgoing is None:
            return None
        pipeline = self._built.get(self._active_spec)
        need = round(pipeline.latency_ms * PIPELINE_SR / 1000.0) if pipeline is not None else 0
        return {
            "from": self._outgoing[0],
            "priming_ms": round(1000.0 * max(0, need - self._incoming_n) / PIPELINE_SR, 1),
            "fade": round(min(1.0, self._fade_pos / self._fade_n), 3) if self._fade_n else 1.0,
            "fade_kind": self._switch_fade,
        }

    def _update_context(self, samples: np.ndarray, out: list) -> None:
        """Measure the raw room; once per second of audio emit ``context`` and maybe a
        ``recommendation`` (or an Auto switch)."""
        vector = self.context.push(samples, self.ctx, self.effective_mode)
        if vector is None:
            return
        self._context_s += 1
        self._update_auto_feed(vector.speakers)
        out.append(vector.message(auto=self.mode == "auto"))
        out.extend(self.advisor.on_context(vector, self.mode, float(self._context_s)))
        self.ctx.mode = self.effective_mode

    # ------------------------------------------------------------ vision

    def _poll_vision(self, timeout: float = 0.0) -> None:
        """Apply finished video frames; their events wait for :meth:`_vision_events`."""
        self._vision_pending.extend(self.vision.poll(self.ctx, timeout))
        self._refresh_voice()
        self.same_bucket.update(self.ctx.tracks, lambda tid: self.voice_matcher.choice(tid).bucket,
                                float(self._last_ts or 0.0))
        self.same_bucket.publish(self.ctx)
        if not self._same_bucket_guard:  # team preset, verified M/F pair (T-130)
            self.ctx.telemetry.extras[SAME_BUCKET_KEY] = False
        elif self.enroller.profile_negative:
            # T-134: two saved voices, the closed set tells them apart. T-155: read from the enroller, not
            # voice_profile.other (only the session-start pair set it, so a key 1/2 swap dropped to the gate)
            self.ctx.telemetry.extras[SAME_BUCKET_KEY] = False
        target = self.ctx.target
        if self._lock_on_start and target.track_id is not None and not target.locked:
            self._lock_on_start = False  # team preset: the first face attention picks becomes a lock
            self.vision.lock(target.track_id)
            self.ctx.target = self.vision.selector.state

    def _warm_face_estimator(self) -> None:
        """Load the face model on the vision worker (one blank-frame estimate, discarded)."""
        estimator = self.face_estimator
        if estimator is not None:
            try:
                estimator.estimate(np.zeros((360, 640, 3), np.uint8), (0.25, 0.1, 0.5, 0.8))
            except Exception as e:  # noqa: BLE001  (on_face reports it on the session thread)
                log.debug("face estimator warm-up failed: %s", type(e).__name__)

    def _vision_events(self) -> list[dict]:
        events, self._vision_pending = self._vision_pending, []
        return events

    def wait_vision(self, timeout: float = 1.0) -> list[dict]:
        """Apply the in-flight video frame, waiting up to ``timeout`` (tests, offline runs)."""
        self.vision.wait(timeout)
        self._poll_vision()
        return self._vision_events()

    def _tier(self, pipeline: Pipeline | None) -> str:
        if not self.vision.live:
            return DEFAULT_TIER
        names = [s.name.split("(")[0] for s in pipeline.stages] if pipeline is not None else []
        return "gate" if "gate" in names else "av"

    # ------------------------------------------------------------ telemetry

    def _enrollment_status(self) -> dict:
        """The shared enroller for the UI's Calibrate button (T-085): ``state`` is ``calibrating``,
        ``calibrated``, ``ready`` (automatic clip), ``collecting`` or ``failed`` (last calibration had
        too little speech)."""
        e = self.enroller
        done, total = e.calibration_progress
        state = ("calibrating" if e.calibrating else "calibrated" if e.calibrated else "ready" if e.ready
                 else "failed" if e.calibration_failed else "collecting")
        return {"state": state, "collected_s": round(e.collected_s, 2),
                "calibrate_elapsed_s": round(done, 2), "calibrate_s": round(total, 2),
                "source": e.source if e.ready or e.holding else None, "profile": e.profile,
                "profiles": self._saved_profile_names()}

    def _saved_profile_names(self) -> list[str]:
        """The saved voice profiles (T-136: /judge keys 1/2 pick the first two), listed once per session."""
        names = getattr(self, "_profile_names", None)
        if names is None:
            names = self._profile_names = list_profiles()
        return names

    def _acquisition(self, pipeline: Pipeline | None) -> dict:
        """Explain selection/enrollment without claiming visual selection isolates audio."""
        selector = self.vision.selector
        target_id = self.ctx.target.track_id
        enroller = next((s.enroller for s in _nested(pipeline)
                         if getattr(s, "enroller", None) is not None), None)
        if enroller is not None:
            current = (enroller, enroller.track_id, enroller.collected_s)
            previous = self._enrollment_previous
            if (previous and previous[0] is enroller and previous[2] > 0
                    and (current[1] != previous[1] or current[2] < previous[2])):
                self._enrollment_restarted = True
            self._enrollment_previous = current
            if enroller.ready and enroller.track_id == target_id:
                self._enrollment_restarted = False
        else:
            self._enrollment_previous = None
            self._enrollment_restarted = False

        if enroller is not None and enroller.calibrating:  # manual calibration (T-085) says so first
            done, total = enroller.calibration_progress
            state, detail = "acquiring", f"calibrating {done:.1f}/{total:.1f} s"
        elif self.audio_only and enroller is not None and enroller.ready and enroller.source == "profile":
            # T-329: no camera, so no face target: the saved voice is the lock, once Enrolled plays it
            state, detail = self._profile_acquisition(pipeline, enroller.profile)
        elif selector.lock_lost:  # T-107: an explicit lock never falls back to free selection
            state, detail = "searching", LOCK_LOST_DETAIL
        elif (target_id is not None or self.ctx.target.locked) and selector.missing_s > 0:
            state, detail = "reacquiring", f"target lost {selector.missing_s:.1f} s"
        elif selector.candidate_id is not None:
            state, detail = "candidate", "waiting for face dwell"
        elif target_id is None:
            state, detail = "searching", "looking for a speaking face"
        elif self._pipeline_tier in {"focus_hq", "av"}:
            state, detail = "locked", "target isolation active"
        elif enroller is not None and (not enroller.ready or enroller.track_id != target_id):
            state = "reacquiring" if self._enrollment_restarted else "acquiring"
            collected = enroller.collected_s if enroller.track_id == target_id else 0.0
            detail = f"enrolling {collected:.1f}/{enroller.need / enroller.sr:.1f} s"
            hint = getattr(enroller, "hint", "") if enroller.track_id == target_id else ""
            if hint:  # T-092: what the collection is waiting for
                detail += f" · {hint}"
        elif self._pipeline_tier in {"enrolled", "focus_hq", "av"}:
            state, detail = "locked", "calibrated" if enroller is not None and enroller.calibrated else "target isolation active"
            if enroller is not None and enroller.source == "profile":  # T-125: the /judge pill names it
                detail = f"known voice: {enroller.profile}"
        else:
            state, detail = "locked-visual", "face selected; waiting for isolation"
        key = (state, target_id, selector.candidate_id)
        now = time.monotonic()
        if key != self._acquisition_key:
            self._acquisition_key, self._acquisition_since = key, now
        return {"state": state, "since_s": round(max(0.0, now - self._acquisition_since), 3),
                "detail": detail}

    def _profile_acquisition(self, pipeline: Pipeline | None, name: str | None) -> tuple[str, str]:
        """T-329 (audio_only): ``locked`` / ``known voice: <name>`` only while the Enrolled tier plays;
        ``acquiring`` / ``loading saved voice: <name>`` while it loads or warms up (the cold start);
        ``acquiring`` / ``saved voice <name> not playing: <why>`` when the ladder holds it out."""
        if self._pipeline_tier == "enrolled":
            return "locked", f"known voice: {name}"
        ladder = _ladder(pipeline)
        tiers = ladder.status()["tiers"] if ladder is not None and ladder.active else []
        why = next((t["reason"] for t in tiers if t["name"] == "enrolled"), "")
        if why in PROFILE_LOADING_REASONS:
            return "acquiring", f"loading saved voice: {name}"
        return "acquiring", f"saved voice {name} not playing: {why}"

    def telemetry(self) -> dict:
        """The wire ``telemetry`` event; per-stage compute is averaged since the last call.

        ``latency_ms``: one key per stage (mean compute per 10 ms hop), ``algorithmic``
        (sum of stage lookahead), ``resample`` (stream resampler delay, in + out, 0 at
        16 kHz), ``server`` (mean wall time per client frame, decode to encode),
        ``total`` = algorithmic + resample + server + client frame duration (capture
        buffering). Network and playback buffering are only visible to the client.
        """
        if self._stage_n:
            self._latency = {k: self._stage_sum[k] / self._stage_n[k] for k in self._stage_n}
            self._stage_sum.clear()
            self._stage_n.clear()
        if self._server_n:
            self._server_ms = self._server_sum / self._server_n
            self._server_sum, self._server_n = 0.0, 0
        spec = self.active_spec
        pipeline = self._built.get(spec)
        algorithmic = float(pipeline.latency_ms) if pipeline is not None else 0.0
        latency = {k: round(v, 3) for k, v in self._latency.items()}
        latency["algorithmic"] = round(algorithmic, 3)
        latency["resample"] = round(self._resample_ms, 3)
        latency["server"] = round(self._server_ms, 3)
        total = algorithmic + self._resample_ms + self._server_ms + self._frame_ms
        latency["total"] = round(total, 3)
        self._poll_vision()  # its events go out with the next reply
        if self.vision.frames_in:
            latency["vision"] = round(self.vision.latency_ms(), 3)
        self.ctx.telemetry.tier = (self._pipeline_tier if pipeline is not None and self._pipeline_tier
                                   else self._tier(pipeline))
        ladder = _ladder(pipeline) if spec == self._active_spec else None
        synth = self._synth() if pipeline is not None and spec == self._active_spec else None
        if synth is not None:
            stats = synth.latency_stats()
            if stats["n"] >= 1:
                latency["synth_phrase_p50"] = round(float(stats["p50"]), 3)
        self._refresh_voice()
        acquisition = self._acquisition(pipeline)
        self._isolated = acquisition["state"] == "locked"
        self._pin_hops(pipeline, acquisition)
        target = self.ctx.target
        target_state = {"track_id": target.track_id, "confidence": float(target.confidence),
                        "locked": bool(target.locked)}
        ours = self.ctx.telemetry.extras.get("tse") or {}
        # T-134 confidence; T-382 gated / soft_gated (the enrolled pick's last window) for the gate pill
        tse = {"confidence": ours.get("confidence"), "gated": ours.get("gated"), "soft_gated": ours.get("soft_gated")}
        remote = self._remote_report(latency)
        if remote is not None and remote["source"] != "local":
            # T-156: the GPU's audio plays, so what it isolates is the remote session's state
            theirs = remote.get("session") or {}
            if isinstance(theirs.get("target"), dict):
                target_state.update(confidence=float(theirs["target"].get("confidence") or 0.0),
                                    locked=bool(theirs["target"].get("locked")), source="remote")
            if isinstance(theirs.get("acquisition"), dict):
                acquisition = {**theirs["acquisition"], "source": "remote"}
            if isinstance(theirs.get("tse"), dict):
                tse = {"confidence": theirs["tse"].get("confidence"), "gated": theirs["tse"].get("gated"),
                       "soft_gated": theirs["tse"].get("soft_gated"), "source": "remote"}
        rtt = self.remote_rtt_ms()
        hq = self.hq_status()
        return {
            "type": "telemetry",
            "seq": self._last_seq,
            "tier": self.ctx.telemetry.tier,
            "acquisition": acquisition,
            "latency_ms": latency,
            "faces": [
                {"track_id": t.track_id, "box": [float(v) for v in t.box],
                 "p_speaking": float(t.p_speaking)}
                for t in self.ctx.tracks
            ],
            "target": target_state,
            "voice": voice_match.to_telemetry(self._voice_choice, self.voice_override),
            "vision": self.vision.telemetry(),
            "mode": self.mode,
            "effective_mode": self.effective_mode,
            "pipeline": spec,
            "device": self.device,
            "tiers": self._tiers_status(pipeline, ladder),
            "transition": self._transition(),
            "ab": self.ab,
            "attention": self.vision.selector.attention,
            "hq_backend": self.hq_backend,
            "latency_preset": self.latency_preset,
            "isolation": self.isolation,
            "enrollment": self._enrollment_status(),
            "tse": tse,
            "profile": self.profile,
            "agc": self.out_agc.status() if self._agc_on else None,  # T-135: output gain under the judge profile
            "auto_calibrate": self.auto_calibrator.status(),
            "preset": ({"name": "team", "pins": self._preset_pins, "video_dropped": self.video_dropped}
                       if self.preset is not None else None),
            "name_call": self._namecall_status(),
            "synth_feed": self.synth_feed,
            "quiet": dict(self._quiet_feed) if self.quiet and self._quiet_feed else None,  # T-149
            "synth_feed_request": self.synth_feed_request,
            "hq_active": hq["active"],
            "hq_reason": hq["reason"],
            "remote_rtt_ms": round(rtt, 3) if rtt is not None else None,
            "remote": remote,
        }

    def _remote_report(self, latency: dict) -> dict | None:
        """T-156: the playing ``remote`` stage's stats for the wire (``None`` without one): its
        :meth:`RemoteStage.stats` (rtt p50/p95, hops, misses, connected, remote latency and device,
        ``source``, ``remote_tier``: the remote ladder's playing tier) and ``session``, the remote
        session's own ``target``/``acquisition``/``tse``/``enrollment``. Adds ``latency_ms.remote_stage``
        (playout buffer + remote algorithmic while the GPU plays) and ``latency_ms.remote_rtt`` (p50)."""
        stage = self._remote_stage()
        if stage is None or stage.client is None:
            return None
        stats = stage.stats()
        latency["remote_stage"] = round(stage.latency_ms, 3)
        if stats["rtt_ms"]["p50"] is not None:
            latency["remote_rtt"] = stats["rtt_ms"]["p50"]
        # T-170: the code the GPU runs, from the last /health a monitor or the preset's probe read
        stats["git_rev"] = team_preset.remote_rev(stage.url)
        stats["rev_warning"] = team_preset.rev_warning(team_preset.local_rev(), stats["git_rev"])
        theirs = stage.client.remote_telemetry or {}
        stats["session"] = {k: theirs.get(k) for k in ("target", "acquisition", "tse", "enrollment")}
        return stats

    def _tiers_status(self, pipeline: Pipeline | None, ladder) -> dict | None:
        """The ``tiers`` block: while a ``remote`` stage plays remote audio, the remote ladder's
        (its playing tier, and ``delay_ms`` = buffer + remote delay, what the stream carries) rather
        than the local fallback ladder's, which runs but is not heard (T-108)."""
        if pipeline is not None and self.active_spec == self._active_spec:
            stage = next((s for s in _nested(pipeline) if isinstance(s, RemoteStage)), None)
            remote_tiers = stage.remote_tiers() if stage is not None else None
            if remote_tiers is not None:
                return remote_tiers
        return ladder.status() if ladder is not None and ladder.active else None

    def close(self) -> None:
        """Stop stage workers (synth TTS threads) and release engines; the session is done."""
        self._prefs_pool.shutdown(wait=True)
        for event in self.poll_pref_errors():
            log.warning("session error on close: %s", event)
        for pipeline in self._built.values():
            for stage in pipeline.stages:
                close = getattr(stage, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as e:  # noqa: BLE001  (closing must not raise into the server)
                        log.warning("closing stage %s failed: %s", stage.name, e)
        self._built.clear()
        self._keys.clear()
        self._active_spec = None
        self._outgoing = None
        if self._voice_worker is not None:
            self._voice_worker.close()
        if self._hq_monitor is not None:
            self._hq_monitor.close()
            self._hq_monitor = None
        if self._hq_reprobe is not None:
            self._hq_reprobe.close()
            self._hq_reprobe = None
        self.vision.close()
        if self.namecall is not None:
            self.namecall.close()
