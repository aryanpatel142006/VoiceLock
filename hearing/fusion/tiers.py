"""The fallback ladder: Focus HQ (AV-TSE) -> Enrolled -> Gate -> Audio-only, crossfaded.

``tiers`` (``tiers(config="config/tiers.yaml", only="gate,audio-only")``) is one ``Stage``
holding one sub-pipeline per tier (``config/tiers.yaml``). Every chunk:

1. Every **hot** tier's pipeline processes the chunk: the playing tier, the one fading out, every
   better tier (promotion candidates) and the fallback below (the first eligible lower tier, else
   the last), plus every tier whose eligibility needs ``healthy`` or that holds a ``remote``
   stage (their health only moves while they are fed). The other tiers are **suspended**
   (T-116: feeding every tier every chunk let an idle ``dfn_ll`` take the CPU the playing tier
   needed, T-115) and, when they turn hot or are picked, **primed** first with the input they
   missed (the last ``prime_s``, default 0.5 s) chunk by chunk, so their state and delay line
   hold the right moment before they play and every switch stays sample aligned.
   ``idle: run`` (ladder-wide or per tier, or ``HEARING_TIERS_IDLE=run``) keeps a tier running.
   Each tier plays at **its own** delay (its pipeline's
   ``latency_ms``), and the ladder's ``latency_ms`` is the playing tier's: Focus HQ on the
   enrolled tier is 1.5 s behind on the Mac, not the 3 s of the cold-start AV tier.
2. :func:`eligible` checks each tier's needs against ``ctx`` and the tier's stages: ``camera``
   (face tracks present), ``target`` (the target track among them with
   ``confidence >= min_target_confidence``) and ``healthy`` (every stage with a ``health()``
   reports ``ok``: model loaded, GPU present, no overrun/late window, lips or enrollment
   present; a ``remote`` stage's server is up). A tier may also list conditions to **avoid**
   (``avoid_when``): ``same_bucket`` makes it ineligible while ``ctx.telemetry.extras["same_bucket"]``
   is true (the target and another face seen in the last 3 s share a voice group,
   :class:`hearing.synth.voice_match.SameBucket`). The Enrolled tier avoids it because ECAPA voice
   matching swaps two same-gender talkers (T-069: M/M 1.75 s, F/F 1.22 s of the other person per
   10 s), so the ladder plays AV Focus HQ (lips; built on CUDA only, T-072) or the gate (never swaps
   people) instead; the
   tier's reason reads ``same-gender pair: enrolled skipped``.
3. :class:`TierPolicy` picks the tier: a better tier must stay eligible for ``upgrade_after_s``
   (per tier or ladder-wide) and not be cooling before the ladder moves up, at most once per
   ``min_switch_s``; the playing tier is left at once on a hard failure and after
   ``downgrade_after_s`` when it only overruns (T-095: on the loaded Mac the old "leave at the
   first overrun" flapped 7-8 times a minute). The last tier should need nothing; if every
   tier is ineligible the last one plays anyway.
4. On a switch the output goes from the old tier to the new over ``crossfade_ms``; how depends
   on the jump in delay ``delta`` (:func:`fade_gains`, raised-cosine shapes):

   - ``aligned`` (``|delta|`` < 1 ms): the two outputs are versions of the same moment
     (correlated), so the gains sum to 1, which keeps the level.
   - ``splice`` (``|delta|`` <= :data:`DIP_ABOVE_MS`): different moments, roughly uncorrelated,
     so an equal-power crossfade (gains² sum to 1) keeps the loudness.
   - ``dip`` (``|delta|`` > :data:`DIP_ABOVE_MS`, e.g. gate 13 ms <-> enrolled 1.5 s): the old
     tier fades out over the first half and the new one fades in over the second, so the
     listener hears one clean cut instead of two different words on top of each other. Going
     to a longer delay replays ``delta`` of audio, going to a shorter one skips it: the jump is
     the price of not holding every tier at the slowest one's delay.

   No variant leaves a gap or a click longer than ``crossfade_ms``.

The ladder sets ``ctx.telemetry.tier`` to the playing tier and writes
``ctx.telemetry.extras["tiers"]`` (:meth:`TierLadder.status`): active tier, why, every tier's
eligibility and own ``delay_ms``, the playing delay, switch count, crossfade progress and kind,
each tier's stage health and ``compute_headroom`` (1 - compute/hop of the slowest tier). Health,
eligibility and the status are recomputed every :data:`STATUS_EVERY_S` (the telemetry rate), on a
switch and during a fade; the policy itself runs every chunk.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from hearing import device as hdevice
from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.fusion.mixer import ALIGNED_MIX_KEY, RESIDUAL_KEY
from hearing.latency_preset import (
    apply_adapt_hop,
    apply_isolation,
    apply_postfilter,
    apply_preset,
    enrolled_window,
)
from hearing.types import AudioChunk, StageContext

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "tiers.yaml"

TIERS = ("focus_hq", "enrolled", "gate", "audio-only")
NEEDS = frozenset({"camera", "target", "healthy"})
DEVICES = frozenset({"cuda", "mps", "cpu"})
AVOID = frozenset({"same_bucket"})
SAME_BUCKET_KEY = "same_bucket"  # = hearing.synth.voice_match.SAME_BUCKET_KEY (no synth import here)
DEFAULT_CROSSFADE_MS = 150.0
DEFAULT_UPGRADE_AFTER_S = 1.0
DEFAULT_MIN_TARGET_CONFIDENCE = 0.2  # centrality-based (TargetSelector); locked targets are 1.0
STATUS_KEY = "tiers"
ALIGNED_BELOW_MS = 1.0  # delays this close: one moment, amplitude crossfade
DIP_ABOVE_MS = 500.0  # delay jumps above this: fade to silence and back
FADES = ("aligned", "splice", "dip")
STATUS_EVERY_S = 0.1  # health/eligibility/status recomputed at the telemetry rate (10 Hz), not every 10 ms
IDLE = ("suspend", "run")  # what a tier that is not hot does (module doc, step 1)
IDLE_ENV = "HEARING_TIERS_IDLE"  # overrides the config's ladder-wide idle (A/B: run = pre-T-116)
DEFAULT_PRIME_S = 0.5  # input replayed into a suspended tier before it runs again
HOP_MS_WINDOW = 500  # chunks the ladder's own ms/hop p50/p95 are taken over


def compute_headroom(health: dict[str, list[dict]]) -> float | None:
    """``1 - utilisation`` of the slowest tier (stage ``utilisation`` = compute p50 / hop), or
    None while no stage has measured one. The ladder passes the running tiers only."""
    utils = [h["utilisation"] for reports in health.values() for h in reports
             if isinstance(h.get("utilisation"), (int, float)) and math.isfinite(h["utilisation"])]
    return round(1.0 - max(utils), 3) if utils else None


# ------------------------------------------------------------------ config


@dataclass(frozen=True)
class TierSpec:
    name: str
    pipelines: tuple[str, ...]  # first that builds wins
    needs: frozenset[str] = frozenset()
    devices: frozenset[str] = frozenset()  # build only on these hearing.device.pick() values; empty = any
    avoid: frozenset[str] = frozenset()  # conditions that make the tier ineligible (AVOID)
    upgrade_after_s: float | None = None  # this tier's promotion delay; None = the ladder's
    idle: str | None = None  # IDLE; None = the ladder's


@dataclass(frozen=True)
class LadderConfig:
    tiers: tuple[TierSpec, ...]
    crossfade_ms: float = DEFAULT_CROSSFADE_MS
    upgrade_after_s: float = DEFAULT_UPGRADE_AFTER_S
    min_target_confidence: float = DEFAULT_MIN_TARGET_CONFIDENCE
    downgrade_after_s: float = 0.0  # TierPolicy: soft failures (overrun/late) wait this long
    min_switch_s: float = 0.0  # TierPolicy: at most one promotion per this many seconds
    cooldown_s: float = 0.0  # TierPolicy: an overrun tier cools this long before promotion
    idle: str = "suspend"  # IDLE: what tiers that are not hot do
    prime_s: float = DEFAULT_PRIME_S  # input replayed into a suspended tier before it runs again

    @classmethod
    def from_dict(cls, data: dict) -> LadderConfig:
        tiers = []
        for item in data.get("tiers") or []:
            specs = item.get("pipeline")
            specs = (specs,) if isinstance(specs, str) else tuple(specs or ())
            needs = frozenset(item.get("needs") or ())
            devices = frozenset(item.get("devices") or ())
            avoid = item.get("avoid_when") or ()
            avoid = frozenset((avoid,) if isinstance(avoid, str) else avoid)
            if not item.get("name") or not specs:
                raise ValueError(f"a tier needs a name and a pipeline, got {item!r}")
            if needs - NEEDS:
                raise ValueError(f"tier {item['name']!r}: unknown needs {sorted(needs - NEEDS)}")
            if devices - DEVICES:
                raise ValueError(f"tier {item['name']!r}: unknown devices {sorted(devices - DEVICES)}")
            if avoid - AVOID:
                raise ValueError(f"tier {item['name']!r}: unknown avoid_when {sorted(avoid - AVOID)}")
            up = item.get("upgrade_after_s")
            idle = item.get("idle")
            if idle is not None and idle not in IDLE:
                raise ValueError(f"tier {item['name']!r}: idle must be one of {IDLE}, got {idle!r}")
            tiers.append(TierSpec(str(item["name"]), specs, needs, devices, avoid,
                                  None if up is None else float(up), idle))
        if not tiers:
            raise ValueError("the ladder needs at least one tier")
        if len({t.name for t in tiers}) != len(tiers):
            raise ValueError("tier names must be unique")
        cfg = cls(
            tuple(tiers),
            float(data.get("crossfade_ms", DEFAULT_CROSSFADE_MS)),
            float(data.get("upgrade_after_s", DEFAULT_UPGRADE_AFTER_S)),
            float(data.get("min_target_confidence", DEFAULT_MIN_TARGET_CONFIDENCE)),
            float(data.get("downgrade_after_s", 0.0)),
            float(data.get("min_switch_s", 0.0)),
            float(data.get("cooldown_s", 0.0)),
            str(data.get("idle", "suspend")),
            float(data.get("prime_s", DEFAULT_PRIME_S)),
        )
        if cfg.idle not in IDLE:
            raise ValueError(f"idle must be one of {IDLE}, got {cfg.idle!r}")
        times = [cfg.crossfade_ms, cfg.upgrade_after_s, cfg.downgrade_after_s, cfg.min_switch_s,
                 cfg.cooldown_s, cfg.prime_s] + [t.upgrade_after_s or 0.0 for t in tiers]
        if min(times) < 0:
            raise ValueError("crossfade_ms and the policy times must be >= 0")
        return cfg

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> LadderConfig:
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).expanduser().read_text()) or {})

    def only(self, names: list[str]) -> LadderConfig:
        unknown = set(names) - {t.name for t in self.tiers}
        if unknown:
            raise ValueError(f"unknown tiers {sorted(unknown)}; have {[t.name for t in self.tiers]}")
        return replace(self, tiers=tuple(t for t in self.tiers if t.name in names))


# ------------------------------------------------------------------ eligibility


def iter_stages(stage) -> Iterator:
    """``stage`` and everything nested in it (pipelines, ladders, a remote stage's fallback)."""
    yield stage
    for child in getattr(stage, "stages", ()) or ():
        yield from iter_stages(child)
    for tier in getattr(stage, "built", {}).values():
        yield from iter_stages(tier.pipeline)
    fallback = getattr(stage, "fallback", None)
    if fallback is not None and hasattr(fallback, "process"):
        yield from iter_stages(fallback)


def stage_health(pipeline) -> tuple[bool, str, list[dict]]:
    """``(ok, reason, reports)`` over the stages of one tier's pipeline that report health."""
    ok, reason, reports = True, "", []
    for stage in iter_stages(pipeline):
        if callable(getattr(stage, "health", None)):
            report = stage.health()
            reports.append(report)
            if not report.get("ok", False) and ok:
                ok, reason = False, str(report.get("reason") or f"{stage.name} not ready")
        elif callable(getattr(stage, "remote_up", None)):
            up = bool(stage.remote_up())
            reports.append({"stage": stage.name, "ok": up})
            if not up and ok:
                ok, reason = False, "remote GPU server down"
    return ok, reason, reports


def eligible(spec: TierSpec, ctx: StageContext, healthy: tuple[bool, str],
             min_confidence: float = DEFAULT_MIN_TARGET_CONFIDENCE) -> tuple[bool, str]:
    """Whether ``spec``'s needs are met and none of its ``avoid`` conditions holds now, and why not."""
    if "same_bucket" in spec.avoid and ctx.telemetry.extras.get(SAME_BUCKET_KEY):
        return False, f"same-gender pair: {spec.name} skipped"
    if "camera" in spec.needs and not ctx.tracks:
        return False, "no camera"
    if "target" in spec.needs:
        track = ctx.target.track_id
        if track is None or all(t.track_id != track for t in ctx.tracks):
            return False, "no target"
        if float(ctx.target.confidence) < min_confidence:
            return False, f"target confidence {ctx.target.confidence:.2f} < {min_confidence:g}"
    if "healthy" in spec.needs and not healthy[0]:
        return False, healthy[1] or "not ready"
    return True, ""


@dataclass
class TierPolicy:
    """Hysteresis over an ordered tier list (index 0 = best). Time is the audio clock in seconds.

    - **Promote** to a better tier only once it has been eligible for its ``upgrade_after_s``
      (one number, or one per tier) **and** is not cooling, and at least ``min_switch_s`` after
      the last switch.
    - **Demote** from the playing tier once it has been ineligible for ``downgrade_after_s``; a
      *hard* failure (crash, model missing, enrollment lost, same-gender pair: anything but an
      overrun/late window) leaves at once. Forced demotions ignore ``min_switch_s``.
    - **Cooling**: a tier that is ineligible because it overran (``overrun[i]``) may not be
      promoted until ``cooldown_s`` after its last overrun, so an overloaded tier that recovers
      for a second does not come back and fall out again (T-095).

    The defaults reproduce the old policy (leave at once, no rate limit, no cooling); the Mac
    ladder's values are in ``config/tiers.yaml``.
    """

    names: tuple[str, ...]
    upgrade_after_s: float | tuple[float, ...] = DEFAULT_UPGRADE_AFTER_S
    downgrade_after_s: float = 0.0
    min_switch_s: float = 0.0
    cooldown_s: float = 0.0
    current: int | None = None
    _ok_since: dict[int, float] = field(default_factory=dict)
    _bad_since: dict[int, float] = field(default_factory=dict)
    _cool_until: dict[int, float] = field(default_factory=dict)
    _last_switch: float | None = None

    def upgrade_after(self, i: int) -> float:
        up = self.upgrade_after_s
        return float(up[i]) if isinstance(up, (tuple, list)) else float(up)

    def cooling(self, i: int, now: float) -> float:
        """Seconds tier ``i`` still has to cool before it may be promoted (0 = not cooling)."""
        return max(0.0, self._cool_until.get(i, -math.inf) - now)

    def promotable(self, i: int, ok: list[bool], now: float) -> bool:
        return (ok[i] and now - self._ok_since[i] >= self.upgrade_after(i)
                and self.cooling(i, now) <= 0.0)

    def choose(self, ok: list[bool], now: float, hard: list[bool] | None = None,
               overrun: list[bool] | None = None) -> int:
        """The tier to play given each tier's eligibility at ``now``. ``hard[i]``: tier ``i``'s
        failure is hard (default: every failure is); ``overrun[i]``: it failed by overrunning."""
        hard = hard if hard is not None else [True] * len(ok)
        overrun = overrun if overrun is not None else [False] * len(ok)
        for i, good in enumerate(ok):
            if good:
                self._ok_since.setdefault(i, now)
                self._bad_since.pop(i, None)
            else:
                self._ok_since.pop(i, None)
                self._bad_since.setdefault(i, now)
                if overrun[i] and self.cooldown_s > 0:
                    self._cool_until[i] = now + self.cooldown_s
        best = next((i for i, good in enumerate(ok) if good), len(ok) - 1)
        cur = self.current
        if cur is None:
            self.current = best  # start: the best tier available now
        elif not ok[cur]:
            if hard[cur] or now - self._bad_since[cur] >= self.downgrade_after_s:
                # Leave the failed tier: the best one that is fine to play (a better one only if
                # it may be promoted), else the last tier.
                pick = next((i for i, good in enumerate(ok)
                             if good and (i > cur or self.promotable(i, ok, now))), len(ok) - 1)
                self._set(pick, now)
        elif best < cur and (self._last_switch is None or now - self._last_switch >= self.min_switch_s):
            ready = [i for i in range(cur) if self.promotable(i, ok, now)]
            if ready:
                self._set(ready[0], now)
        return self.current

    def _set(self, i: int, now: float) -> None:
        if i != self.current:
            self.current, self._last_switch = i, now

    def reset(self) -> None:
        self.current = None
        self._last_switch = None
        self._ok_since.clear()
        self._bad_since.clear()
        self._cool_until.clear()


SOFT_REASONS = ("overrun", "late")  # a tier failing only by these is overloaded, not broken


def soft_failure(reason: str) -> bool:
    """A tier ineligible for this ``reason`` may recover by itself (overrun / late window)."""
    return reason in SOFT_REASONS


# ------------------------------------------------------------------ crossfades


def fade_kind(delta_ms: float) -> str:
    """How to switch between two tiers whose delays differ by ``delta_ms`` (module doc, step 4)."""
    delta_ms = abs(delta_ms)
    if delta_ms < ALIGNED_BELOW_MS:
        return "aligned"
    return "splice" if delta_ms <= DIP_ABOVE_MS else "dip"


def fade_gains(kind: str, pos: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(old, new)`` gains at sample positions ``pos`` of an ``n``-sample transition."""
    u = np.clip((pos + 0.5) / max(n, 1), 0.0, 1.0)
    if kind == "aligned":
        new = np.sin(0.5 * np.pi * u) ** 2
        return 1.0 - new, new
    if kind == "splice":
        return np.cos(0.5 * np.pi * u), np.sin(0.5 * np.pi * u)
    if kind == "dip":
        old = np.cos(np.pi * np.clip(u, 0.0, 0.5)) ** 2  # 1 -> 0 over the first half
        new = np.sin(np.pi * (np.clip(u, 0.5, 1.0) - 0.5)) ** 2  # 0 -> 1 over the second
        return old, new
    raise ValueError(f"unknown fade {kind!r}; expected one of {FADES}")


# ------------------------------------------------------------------ stage


class _Built:
    """One tier: its pipeline, played at its own delay. ``always``: never suspended (its config
    says ``idle: run``, its eligibility needs ``healthy``, or it holds a ``remote`` stage)."""

    def __init__(self, spec: TierSpec, pipeline: registry.Pipeline, idle: str = "suspend"):
        self.spec, self.pipeline = spec, pipeline
        self.error: str | None = None
        self.always = ((spec.idle or idle) == "run" or "healthy" in spec.needs
                       or any(callable(getattr(s, "remote_up", None)) for s in iter_stages(pipeline)))
        self.running = True
        self.stopped_at: int | None = None  # ladder clock (samples) when it was suspended
        self.primes = 0

    @property
    def delay_ms(self) -> float:
        return float(self.pipeline.latency_ms)

    def run(self, chunk: AudioChunk, ctx: StageContext) -> np.ndarray:
        for key in (RESIDUAL_KEY, ALIGNED_MIX_KEY):  # never inherit another tier's side channel
            ctx.telemetry.extras.pop(key, None)
        if self.error is not None:  # disabled until the ladder is reset
            return np.zeros(len(chunk.samples), np.float32)
        try:
            y = np.asarray(self.pipeline.process(chunk, ctx).samples, dtype=np.float32)
        except Exception as e:  # noqa: BLE001  (a crashing tier is disabled, the ladder plays on)
            self.error = f"{type(e).__name__}: {e}"
            log.warning("tier %s failed: %s", self.spec.name, self.error)
            y = np.zeros(len(chunk.samples), np.float32)
        if len(y) != len(chunk.samples):
            y = np.resize(y, len(chunk.samples)) if len(y) else np.zeros(len(chunk.samples), np.float32)
        return y


@registry.register("tiers")
class TierLadder:
    """See the module doc. ``config``: YAML path (default ``config/tiers.yaml``) or a dict;
    ``only``: comma-separated tier names to keep (e.g. ``"gate,audio-only"``)."""

    sample_rate = PIPELINE_SR

    def __init__(self, config: str | dict | LadderConfig | None = None, only: str = "",
                 crossfade_ms: float | None = None, upgrade_after_s: float | None = None,
                 preset: str = "balanced", isolation: str = "standard", min_switch_s: float | None = None,
                 adapt_hop: bool = True):
        if isinstance(config, LadderConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = LadderConfig.from_dict(config)
        else:
            cfg = LadderConfig.load(config or DEFAULT_CONFIG)
        if only:
            cfg = cfg.only([n.strip() for n in only.split(",") if n.strip()])
        enrolled_window(preset)
        cfg = replace(cfg, tiers=tuple(
            replace(t, pipelines=tuple(apply_postfilter(apply_adapt_hop(apply_isolation(apply_preset(p, preset), isolation), adapt_hop))
                                       for p in t.pipelines))
            for t in cfg.tiers))
        if crossfade_ms is not None:
            cfg = replace(cfg, crossfade_ms=float(crossfade_ms))
        if upgrade_after_s is not None:
            cfg = replace(cfg, upgrade_after_s=float(upgrade_after_s))
        if min_switch_s is not None:  # T-120: the Name Call ladder re-targets without the 10 s hold
            cfg = replace(cfg, min_switch_s=float(min_switch_s))
        env_idle = os.environ.get(IDLE_ENV, "").strip()
        if env_idle:
            if env_idle not in IDLE:
                raise ValueError(f"{IDLE_ENV} must be one of {IDLE}, got {env_idle!r}")
            cfg = replace(cfg, idle=env_idle, tiers=tuple(replace(t, idle=None) for t in cfg.tiers))
        self.config = cfg
        self.unavailable: dict[str, str] = {}
        pipelines: list[tuple[TierSpec, registry.Pipeline]] = []
        for spec in cfg.tiers:
            pipeline = self._build(spec)
            if pipeline is not None:
                pipelines.append((spec, pipeline))
        if not pipelines:
            raise ValueError(f"no tier could be built: {self.unavailable}")
        for spec, pipeline in pipelines:
            if pipeline.sample_rate != self.sample_rate:
                raise ValueError(f"tier {spec.name} runs at {pipeline.sample_rate} Hz, not 16 kHz")
        self.built = {spec.name: _Built(spec, pipeline, cfg.idle) for spec, pipeline in pipelines}
        self.order = tuple(self.built)
        self.name = "tiers(" + ",".join(self.order) + ")"  # the tiers that run on this host
        self.policy = TierPolicy(
            self.order,
            tuple(cfg.upgrade_after_s if t.spec.upgrade_after_s is None or upgrade_after_s is not None
                  else t.spec.upgrade_after_s for t in self.built.values()),
            cfg.downgrade_after_s, cfg.min_switch_s, cfg.cooldown_s)
        self._fade_n = round(cfg.crossfade_ms * self.sample_rate / 1000.0)
        self.reset()

    def _build(self, spec: TierSpec) -> registry.Pipeline | None:
        if spec.devices:
            device = hdevice.pick()
            if device not in spec.devices:
                # Not built at all: a tier that can never keep up must not eat the machine (T-072).
                self.unavailable[spec.name] = f"needs {'/'.join(sorted(spec.devices))}, this host is {device}"
                log.info("tier %s skipped: %s", spec.name, self.unavailable[spec.name])
                return None
        errors = []
        for text in spec.pipelines:
            try:
                return registry.build_pipeline(text)
            except Exception as e:  # noqa: BLE001  (unknown stage, missing weights: try the next)
                errors.append(f"{text!r}: {type(e).__name__}: {e}")
        self.unavailable[spec.name] = "; ".join(errors)
        log.warning("tier %s unavailable: %s", spec.name, self.unavailable[spec.name])
        return None

    @property
    def latency_ms(self) -> float:
        """The playing tier's delay (before the first chunk: the slowest tier's, the safe bound)."""
        if self.active is not None:
            return self.built[self.active].delay_ms
        return max(tier.delay_ms for tier in self.built.values())

    # ------------------------------------------------------------ state

    def reset(self) -> None:
        for tier in self.built.values():
            tier.pipeline.reset()
            tier.error = None
            tier.running, tier.stopped_at, tier.primes = True, None, 0
        self.policy.reset()
        self._history: deque[tuple[int, AudioChunk]] = deque()  # (clock at its start, chunk)
        self._history_n = 0
        self._hop_ms: deque[float] = deque(maxlen=HOP_MS_WINDOW)  # ladder compute per 10 ms
        self.active: str | None = None
        self._from: str | None = None  # tier faded out of
        self._fade_kind = "aligned"
        self._fade_pos = 0
        self._clock = 0  # samples processed (the policy's clock)
        self.switches = 0
        self.reason = ""
        self._eligibility: dict[str, tuple[bool, str]] = {}
        self._soft: dict[str, bool] = {}  # not eligible for a soft (overrun/late) reason
        self._health: dict[str, list[dict]] = {}
        self._same_bucket = False
        self._checked_at = None  # clock of the last health/eligibility check
        self._status: dict | None = None
        self.history: deque[tuple[float, str, str]] = deque(maxlen=32)  # (t_s, tier, reason)

    def close(self) -> None:
        for tier in self.built.values():
            for stage in iter_stages(tier.pipeline):
                if stage is not tier.pipeline and callable(getattr(stage, "close", None)):
                    try:
                        stage.close()
                    except Exception as e:  # noqa: BLE001  (closing must not raise)
                        log.warning("closing %s failed: %s", stage.name, e)

    # ------------------------------------------------------------ Stage

    def hot(self) -> set[str]:
        """Tiers that run this chunk (module doc, step 1); the others are suspended."""
        always = {name for name, tier in self.built.items() if tier.always}
        if self.active is None:
            return set(self.order)
        a = self.order.index(self.active)
        hot = set(self.order[:a + 1]) | always  # playing + promotion candidates
        if self._from is not None:
            hot.add(self._from)
        below = self.order[a + 1:]
        if below:  # where TierPolicy falls on a hard failure: the first eligible lower tier, else the last
            hot.add(next((name for name in below if self._eligibility.get(name, (False, ""))[0]), below[-1]))
        return hot

    def _prime(self, tier: _Built, ctx: StageContext) -> None:
        """Feed a suspended tier the input it missed (at most ``prime_s``) so its state and delay
        line reach the current moment before it runs on the current chunk."""
        since = tier.stopped_at if tier.stopped_at is not None else -1
        for start, past in self._history:
            if start >= since:
                tier.run(past, ctx)
        tier.running, tier.stopped_at = True, None
        tier.primes += 1

    def _run(self, name: str, chunk: AudioChunk, ctx: StageContext) -> np.ndarray:
        tier = self.built[name]
        if not tier.running:
            self._prime(tier, ctx)
        return tier.run(chunk, ctx)

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        t0 = time.perf_counter()
        n = len(chunk.samples)
        self._same_bucket = bool(ctx.telemetry.extras.get(SAME_BUCKET_KEY))
        hot = self.hot()
        outs = {}
        for name, tier in self.built.items():
            if name in hot:
                outs[name] = self._run(name, chunk, ctx)
            elif tier.running:
                tier.running, tier.stopped_at = False, self._clock
        start = self._clock
        now = start / self.sample_rate
        self._clock += n
        refresh = (self._checked_at is None or now - self._checked_at >= STATUS_EVERY_S - 1e-9
                   or any(t.error is not None for t in self.built.values()))
        if refresh:  # health and eligibility at the telemetry rate; the policy runs every chunk
            self._checked_at = now
            for name, tier in self.built.items():
                healthy = stage_health(tier.pipeline)
                self._health[name] = healthy[2]
                if tier.error is not None:
                    verdict = (False, f"crashed: {tier.error}")
                else:
                    verdict = eligible(tier.spec, ctx, healthy[:2], self.config.min_target_confidence)
                self._eligibility[name] = verdict
                self._soft[name] = tier.error is None and soft_failure(verdict[1])
        ok = [self._eligibility[name][0] for name in self.order]
        hard = [not ok[i] and not self._soft[name] for i, name in enumerate(self.order)]
        overrun = [not ok[i] and self._soft[name] for i, name in enumerate(self.order)]
        pick = self.order[self.policy.choose(ok, now, hard, overrun)]
        switched = pick != self.active
        if switched:
            self._switch(pick, now)
            if pick not in outs:  # a suspended tier was picked: prime it before it plays
                outs[pick] = self._run(pick, chunk, ctx)
        self._history.append((start, chunk))
        self._history_n += n
        while self._history and self._history_n - len(self._history[0][1].samples) >= \
                self.config.prime_s * self.sample_rate:
            self._history_n -= len(self._history.popleft()[1].samples)
        y = self._mix(outs, n)
        ctx.telemetry.tier = self.active
        if refresh or switched or self._from is not None or self._status is None:
            self._status = self.status()
        ctx.telemetry.extras[STATUS_KEY] = self._status
        self._hop_ms.append((time.perf_counter() - t0) * 1000.0 * (PIPELINE_SR / 100) / max(n, 1))
        return replace(chunk, samples=y)

    def hop_ms(self) -> dict[str, float] | None:
        """The ladder's own compute per 10 ms hop (all running tiers, priming included), p50/p95
        over the last :data:`HOP_MS_WINDOW` chunks."""
        if not self._hop_ms:
            return None
        p50, p95 = np.percentile(np.asarray(self._hop_ms), [50, 95])
        return {"p50": round(float(p50), 3), "p95": round(float(p95), 3)}

    def _switch(self, pick: str, now: float) -> None:
        down = self.active is not None and self.order.index(pick) > self.order.index(self.active)
        if self.active is None:
            self.reason = "start"
        elif down:
            why = self._eligibility[self.active][1] or "unavailable"
            self.reason = why if why.startswith("same-gender") else f"{self.active}: {why}"
        else:
            self.reason = f"{pick} ready"
        if self.active is not None:
            self.switches += 1
            self._from = self.active if self._fade_n else None
            self._fade_kind = fade_kind(self.built[pick].delay_ms - self.built[self.active].delay_ms)
            self._fade_pos = 0
            log.info("tier %s -> %s (%s, %s fade)", self.active, pick, self.reason, self._fade_kind)
        self.active = pick
        self.history.append((round(now, 3), pick, self.reason))

    def _mix(self, outs: dict[str, np.ndarray], n: int) -> np.ndarray:
        new = outs[self.active]
        if self._from is None or self._from not in outs:
            return new
        old_gain, new_gain = fade_gains(self._fade_kind, self._fade_pos + np.arange(n), self._fade_n)
        y = old_gain * outs[self._from] + new_gain * new
        self._fade_pos += n
        if self._fade_pos >= self._fade_n:
            self._from = None
        return y.astype(np.float32)

    # ------------------------------------------------------------ telemetry

    def status(self) -> dict:
        """JSON-safe ladder state for telemetry."""
        now = self._clock / self.sample_rate
        return {
            "active": self.active,
            "reason": self.reason,
            "switches": self.switches,
            "fading_from": self._from,
            "fade": (min(1.0, self._fade_pos / self._fade_n) if self._from and self._fade_n else None),
            "fade_kind": self._fade_kind if self._from else None,
            "delay_ms": self.latency_ms,  # the playing tier's own delay
            "same_bucket": bool(self._same_bucket),  # target and another face share a voice group
            "compute_headroom": compute_headroom(  # 1 - utilisation of the slowest running tier
                {k: v for k, v in self._health.items() if self.built[k].running}),
            "hop_ms": self.hop_ms(),  # the ladder's compute per 10 ms hop (T-116)
            "suspended": [name for name in self.order if not self.built[name].running],
            "tiers": [
                {"name": name, "eligible": self._eligibility.get(name, (False, ""))[0],
                 "reason": self._eligibility.get(name, (False, ""))[1],
                 "delay_ms": self.built[name].delay_ms,
                 "cooling_s": round(self.policy.cooling(i, now), 2),
                 "running": self.built[name].running, "primes": self.built[name].primes,
                 "health": [_json_safe(h) for h in self._health.get(name, [])]}
                for i, name in enumerate(self.order)
            ] + [{"name": name, "eligible": False, "reason": f"unavailable: {why}", "delay_ms": None,
                  "health": []}
                 for name, why in self.unavailable.items()],
        }


def _json_safe(report: dict) -> dict:
    out = {}
    for k, v in report.items():
        if isinstance(v, float) and not math.isfinite(v):
            v = None
        elif isinstance(v, (np.floating, np.integer)):
            v = v.item()
        out[k] = v
    return out
