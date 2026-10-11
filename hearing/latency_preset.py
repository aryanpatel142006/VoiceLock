"""Focus HQ latency presets: speed / balanced / accuracy -> enrolled-tier window and hop.

The bake-off measured enrolled SI-SDRi as flat from 0.5 s to 2 s windows (docs/BENCHMARKS.md), so
the preset mostly trades delay (a tse stage plays window + hop behind) against robustness to
short windows. ``balanced`` is the configured spec as written: ``apply_preset`` leaves it alone,
so ``config/tiers.yaml`` (1.0 s / 0.5 s on the Mac ladder) and the GPU box's ``remote.enrolled``
(0.5 s / 0.25 s, served through ``REMOTE_PIPELINES``) keep their own windows (T-078). ``speed``
and ``accuracy`` rewrite every enrolled tier to the absolute windows below.
"""

from __future__ import annotations

import json
import os

from hearing.registry import _parse_stage, _split

PRESETS = ("speed", "balanced", "accuracy")
ISOLATIONS = ("standard", "strict")  # control.isolation (T-085)
DEFAULT_ISOLATION = "standard"
# "strict": the T-069 pause_duck_asd rule on the enrolled candidate. A window where no separated
# stream reaches cos 0.25 to the target while the target's ASD is below 0.5 plays silence (the
# mixer then plays only the attenuated room) instead of a guess: fewer wrong-person seconds, a few
# more dropouts (docs/decisions/association.md).
STRICT_KWARGS: dict = {"pause_floor": 0.25, "pause_p": 0.5, "pause_action": "duck"}
DEFAULT_PRESET = "balanced"

# preset -> (window_s, hop_s) of the enrolled tier; mic->ear delay is window + hop. "balanced" is
# what the Mac ladder ships; apply_preset never writes it (the configured windows are balanced).
ENROLLED_WINDOWS: dict[str, tuple[float, float]] = {
    "speed": (0.5, 0.25),
    "balanced": (1.0, 0.5),
    "accuracy": (1.5, 0.75),
}


def enrolled_window(preset: str) -> tuple[float, float]:
    if preset not in ENROLLED_WINDOWS:
        raise ValueError(f"unknown latency_preset {preset!r}; expected one of {PRESETS}")
    return ENROLLED_WINDOWS[preset]


def apply_preset(pipeline: str, preset: str) -> str:
    """Set enrolled TSE windows and ladder presets using the pipeline's scalar grammar.

    ``balanced`` (the default) keeps every configured window: only an explicit ``tiers(preset=...)``
    argument is rewritten back to it. ``speed`` / ``accuracy`` set :data:`ENROLLED_WINDOWS`."""
    window, hop = enrolled_window(preset)
    stages = _split(pipeline, ">")
    changed = False
    for i, stage in enumerate(stages):
        name, kwargs = _parse_stage(stage)
        if name == "tse" and kwargs.get("candidate") == "enrolled" and preset != DEFAULT_PRESET:
            kwargs.update(window_s=window, hop_s=hop)
        elif name == "tiers" and (preset != DEFAULT_PRESET or "preset" in kwargs):
            kwargs["preset"] = preset
        else:
            continue
        if kwargs:
            args = ", ".join(f"{key}={json.dumps(value)}" for key, value in kwargs.items())
            stages[i] = f"{name}({args})"
            changed = True
    return ">".join(stages) if changed else pipeline


def apply_isolation(pipeline: str, isolation: str) -> str:
    """``strict`` adds :data:`STRICT_KWARGS` to every enrolled ``tse`` stage and ``isolation="strict"``
    to every ``tiers`` ladder (which passes it to its own enrolled tiers); ``standard`` leaves the
    spec as configured (an explicit ``tiers(isolation=...)`` is rewritten back)."""
    if isolation not in ISOLATIONS:
        raise ValueError(f"unknown isolation {isolation!r}; expected one of {ISOLATIONS}")
    stages = _split(pipeline, ">")
    changed = False
    for i, stage in enumerate(stages):
        name, kwargs = _parse_stage(stage)
        if name == "tse" and kwargs.get("candidate") == "enrolled" and isolation == "strict":
            kwargs.update(STRICT_KWARGS)
        elif name == "tiers" and (isolation != DEFAULT_ISOLATION or "isolation" in kwargs):
            kwargs["isolation"] = isolation
        else:
            continue
        args = ", ".join(f"{key}={json.dumps(value)}" for key, value in kwargs.items())
        stages[i] = f"{name}({args})"
        changed = True
    return ">".join(stages) if changed else pipeline


POSTFILTER_ENV = "HEARING_TSE_POSTFILTER"  # T-180 P2: "1" puts ``dfn`` between every enrolled tse and its mixer


def apply_postfilter(pipeline: str, on: bool | None = None) -> str:
    """P2 (T-180, A/B knob, off by default): ``tse(candidate="enrolled", ...)>mixer`` becomes
    ``tse(...)>dfn>mixer`` (the ``enrolled_dfn`` spec of ``config/tiers-remote.yaml``). ``on=None``
    reads ``HEARING_TSE_POSTFILTER``; a spec that already has a stage after the tse is left alone."""
    if on is None:
        on = os.environ.get(POSTFILTER_ENV, "").strip() == "1"
    if not on:
        return pipeline
    stages = _split(pipeline, ">")
    out: list[str] = []
    for i, stage in enumerate(stages):
        out.append(stage)
        name, kwargs = _parse_stage(stage)
        nxt = _parse_stage(stages[i + 1])[0] if i + 1 < len(stages) else None
        if name == "tse" and kwargs.get("candidate") == "enrolled" and nxt == "mixer":
            out.append("dfn")
    return ">".join(out) if len(out) != len(stages) else pipeline


def apply_adapt_hop(pipeline: str, adapt_hop: bool) -> str:
    """``adapt_hop=False`` pins every ``tse`` stage's hop (no HOP_STEPS widening, T-135: each hop change
    was an audible dip/replay) and is passed down by every ``tiers`` ladder; ``True`` leaves the spec."""
    if adapt_hop:
        return pipeline
    stages = _split(pipeline, ">")
    changed = False
    for i, stage in enumerate(stages):
        name, kwargs = _parse_stage(stage)
        if name not in ("tse", "tiers") or kwargs.get("adapt_hop") is False:
            continue
        kwargs["adapt_hop"] = False
        args = ", ".join(f"{key}={json.dumps(value)}" for key, value in kwargs.items())
        stages[i] = f"{name}({args})"
        changed = True
    return ">".join(stages) if changed else pipeline
