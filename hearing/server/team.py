"""The 'team' demo preset (T-130): ``config/demo-team.yaml`` on top of the judge profile.

``HEARING_DEMO_PRESET=team`` (``scripts/demo.sh --judge --team``) turns every ``control.profile =
"judge"`` into the judge profile plus the preset's pins; ``control.profile = "team"`` asks for it
explicitly. :func:`team_fields` turns the YAML into ``control`` fields, :func:`team_spec` rewrites the
Mac ladder, :func:`pinned_backend` pins the Focus HQ backend: the Mac by default (T-181, T-129's verdict), and only
under ``HEARING_HQ_BACKEND=remote`` (``scripts/demo.sh --judge --team --gpu``) from a ``/health`` probe of the A100
(:func:`pick_backend`; a GPU-caused local pin is re-probed by :class:`Reprobe` and moves to remote at a pause, T-154), and
:func:`asr_env` gives the caption ASR its env. Preset off (no env, profile ``judge``): nothing here runs.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from hearing.registry import _parse_stage, _split

PRESET_ENV = "HEARING_DEMO_PRESET"
PRESETS = ("team",)
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
MF = {"male", "female"}
REPROBE_S = 10.0  # T-154: a GPU-caused local pin is re-probed this often
HQ_ENV = "HEARING_HQ_BACKEND"  # T-181: ``remote`` (demo.sh --gpu) makes the preset probe and pin the A100
MAC_REASON = "Mac (measured: beats the A100 with profiles, T-129)"
PAUSE_S = 0.3  # T-154: quiet this long (input below VOICED_RMS) counts as a pause for a backend switch
log = logging.getLogger(__name__)


def active_preset() -> str | None:
    """The preset named by ``$HEARING_DEMO_PRESET`` (``team``), else None (off)."""
    name = os.environ.get(PRESET_ENV, "").strip().lower()
    return name if name in PRESETS else None


def load_preset(name: str = "team", root: Path | None = None) -> dict:
    """``config/demo-<name>.yaml`` as a dict."""
    if name not in PRESETS:
        raise ValueError(f"unknown demo preset {name!r}; expected one of {PRESETS}")
    return yaml.safe_load(((root or CONFIG_DIR) / f"demo-{name}.yaml").read_text()) or {}


DEMO_INPUT = "MacBook Pro Microphone"  # T-143: the demo path's devices when the YAML does not name them
DEMO_OUTPUT = "AirPods"
BT_ROUND_TRIP_MAX_MS = 300.0
INPUT_ENV = "HEARING_INPUT_DEVICE"  # T-171: the demo mic's exact label, over config/demo-team.yaml's input_device


def input_device(preset: dict | None = None) -> tuple[str, str]:
    """T-171: (the demo mic label, where it came from): ``$HEARING_INPUT_DEVICE`` (exact label, e.g.
    ``Aryan patel's iPhone Microphone``, the iPhone over USB/Continuity), else ``input_device`` of
    ``config/demo-team.yaml``, else the built-in mic."""
    env = os.environ.get(INPUT_ENV, "").strip()
    if env:
        return env, INPUT_ENV
    if preset is None:
        preset = load_preset("team")
    if preset.get("input_device"):
        return str(preset["input_device"]), "config/demo-team.yaml"
    return DEMO_INPUT, "default"


def demo_devices(preset: dict | None = None) -> dict:
    """T-143: the demo mic and output labels (substrings) and the Bluetooth mic-to-ear bar, from
    ``config/demo-team.yaml`` (always read: the built-in mic + AirPods path is the only demo path);
    T-171: ``$HEARING_INPUT_DEVICE`` overrides the mic (:func:`input_device` says which won)."""
    if preset is None:
        preset = load_preset("team")
    return {"input": input_device(preset)[0],
            "output": str(preset.get("output_device") or DEMO_OUTPUT),
            "bt_round_trip_max_ms": float(preset.get("bt_round_trip_max_ms", BT_ROUND_TRIP_MAX_MS))}


def same_bucket_guard(preset: dict) -> bool:
    """True (guard on) unless the profiles' pair is a verified male/female pair."""
    pair = preset.get("pair") or {}
    genders = {(pair.get(k) or {}).get("gender") for k in ("target", "other")}
    return not (pair.get("verified") is True and genders == MF)


def health_url(remote_url: str) -> str:
    """``ws://host:port/ws/session`` -> ``http://host:port/health``."""
    scheme, rest = remote_url.split("://", 1)
    return f"{'https' if scheme in ('wss', 'https') else 'http'}://{rest.split('/', 1)[0]}/health"


REPO_ROOT = Path(__file__).resolve().parents[2]
REMOTE_REVS: dict[str, str] = {}  # T-170: health url -> the git_rev its /health last reported


def note_rev(url: str, health: dict) -> None:
    """Remember the ``git_rev`` a remote ``/health`` (``url``: its health or ws url) answered with."""
    rev = health.get("git_rev") if isinstance(health, dict) else None
    if rev:
        REMOTE_REVS[health_url(url) if "/ws/" in url else url] = str(rev)


def remote_rev(remote_url: str | None) -> str | None:
    """The remote's ``git_rev`` from the last ``/health`` any probe or monitor read, else None."""
    return REMOTE_REVS.get(health_url(remote_url)) if remote_url else None


def read_git_rev(root: Path) -> str:
    """``git rev-parse HEAD`` of ``root``; on a synced tree without ``.git`` (the GPU box, T-110) the
    ``REVISION`` file ``scripts/gpu_up.sh`` writes next to it; else ``unknown``."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=2,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return (root / "REVISION").read_text().strip() or "unknown"
    except OSError:
        return "unknown"


@functools.cache
def local_rev() -> str:
    """The laptop checkout's rev, read once (the code this process runs)."""
    return read_git_rev(REPO_ROOT)


def rev_warning(local: str | None, remote: str | None, root: Path | None = None) -> str | None:
    """T-170: ``"A100 rev <x> is behind local <y>: rerun gpu_up"`` when the local checkout ``root`` is
    a git repo and the local rev is not an ancestor of the remote's (``git merge-base --is-ancestor``);
    None when equal, the remote contains local, either rev is unknown, or git cannot tell."""
    local, remote = (str(r or "").removesuffix("-dirty") for r in (local, remote))
    if not local or not remote or "unknown" in (local, remote) or local == remote:
        return None
    if not _is_behind(local, remote, root or REPO_ROOT):
        return None
    return f"A100 rev {remote[:7]} is behind local {local[:7]}: rerun gpu_up"


@functools.lru_cache(maxsize=64)  # telemetry asks ~10 times a second; commits do not change
def _is_behind(local: str, remote: str, root: Path) -> bool:
    try:
        rc = subprocess.run(["git", "merge-base", "--is-ancestor", local, remote], cwd=root,
                            capture_output=True, timeout=2, check=False).returncode
    except (OSError, subprocess.SubprocessError):
        return False
    return rc == 1  # 0: the remote has local's commit; 128: not a repo or a commit this checkout lacks


def _probe_once(url: str, timeout: float) -> float | None:
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            health = json.loads(r.read() or b"{}")
            ok = r.status == 200 and health.get("ok") is True
        note_rev(url, health)
    except (OSError, urllib.error.URLError, ValueError, AttributeError):
        ok = False
    return (time.perf_counter() - t0) * 1000.0 if ok else None


def probe_rtts(remote_url: str, n: int = 5, timeout: float = 0.5, once=None) -> list[float | None]:
    """``n`` ``GET /health`` round trips in ms (None: failed or not ok, after one retry: T-154)."""
    url, once = health_url(remote_url), once or _probe_once
    out: list[float | None] = []
    for _ in range(n):
        rtt = once(url, timeout)
        out.append(rtt if rtt is not None else once(url, timeout))
        if sum(r is None for r in out) > n - (n // 2 + 1):  # no majority possible: stop waiting
            break
    return out + [None] * (n - len(out))


def pick_backend(rtts: list[float | None], p95_ms: float, min_ok: int | None = None) -> tuple[str, str]:
    """``remote`` when a majority of probes answered (``min_ok``, default 3 of 5) and the answered
    RTT p95 < ``p95_ms``, else ``local``; plus why (T-154: one blip no longer pins local)."""
    if not rtts:
        return "local", "no GPU url configured"
    ok = [r for r in rtts if r is not None]
    need = len(rtts) // 2 + 1 if min_ok is None else min_ok
    if len(ok) < need:
        return "local", f"GPU /health failed {len(rtts) - len(ok)}/{len(rtts)}"
    p95 = float(np.percentile(ok, 95))
    seen = f"GPU /health {len(ok)}/{len(rtts)} ok, RTT p95 {p95:.0f} ms"
    if p95 < p95_ms:
        return "remote", f"{seen} < {p95_ms:g} ms"
    return "local", f"{seen} >= {p95_ms:g} ms"


_PINNED: dict[str, tuple[str, str]] = {}  # remote url -> (backend, reason): one probe per server


def _verdict(remote_url: str, preset: dict, probe) -> tuple[str, str]:
    rtts = (probe or probe_rtts)(remote_url, int(preset.get("hq_probe_n", 5)))
    return pick_backend(rtts, float(preset.get("hq_probe_p95_ms", 60)))


def wants_gpu(preset: dict) -> bool:
    """T-181: probe and pin the A100 only when ``$HEARING_HQ_BACKEND`` is ``remote`` (``demo.sh --gpu``)
    or, with the env unset, the preset says ``hq_backend: remote``; else the Mac is pinned. T-129 measured
    the Mac ahead of every A100 variant with profiles loaded (the A100 only wins delay)."""
    env = os.environ.get(HQ_ENV, "").strip().lower()
    return (env or str(preset.get("hq_backend", "local")).strip().lower()) == "remote"


def pinned_backend(remote_url: str | None, preset: dict, probe=None) -> tuple[str, str]:
    """The Focus HQ backend for this server: the Mac unless :func:`wants_gpu` (no probe, even with the
    tunnel up), else probed once (later sessions reuse it). A local pin caused by the GPU is
    re-probed in the background by :class:`Reprobe` (T-154)."""
    if not wants_gpu(preset):
        return "local", MAC_REASON
    if not remote_url:
        return "local", "no GPU url configured"
    if remote_url not in _PINNED:
        _PINNED[remote_url] = _verdict(remote_url, preset, probe)
    return _PINNED[remote_url]


def safe_to_switch(isolated: bool, quiet_s: float, pause_s: float = PAUSE_S) -> bool:
    """A backend switch never cuts into an utterance: before Isolated, or at a pause of ``pause_s``."""
    return not isolated or quiet_s >= pause_s


class Reprobe:
    """Pinned local because of the GPU (T-154): probe again every ``every_s`` in a background thread.
    :attr:`ready` holds the remote verdict once one probe round picks remote (and re-pins the server);
    the session switches at its next safe point (:func:`safe_to_switch`)."""

    def __init__(self, remote_url: str, preset: dict, probe=None, every_s: float = REPROBE_S):
        self.url, self.preset, self.probe, self.every_s = remote_url, preset, probe, every_s
        self.ready: tuple[str, str] | None = None
        self.last: tuple[str, str] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def tick(self) -> tuple[str, str]:
        """One probe round (the thread calls this; tests call it directly)."""
        self.last = verdict = _verdict(self.url, self.preset, self.probe)
        if verdict[0] == "remote":
            self.ready = _PINNED[self.url] = verdict
        return verdict

    def _run(self) -> None:
        while self.ready is None and not self._stop.wait(self.every_s):
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001  (a probe bug must not kill the session)
                log.warning("GPU re-probe failed: %s", e)

    def start(self) -> Reprobe:
        self._thread = threading.Thread(target=self._run, name="hq-reprobe", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()


def team_fields(preset: dict, hq_backend: str | None, attention: dict) -> dict:
    """The ``control`` fields the preset adds on top of the judge profile."""
    fields: dict = {
        "attention": {**attention, "hold_lock_s": float(preset.get("hold_lock_s", 60))},
        "name": {"text": preset.get("name", ""), "aliases": list(preset.get("aliases") or []),
                 "match": preset.get("match", "strict")},
    }
    if preset.get("synth_feed"):
        fields["synth_feed"] = preset["synth_feed"]
    if preset.get("voice_override"):
        fields["voice_override"] = preset["voice_override"]
    if preset.get("isolation"):  # T-181: standard over the judge profile's strict (T-129 Q-470)
        fields["isolation"] = preset["isolation"]
    if hq_backend is not None:
        fields["hq_backend"] = hq_backend
    return fields


def team_spec(pipeline: str, preset: dict) -> str:
    """``pipeline`` with the preset's ``min_switch_s`` on every ``tiers`` ladder that does not set its own."""
    if "min_switch_s" not in preset:
        return pipeline
    stages = _split(pipeline, ">")
    for i, stage in enumerate(stages):
        name, kwargs = _parse_stage(stage)
        if name == "tiers" and "min_switch_s" not in kwargs:
            kwargs["min_switch_s"] = float(preset["min_switch_s"])
            stages[i] = f"{name}({', '.join(f'{k}={json.dumps(v)}' for k, v in kwargs.items())})"
    return ">".join(stages)


def asr_env(preset: dict) -> dict[str, str]:
    """The caption ASR's env under the preset (``hearing serve`` sets the ones not already set)."""
    env = {"HEARING_ASR": str(preset.get("asr", "local"))}
    if preset.get("caption_temperature") is not None:
        env["HEARING_ASR_TEMPERATURE"] = str(float(preset["caption_temperature"]))
    if preset.get("caption_prompt"):
        env["HEARING_ASR_PROMPT"] = str(preset["caption_prompt"])
        # the same words as T-136's caption vocabulary: a final made only of them near the room floor is dropped
        env["HEARING_ASR_VOCAB"] = ",".join(w.strip(" .") for w in str(preset["caption_prompt"]).split(","))
    return env


def apply_env(preset: dict | None = None) -> dict[str, str]:
    """Set :func:`asr_env` (and ``HEARING_VOICE_PROFILE``, ``HEARING_ENROLLED_CONTRAST``) where unset when the preset is on; what was set."""
    if preset is None:
        if active_preset() is None:
            return {}
        preset = load_preset(active_preset())
    env = asr_env(preset)
    if preset.get("enrolled_contrast") is not None:  # T-178: the enrolled pick floor (EnrolledCandidate.contrast)
        env["HEARING_ENROLLED_CONTRAST"] = str(float(preset["enrolled_contrast"]))
    target = ((preset.get("pair") or {}).get("target") or {}).get("profile")
    if target:
        env["HEARING_VOICE_PROFILE"] = str(target)
    applied = {k: v for k, v in env.items() if not os.environ.get(k)}
    os.environ.update(applied)
    return applied


def pins(preset: dict, backend: tuple[str, str], guard: bool, remote_git_rev: str | None = None,
         rev_warning: str | None = None) -> dict:
    """What the /judge drawer lists under TEAM PRESET (T-170: the GPU's rev and the stale-rev warning)."""
    return {
        "hq_backend": backend[0], "hq_reason": backend[1],
        "remote_git_rev": remote_git_rev, "rev_warning": rev_warning,
        "same_bucket_guard": "on" if guard else "off (verified M/F pair)",
        "lock_on_start": bool(preset.get("lock_on_start")), "hold_lock_s": preset.get("hold_lock_s"),
        "name": preset.get("name"), "aliases": list(preset.get("aliases") or []), "match": preset.get("match"),
        "namecall_quiet_s": preset.get("namecall_quiet_s"), "namecall_timeout_s": preset.get("namecall_timeout_s"),
        "isolation": preset.get("isolation"),
        "voice_override": preset.get("voice_override"), "min_switch_s": preset.get("min_switch_s"),
        "synth_feed": preset.get("synth_feed"), "asr": os.environ.get("HEARING_ASR") or preset.get("asr"),
        "caption_temperature": preset.get("caption_temperature"),
        "caption_prompt": bool(os.environ.get("HEARING_ASR_PROMPT")),
        "demo_noise_level_db": preset.get("demo_noise_level_db"),
        "input_device": input_device(preset)[0], "output_device": preset.get("output_device"),
        "bt_round_trip_max_ms": preset.get("bt_round_trip_max_ms"),
        "audio_only": bool(preset.get("audio_only")),  # T-168: /judge opens no camera, the session drops video
    }
