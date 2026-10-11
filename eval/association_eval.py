"""Face <-> voice association go/no-go for the Enrolled tier (T-061).

The bake-off scored ``enrolled`` with 3 s of CLEAN target speech. Live, :class:`hearing.tse.enroll.AutoEnroller`
harvests the enrollment from the MIXED room signal while the active-speaker detector says the target
alone talks. This eval runs that live path end to end on two-face GRID scenes and measures whether the
extracted stream is the chosen person, and never the other one.

Scenes (``eval/scenes/association/*.yaml``, schema in its README) are rendered from the cached GRID
clips: person A on one half of a 640x360 frame, B on the other, ``mix = A + B + noise`` at ``--snr`` dB
(each talker at the same speech level, so SIR = 0 dB while both talk). ``--pair`` picks the talkers
(``PAIRS``: ``mf`` s1 M / s4 F, the T-061 default; ``mm`` s1 / s2; ``ff`` s4 / s7, added by T-069). Variant
``ab`` puts the pair's first talker as A on the left, ``ba`` the second as A on the right. ``--fetch``
range-downloads 10 clips of each talker of the pair that is not cached yet (~4 MB per talker).

Per scene, two passes:

1. **Vision**, once: every frame is JPEG-encoded and fed to the session's
   :class:`hearing.server.vision.VisionEngine` (MediaPipe faces + IoU tracker + ``--asd``, default
   Light-ASD), with the mix pushed as room audio exactly like a live session (``threaded=False`` so
   the run is deterministic). The target is the wearer's lock, applied as an oracle: the track on the
   chosen person's half, held for 1 s after it vanishes (``TargetSelector.hold_s``), re-locked on the
   first track that reappears there. Target *selection* is scored elsewhere (``eval/target_eval.py``).
2. **TSE**, per ``--configs`` entry: the real ``tse(candidate="enrolled")`` stage (MossFormer2_SS +
   ECAPA, window 1 s / hop 0.5 s as in ``config/tiers.yaml``, ``threaded=False``) fed 10 ms chunks with
   the session-level ``StageContext`` (tracks with ``p_speaking`` and the target) lagging one video frame
   behind the audio, as live. Separations are cached per window, so configs only re-run ECAPA.

Metrics (per scene, then mean and worst case over scenes; ``pooled`` sums windows/frames):

* ``acc`` association accuracy: over extracted windows in which the true target talks (>= 10 % of the
  window's energy at speech level), the chosen stream's SI-SDR against the true target beats its SI-SDR
  against the other talker (when the other talker is silent in the window: the chosen stream is the
  oracle stream, the one closest to the target).
* ``lock_s`` time to first correct lock: from the moment a target is set to the end of the first
  correct extracted window (``never`` if none).
* ``sw/min`` false speaker switches per minute while locked: correct -> wrong transitions over scored
  windows after the first correct one.
* ``wrong/10s`` wrong-target audio: seconds of extracted output (40 ms frames, fallback excluded) in
  which the other talker dominates (its least-squares share of the output exceeds the target's and is
  suppressed by less than 10 dB relative to the mix), per 10 s of extracted output. ``wrong_s`` is the
  raw seconds. ``unfocused_s``: output that fell back to the mix (enrolling, no TSE).
* ``reacq_s`` re-acquisition after the target's face comes back (end of ``away``/``occluded``/
  ``absent``): to the end of the first correct window starting after it.
* ``purity`` enrollment purity: share of the 10 ms chunks inside the enrollment clips actually used
  in which the enrolled track's person was the dominant voice (clean stems: its energy > the other's).

``--candidate av_mossformer2`` scores the AV tier instead (T-075): the vision pass also keeps the 88x88 lip
crops the VisionEngine cuts, the stage reads them from a ``LipFeed`` one frame late, window 2 s / hop 1 s;
``--configs`` and the enrollment metrics (``purity``, ``ready s``) do not apply.

GO when pooled ``acc >= 0.95`` and pooled ``wrong/10s < 0.5`` (``docs/decisions/association.md``).

    .venv/bin/python -m eval.association_eval --fast                  # 5 scenes, variant ab (< 5 min)
    .venv/bin/python -m eval.association_eval                         # 10 scenes x 2 variants
    .venv/bin/python -m eval.association_eval --configs baseline,final      # before / after T-061
    .venv/bin/python -m eval.association_eval --configs baseline,margin,purity,sticky,floor,clean,final
    .venv/bin/python -m eval.association_eval --pair mm --fetch --configs final,pause_hold --vision-cache /tmp/assoc-vc
    .venv/bin/python -m eval.association_eval --pair mm --candidate av_mossformer2 --vision-cache /tmp/assoc-vcl

Writes ``$HEARING_RESULTS_DIR/association_[<pair>_]<stamp>.json`` with ``hearing.device.describe()``,
rewritten after every scene (``"partial": true`` until the run ends).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import itertools
import json
import math
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from hearing.latency_preset import STRICT_KWARGS
from hearing.types import AudioChunk, FaceTrack, StageContext, TargetState

ROOT = Path(__file__).resolve().parents[1]
SCENES_DIR = ROOT / "eval" / "scenes" / "association"
SR = 16000
FPS = 25
SPF = SR // FPS  # 640 samples per video frame
HOP = SR // 100  # 10 ms chunks
LEVEL = 0.05  # speech RMS of each talker over its speech (-26 dBFS)
ACTIVE_SHARE = 0.1  # a talker is active in a span when its energy >= this share of speech level
WRONG_FRAME = SPF  # 40 ms frames for wrong-target audio
LEAK_DB = 10.0  # other talker suppressed by less than this (vs the mix) counts as audible
HOLD_S = 1.0  # TargetSelector.hold_s: a vanished locked track is held this long
KINDS = ("speak", "silent", "mouth", "away", "occluded", "absent")
RETURN_KINDS = ("away", "occluded", "absent")  # ending one is a re-acquisition event
# Speaker pairs (GRID talkers, sex from the corpus docs + median f0): T-061 ran ``mf`` only; T-069 added
# the same-gender pairs, since ECAPA separates a male and a female voice far more easily.
PAIRS = {"mf": ("s1", "s4"), "mm": ("s1", "s2"), "ff": ("s4", "s7")}
SPEAKERS = {"ab": PAIRS["mf"], "ba": PAIRS["mf"][::-1]}  # variant -> (A speaker, B speaker) for ``mf``
A_SIDE = {"ab": "left", "ba": "right"}
FAST_SCENES = ("overlap", "target_pauses", "occlusion", "mouth_no_sound", "switch")
GO_ACC = 0.95
GO_WRONG_PER_10S = 0.5


# ------------------------------------------------------------------------------------- scenes
@dataclass
class Scene:
    name: str
    description: str
    duration_s: float
    persons: dict[str, list[tuple[float, float, str]]]
    target: list[tuple[float, str]]

    @property
    def n_frames(self) -> int:
        return round(self.duration_s * FPS)


def _frames(t: float) -> int:
    f = t * FPS
    if abs(f - round(f)) > 1e-6:
        raise ValueError(f"time {t} is not a multiple of 1/{FPS} s")
    return round(f)


def parse_scene(data: dict) -> Scene:
    duration = float(data["duration_s"])
    _frames(duration)
    persons: dict[str, list[tuple[float, float, str]]] = {}
    for who in ("A", "B"):
        segs = [(float(a), float(b), str(k)) for a, b, k in data["persons"][who]]
        t = 0.0
        for a, b, k in segs:
            if k not in KINDS:
                raise ValueError(f"{data['name']}: unknown kind {k!r}")
            if abs(a - t) > 1e-9 or b <= a:
                raise ValueError(f"{data['name']}: segments of {who} must tile [0, {duration})")
            _frames(a), _frames(b)
            t = b
        if abs(t - duration) > 1e-9:
            raise ValueError(f"{data['name']}: segments of {who} end at {t}, not {duration}")
        persons[who] = segs
    target = [(float(t), str(p)) for t, p in data["target"]]
    if not target or target[0][0] != 0.0 or any(p not in ("A", "B") for _, p in target):
        raise ValueError(f"{data['name']}: target keyframes must start at 0 and name A or B")
    return Scene(str(data["name"]), str(data.get("description", "")), duration, persons, target)


def load_scenes(path: Path = SCENES_DIR) -> list[Scene]:
    return [parse_scene(yaml.safe_load(p.read_text())) for p in sorted(Path(path).glob("*.yaml"))]


def target_segments(scene: Scene) -> list[tuple[float, float, str]]:
    keys = scene.target
    return [(t, keys[i + 1][0] if i + 1 < len(keys) else scene.duration_s, p) for i, (t, p) in enumerate(keys)]


def return_events(scene: Scene) -> list[float]:
    """Times the current target's face comes back (end of away / occluded / absent)."""
    out = []
    for t0, t1, who in target_segments(scene):
        for a, b, k in scene.persons[who]:
            if k in RETURN_KINDS and t0 < b < t1:
                out.append(b)
    return out


# ------------------------------------------------------------------------------------- talkers
@dataclass
class Talker:
    """One person's material: a stream of talking frames + speech and a stream of idle frames."""

    speaker: str
    speech_frames: np.ndarray  # (N, H, W, 3)
    speech_audio: np.ndarray  # N * SPF samples at LEVEL
    idle_frames: np.ndarray  # (M, H, W, 3), mouth closed
    box: tuple[float, float, float, float]  # normalised face box in the native frame
    background: np.ndarray = field(default=None)  # (3,) uint8 colour behind the face

    def __post_init__(self):
        if self.background is None:
            self.background = np.median(self.idle_frames[0, :16, :16].reshape(-1, 3), axis=0).astype(np.uint8)


def grid_root() -> Path:
    """The GRID source cache: this checkout's, else the main checkout's (worktrees have none)."""
    local = ROOT / "eval" / "fixtures" / "generated" / "_sources" / "grid"
    if local.is_dir():
        return local
    try:
        common = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return local
    return Path(common).parent / "eval" / "fixtures" / "generated" / "_sources" / "grid"


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) if len(x) else 0.0


def load_talker(speaker: str, root: Path | None = None, rotate: int = 0) -> Talker:
    """Cut the cached GRID clips of ``speaker`` into speech and idle streams (no network)."""
    from eval.fixtures import grid as fxgrid
    from eval.make_fixtures import av_clip_from_grid

    root = Path(root) if root is not None else grid_root()
    corpus = fxgrid.GridCorpus(root)
    utts = sorted(p.stem for p in (root / speaker).glob("*.mpg"))
    if not utts:
        raise FileNotFoundError(f"no cached GRID clips under {root / speaker} (run eval.make_fixtures --av --full)")
    rotate %= len(utts)
    utts = utts[rotate:] + utts[:rotate]
    sf, sa, idle = [], [], []
    for u in utts:
        clip = av_clip_from_grid(corpus.load(speaker, u))
        if not clip.intervals:
            continue
        s0, e0 = clip.intervals[0][0], clip.intervals[-1][1]
        f0 = max(0, math.floor((s0 - 0.08) * FPS))
        f1 = min(len(clip.frames), math.ceil((e0 + 0.08) * FPS))
        audio = clip.utt.audio[f0 * SPF : f1 * SPF].astype(np.float32)
        speech = np.concatenate([clip.utt.audio[round(a * SR) : round(b * SR)] for a, b in clip.intervals])
        audio *= LEVEL / max(_rms(speech), 1e-6)
        sf.append(clip.frames[f0:f1])
        sa.append(audio)
        quiet = max(0, math.floor((s0 - 0.12) * FPS))
        if quiet >= 4:
            idle.append(clip.frames[:quiet])
    if not idle:
        idle = [sf[0][:1]]
    return Talker(speaker, np.concatenate(sf), np.concatenate(sa), np.concatenate(idle), fxgrid.face_box(speaker))


def fetch_talker(speaker: str, root: Path | None = None, n: int = 10, seed: int = 69) -> int:
    """Range-download ``n`` GRID clips of ``speaker`` (seeded pick) unless ``n`` are cached; returns bytes fetched."""
    from eval.fixtures import grid as fxgrid

    root = Path(root) if root is not None else grid_root()
    if len(list((root / speaker).glob("*.mpg"))) >= n:
        return 0
    corpus = fxgrid.GridCorpus(root)
    for utt in corpus.pick(speaker, n, np.random.default_rng(seed)):
        try:
            corpus.video_path(speaker, utt)
        except FileNotFoundError as e:  # a handful of GRID clips are missing from the release
            print(f"[fetch] skipping {speaker}/{utt}: {e}", file=sys.stderr)
    return corpus.bytes_downloaded


# ------------------------------------------------------------------------------------- render
FrameOp = tuple[str, int, str]  # (source "speech" / "idle", index, mod "" / "bg" / "cover")


@dataclass
class Rendered:
    """One scene x variant: audio stems and mix in memory, video frames composed on demand."""

    scene: Scene
    variant: str
    talkers: tuple[Talker, Talker]  # (A, B)
    plans: tuple[list[FrameOp], list[FrameOp]]  # per-frame ops of A and B
    stems: dict[str, np.ndarray]  # "A", "B", "noise"
    mix: np.ndarray
    a_side: str  # "left" / "right"

    @property
    def n(self) -> int:
        return len(self.mix)

    def side_of(self, who: str) -> str:
        return self.a_side if who == "A" else ("right" if self.a_side == "left" else "left")

    def frame(self, i: int) -> np.ndarray:
        """The 640x360 composite of video frame ``i``."""
        from eval.fixtures import video as fxvideo

        fa = person_frame(self.talkers[0], self.plans[0][i])
        fb = person_frame(self.talkers[1], self.plans[1][i])
        left, right = (fa, fb) if self.a_side == "left" else (fb, fa)
        return fxvideo.side_by_side(left[None], right[None])[0]

    def target_at(self) -> np.ndarray:
        """Per-sample index of the chosen person: 0 = A, 1 = B."""
        out = np.zeros(self.n, np.int8)
        for t0, t1, who in target_segments(self.scene):
            out[round(t0 * SR) : round(t1 * SR)] = 0 if who == "A" else 1
        return out


def _cover(frame: np.ndarray, box: tuple[float, float, float, float], pad: float = 0.15) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]
    x, y, bw, bh = box
    x0, x1 = int(max(0, (x - pad * bw) * w)), int(min(w, (x + bw * (1 + pad)) * w))
    y0, y1 = int(max(0, (y - pad * bh) * h)), int(min(h, (y + bh * (1 + pad)) * h))
    out[y0:y1, x0:x1] = 30
    return out


def person_frame(talker: Talker, op: FrameOp) -> np.ndarray:
    source, idx, mod = op
    f = (talker.speech_frames if source == "speech" else talker.idle_frames)[idx]
    if mod == "bg":
        return np.broadcast_to(talker.background, f.shape).copy()
    if mod == "cover":
        return _cover(f, talker.box)
    return f


def render_person(talker: Talker, segments: list[tuple[float, float, str]]) -> tuple[list[FrameOp], np.ndarray]:
    """``(per-frame ops, audio)`` of one person over the scene's segments."""
    ops: list[FrameOp] = []
    audio = []
    sp = ip = 0  # cursors into the speech and idle streams (continue across segments)
    ns, ni = len(talker.speech_frames), len(talker.idle_frames)
    pingpong = np.concatenate([np.arange(ni), np.arange(ni - 2, 0, -1)]) if ni > 2 else np.arange(ni)
    mods = {"away": "bg", "absent": "bg", "occluded": "cover"}
    for a, b, kind in segments:
        n = _frames(b) - _frames(a)
        if kind in ("speak", "mouth", "away", "occluded"):
            idx = (sp + np.arange(n)) % ns
            sp += n
            src = "speech"
            x = np.concatenate([talker.speech_audio[i * SPF : (i + 1) * SPF] for i in idx])
        else:
            idx = pingpong[(ip + np.arange(n)) % len(pingpong)]
            ip += n
            src = "idle"
            x = np.zeros(n * SPF, np.float32)
        if kind in ("mouth", "absent"):
            x = np.zeros_like(x)
        ops.extend((src, int(i), mods.get(kind, "")) for i in idx)
        audio.append(x)
    return ops, np.concatenate(audio).astype(np.float32)


def make_noise(n: int, kind: str, seed: int) -> np.ndarray:
    """Unit-RMS noise: ``babble`` (the committed tiny babble, looped from a seeded offset) or ``pink``."""
    rng = np.random.default_rng(seed)
    if kind == "pink":
        from eval.fixtures.sources import pink_noise

        return pink_noise(n, rng)
    from hearing.audio import io as aio

    base, _ = aio.load(ROOT / "eval" / "fixtures" / "tiny" / "noise_babble.wav", SR)
    reps = -(-(n + len(base)) // len(base))
    x = np.tile(base, reps)
    start = int(rng.integers(len(base)))
    x = x[start : start + n].astype(np.float32)
    return x / max(_rms(x), 1e-9)


def speakers(variant: str, pair: str = "mf") -> tuple[str, str]:
    """``(A speaker, B speaker)``: variant ``ab`` puts the pair's first talker as A, ``ba`` the second."""
    a, b = PAIRS[pair]
    return (a, b) if variant == "ab" else (b, a)


def render(scene: Scene, variant: str, talkers: dict[str, Talker], snr_db: float, noise: str, seed: int,
           pair: str = "mf") -> Rendered:
    ta, tb = (talkers[s] for s in speakers(variant, pair))
    pa, xa = render_person(ta, scene.persons["A"])
    pb, xb = render_person(tb, scene.persons["B"])
    nz = make_noise(len(xa), noise, seed) * (LEVEL / 10 ** (snr_db / 20))
    mix = (xa + xb + nz).astype(np.float32)
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    g = 0.95 / peak if peak > 0.95 else 1.0
    stems = {"A": xa * g, "B": xb * g, "noise": nz * g}
    return Rendered(scene, variant, (ta, tb), (pa, pb), stems, (mix * g).astype(np.float32), A_SIDE[variant])


# ------------------------------------------------------------------------------------- vision pass
@dataclass
class FrameState:
    tracks: list[FaceTrack]
    target_id: int | None
    sides: dict[int, str]  # track id -> left / right
    lips: list[tuple] = field(default_factory=list)  # (ts, track id, 88x88 crop, p) cut on this frame (--candidate av)


def _jpeg(image: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _side(t: FaceTrack) -> str:
    return "left" if t.box[0] + t.box[2] / 2 < 0.5 else "right"


class OracleLock:
    """The wearer's lock on a person: the track on that person's half, held ``HOLD_S`` after it vanishes."""

    def __init__(self):
        self.track_id: int | None = None
        self.seen = -1e9

    def update(self, tracks: list[FaceTrack], side: str, ts: float) -> int | None:
        ids = {t.track_id: t for t in tracks}
        if self.track_id is not None and _side_ok(ids.get(self.track_id), side):
            self.seen = ts
            return self.track_id
        fresh = [t for t in tracks if _side(t) == side]
        if fresh:
            self.track_id, self.seen = max(fresh, key=lambda t: t.last_seen_ts).track_id, ts
            return self.track_id
        if self.track_id is not None and ts - self.seen >= HOLD_S:
            self.track_id = None
        return self.track_id


def _side_ok(t: FaceTrack | None, side: str) -> bool:
    return t is not None and _side(t) == side


def run_vision(r: Rendered, faces: str = "mediapipe", asd: str = "light_asd",
               lips: bool = False) -> tuple[list[FrameState], dict]:
    """Faces, tracks, p_speaking and the oracle lock per video frame, through the session's VisionEngine.

    With ``lips`` every frame also keeps the mouth crops the engine cut for each track (the session's
    :class:`~hearing.tse.lips.LipFeed` entries), for the AV candidate.
    """
    from hearing.server.vision import VisionEngine
    from hearing.tse.lips import feed_of

    engine = VisionEngine(faces, asd, threaded=False, lips=lips)
    ctx = StageContext.default()
    feed = feed_of(ctx, create=True) if lips else None
    cursor = 0
    lock = OracleLock()
    seg = target_segments(r.scene)
    states: list[FrameState] = []
    t0 = time.perf_counter()
    try:
        for i in range(r.scene.n_frames):
            for c in range(SPF // HOP):
                pos = i * SPF + c * HOP
                engine.push_audio(r.mix[pos : pos + HOP], pos / SR)
            ts = (i + 1) / FPS - 0.005  # captured at the end of its 40 ms: the frame's audio is in
            engine.submit(i, ts, _jpeg(r.frame(i)))
            for ev in engine.poll(ctx):
                if ev.get("degraded_to"):
                    raise RuntimeError(f"vision degraded: {ev}")
            who = next(p for a, b, p in seg if a <= i / FPS < b)
            tracks = [replace(t) for t in ctx.tracks]
            tid = lock.update(tracks, r.side_of(who), ts)
            cut = []
            if feed is not None:
                for f in feed.since(cursor):
                    cursor = f.index + 1
                    cut.append((f.ts, f.track_id, f.crop, f.p_speaking))
            states.append(FrameState(tracks, tid, {t.track_id: _side(t) for t in tracks}, cut))
        info = {"asd": engine.asd.name if engine.asd is not None else asd, "faces": faces,
                "vision_s": round(time.perf_counter() - t0, 2)}
    finally:
        engine.close()
    return states, info


def asd_stats(r: Rendered, states: list[FrameState]) -> dict:
    """How the enrollment gate sees the scene: per truth case, mean p of each person and gate-open seconds.

    Truth per video frame from the clean stems: ``target`` / ``other`` / ``both`` / ``none`` talking.
    The gate is AutoEnroller's (target p >= TARGET_P, every other track < OTHER_P).
    """
    from hearing.tse.enroll import OTHER_P, TARGET_P

    tgt_at = r.target_at()
    stems = (r.stems["A"], r.stems["B"])
    cases: dict[str, dict] = {}
    for i, st in enumerate(states):
        sl = slice(i * SPF, (i + 1) * SPF)
        who = int(tgt_at[i * SPF])
        t_on, o_on = active(stems[who][sl]), active(stems[1 - who][sl])
        case = "both" if t_on and o_on else "target" if t_on else "other" if o_on else "none"
        p_t = max((t.p_speaking for t in st.tracks if t.track_id == st.target_id), default=0.0)
        p_o = max((t.p_speaking for t in st.tracks if t.track_id != st.target_id), default=0.0)
        c = cases.setdefault(case, {"s": 0.0, "p_target": 0.0, "p_other": 0.0, "gate_s": 0.0})
        c["s"] += 1 / FPS
        c["p_target"] += p_t
        c["p_other"] += p_o
        c["gate_s"] += (st.target_id is not None and p_t >= TARGET_P and p_o < OTHER_P) / FPS
    for c in cases.values():
        n = c["s"] * FPS
        c["p_target"], c["p_other"] = round(c["p_target"] / n, 2), round(c["p_other"] / n, 2)
        c["s"], c["gate_s"] = round(c["s"], 2), round(c["gate_s"], 2)
    return cases


# ------------------------------------------------------------------------------------- TSE pass
class CachedSeparator:
    """Wraps the real separator: one separation per distinct window, kept across configs."""

    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[bytes, np.ndarray] = {}
        self.last: np.ndarray | None = None
        self.calls = 0

    @property
    def device(self):
        return self.inner.device

    def load(self):
        self.inner.load()

    def separate(self, mix, sr=SR):
        key = np.asarray(mix, np.float32).tobytes()
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = self.inner.separate(mix, sr)
        self.last = self.cache[key]
        return self.last


class CachedEmbedder:
    def __init__(self, inner):
        self.inner = inner
        self.cache: dict[bytes, np.ndarray] = {}
        self.device = getattr(inner, "device", "cpu")

    def embed(self, audio):
        key = np.asarray(audio, np.float32).tobytes()
        if key not in self.cache:
            self.cache[key] = np.asarray(self.inner.embed(audio))
        return self.cache[key]


@dataclass
class Config:
    name: str
    candidate: dict = field(default_factory=dict)  # EnrolledCandidate keyword arguments
    enroller: dict = field(default_factory=dict)  # AutoEnroller keyword arguments
    calibrate_s: float = 0.0  # T-085: manual calibration of this many seconds of the target alone at t=0


# The Enrolled tier as it was before T-061; every other config changes one thing from it, except
# ``final`` (the new defaults of EnrolledCandidate / AutoEnroller: clean enrollment, margin 0.1,
# negative enrollment, 2 s, OTHER_P 0.7, sticky).
_OLD_CAND = {"clean_enroll": False, "margin": 0.0}
_OLD_ENR = {"seconds": 3.0, "other_p": 0.5, "other_s": 0.0, "sticky": False}


def _old(cand: dict | None = None, enr: dict | None = None) -> tuple[dict, dict]:
    return {**_OLD_CAND, **(cand or {})}, {**_OLD_ENR, **(enr or {})}


CONFIGS: dict[str, Config] = {
    "baseline": Config("baseline", *_old()),
    "margin": Config("margin", *_old({"margin": 0.1})),  # (a) hold the followed stream unless clearly better
    "purity": Config("purity", *_old({"purify": 0.4})),  # (b) drop enrollment pieces unlike the rest
    "sticky": Config("sticky", *_old(enr={"sticky": True})),  # (c) no re-enrollment when a face is re-detected
    "floor": Config("floor", *_old({"floor": 0.25})),  # silence when no stream reaches 0.25
    "clean": Config("clean", *_old({"clean_enroll": True})),  # embed the separated enrollment
    "fast_enroll": Config("fast_enroll", *_old(enr={"seconds": 2.0, "other_p": 0.7})),
    "negative": Config("negative", *_old(enr={"other_s": 2.0})),  # contrast against the other person
    "final": Config("final"),
    "other_hold": Config("other_hold", enroller={"other_hold_s": 0.2}),  # final + other face quiet 0.2 s
    # T-069 pause handling, each = final + one rule for windows where no stream reaches cos 0.25
    # (median best cosine 0.18 in final's misses vs 0.44 in its correct windows, M/F SNR 5 dB):
    "pause_hold": Config("pause_hold", {"pause_floor": 0.25}),  # keep following the previous stream
    "pause_hold_asd": Config("pause_hold_asd", {"pause_floor": 0.25, "pause_p": 0.5}),  # ... if ASD says quiet
    "pause_duck_asd": Config("pause_duck_asd", {"pause_floor": 0.25, "pause_p": 0.5, "pause_action": "duck"}),
    "pause_loud": Config("pause_loud", {"pause_floor": 0.25, "pause_action": "loud"}),  # play the louder stream
    # T-069 same-gender mitigations, each = final + one change (M/M final: negative ignored in 84 % of
    # windows because cos(target, other) > SAME_VOICE, and the 0.1 margin holds the wrong stream):
    "sg_margin": Config("sg_margin", {"margin": 0.05}),
    "sg_negative": Config("sg_negative", {"same_voice": 0.9}),  # keep a negative up to cos 0.9
    "sg_enroll3": Config("sg_enroll3", enroller={"seconds": 3.0}),
    "sg_combo": Config("sg_combo", {"margin": 0.05, "same_voice": 0.9}),
    # T-085 manual calibration (control.enroll): 3 s of the target talking alone, clean (another part of
    # the target's GRID material, never the scene's), fed through AutoEnroller.calibrate before t=0, as
    # the "Calibrate voice" button gives; "strict" = the control.isolation preset (STRICT_KWARGS).
    "calibrated": Config("calibrated", calibrate_s=3.0),
    "strict": Config("strict", dict(STRICT_KWARGS)),
    "calibrated_strict": Config("calibrated_strict", dict(STRICT_KWARGS), calibrate_s=3.0),
    "strict_c": Config("strict_c", {**STRICT_KWARGS, "contrast": 0.05}),
    "calibrated_strict_c": Config("calibrated_strict_c", {**STRICT_KWARGS, "contrast": 0.05}, calibrate_s=3.0),
}


@dataclass
class WindowRec:
    k: int
    start: int
    status: str
    reason: str
    y: np.ndarray | None = None
    streams: np.ndarray | None = None
    selected: int | None = None
    similarities: list[float] | None = None
    scores: list[float] | None = None
    paused: bool = False
    p_target: float | None = None  # mean p_speaking of the locked track over the window (vision lags a frame)


@dataclass
class TseRun:
    out: np.ndarray  # stage output aligned to the input timeline (first n - delay samples)
    delay: int
    window: int
    hop: int
    windows: list[WindowRec]
    clips: list[tuple[int | None, list[tuple[int, int]]]]  # finished enrollments: (track id, chunk spans)
    compute_s: float


def _recording_enroller(cls, clips: list, **kw):
    """An AutoEnroller whose every finished clip is appended to ``clips`` as ``(track id, spans)``."""

    class Recording(cls):
        def update(self, samples, ctx):
            before = self.ready
            kept = super().update(samples, ctx)
            if self.ready and not before:
                clips.append((self.track_id, list(self.spans)))
            return kept

    return Recording(**kw)


def _window_p(states: list[FrameState], start: int, length: int) -> float | None:
    ps = []
    for i in range(start // SPF, min(len(states), -(-(start + length) // SPF))):
        st = states[i - 1] if i > 0 else FrameState([], None, {})
        ps.extend(t.p_speaking for t in st.tracks if t.track_id == st.target_id)
    return round(float(np.mean(ps)), 3) if ps else None


def run_tse(r: Rendered, states: list[FrameState], cfg: Config, separator, embedder,
            window_s: float = 1.0, hop_s: float = 0.5) -> TseRun:
    from hearing.tse import enroll as henroll
    from hearing.tse.enrolled import EnrolledCandidate
    from hearing.tse.stage import FocusHQ

    cand = EnrolledCandidate(separator=separator, embedder=embedder, **cfg.candidate)
    calls: list[dict] = []
    inner_extract = cand.extract

    def extract(mix, sr, **kw):
        y = inner_extract(mix, sr, **kw)
        calls.append({"y": np.asarray(y, np.float32).copy(), "streams": separator.last,
                      "selected": cand.selected_stream, "sims": list(getattr(cand, "similarities", None) or []),
                      "scores": list(getattr(cand, "scores", None) or []),
                      "paused": bool(getattr(cand, "paused", False))})
        return y

    cand.extract = extract
    stage = FocusHQ(candidate="enrolled", window_s=window_s, hop_s=hop_s, threaded=False, residual=False,
                    _candidate=cand)
    clips: list = []
    kw = {"seconds": stage.enroller.need / SR, **cfg.enroller}
    stage.enroller = _recording_enroller(henroll.AutoEnroller, clips, sr=SR, **kw)
    stage.load()
    if stage.error:
        raise RuntimeError(stage.error)
    ctx = StageContext.default()
    if cfg.calibrate_s > 0:  # the wearer pressed Calibrate while the target spoke alone, before t=0
        first = target_segments(r.scene)[0][2] if target_segments(r.scene) else "A"
        talker = r.talkers[0 if first == "A" else 1]
        clip = talker.speech_audio[-round(cfg.calibrate_s * SR):]  # the scenes start at the head
        stage.enroller.calibrate(cfg.calibrate_s)
        ctx.target = TargetState(None, 0.0, False, 0.0)
        for pos in range(0, len(clip), HOP):
            stage.enroller.update(clip[pos : pos + HOP], ctx)
        if not stage.enroller.calibrated:
            raise RuntimeError("calibration clip had too little speech")
        clips.clear()  # scored purity covers automatic clips only (this one is clean by construction)
        stage.enroller.offered = 0
    return _drive(stage, r, states, ctx, calls, clips)


def run_av(r: Rendered, states: list[FrameState], candidate: str = "av_mossformer2",
           window_s: float = 2.0, hop_s: float = 1.0) -> TseRun:
    """The AV tier (``tse(candidate=...)`` with lips, as Focus HQ runs it) on the vision pass's lip crops.

    The crops reach the stage through a :class:`~hearing.tse.lips.LipFeed` in ``ctx`` one video frame
    after their audio, like a live session; the target is the oracle lock. No enrollment, no ECAPA.
    """
    from hearing.tse.lips import feed_of
    from hearing.tse.stage import FocusHQ

    stage = FocusHQ(candidate=candidate, window_s=window_s, hop_s=hop_s, threaded=False, residual=False)
    calls: list[dict] = []
    inner_extract = stage.cand.extract

    def extract(mix, sr, **kw):
        y = inner_extract(mix, sr, **kw)
        calls.append({"y": np.asarray(y, np.float32).copy(), "streams": None, "selected": None, "sims": [],
                      "scores": [], "paused": False})
        return y

    stage.cand.extract = extract
    stage.load()
    if stage.error:
        raise RuntimeError(stage.error)
    ctx = StageContext.default()
    return _drive(stage, r, states, ctx, calls, [], feed=feed_of(ctx, create=True))


def _drive(stage, r: Rendered, states: list[FrameState], ctx: StageContext, calls: list[dict], clips: list,
           feed=None) -> TseRun:
    """Feed the scene in 10 ms chunks with the vision state one frame late; collect every window."""
    outs = []
    windows: list[WindowRec] = []
    seen_k = 0
    t0 = time.perf_counter()
    for i in range(r.scene.n_frames):
        st = states[i - 1] if i > 0 else FrameState([], None, {})  # vision lags one frame
        if feed is not None:
            for ts, tid, crop, p in getattr(st, "lips", ()):
                feed.append(ts, tid, crop, p)
        ctx.tracks = st.tracks
        ctx.target = TargetState(st.target_id, 1.0 if st.target_id is not None else 0.0, st.target_id is not None, 0.0)
        for c in range(SPF // HOP):
            pos = i * SPF + c * HOP
            out = stage.process(AudioChunk(r.mix[pos : pos + HOP], SR, pos // HOP, pos / SR), ctx)
            outs.append(out.samples)
            while seen_k < stage._next_k:
                w = stage._windows[seen_k]
                rec = WindowRec(seen_k, seen_k * stage.hop, w.status, w.reason,
                                p_target=_window_p(states, seen_k * stage.hop, stage.window))
                if w.status == "ok":
                    call = calls.pop(0)
                    rec.y, rec.streams, rec.selected, rec.similarities = call["y"], call["streams"], call["selected"], call["sims"]
                    rec.scores, rec.paused = call["scores"], call["paused"]
                windows.append(rec)
                seen_k += 1
    compute = time.perf_counter() - t0
    full = np.concatenate(outs)
    stage.close()
    return TseRun(full[stage.delay :], stage.delay, stage.window, stage.hop, windows, clips, compute)


# ------------------------------------------------------------------------------------- metrics
def _si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    from eval.metrics import si_sdr

    return si_sdr(est, ref)


def _energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, np.float64) ** 2))


def active(x: np.ndarray) -> bool:
    """The talker carries >= ACTIVE_SHARE of speech-level energy over the span."""
    return _energy(x) >= ACTIVE_SHARE * LEVEL**2 * len(x)


def window_correct(y: np.ndarray, streams: np.ndarray | None, tgt: np.ndarray, oth: np.ndarray) -> bool:
    if active(oth):
        return _si_sdr(y, tgt) > _si_sdr(y, oth)
    if streams is None:
        return _si_sdr(y, tgt) > 0.0
    best = int(np.argmax([_si_sdr(s, tgt) for s in streams]))
    return bool(np.allclose(y, streams[best], atol=1e-6))


def ls_shares(y: np.ndarray, parts: list[np.ndarray]) -> list[float]:
    """Energy of each part's least-squares contribution to ``y``."""
    a = np.stack([np.asarray(p, np.float64) for p in parts], axis=1)
    gram = a.T @ a + 1e-9 * np.eye(a.shape[1])
    coef = np.linalg.solve(gram, a.T @ np.asarray(y, np.float64))
    return [float(np.sum((coef[i] * a[:, i]) ** 2)) for i in range(a.shape[1])]


def score(r: Rendered, run: TseRun, sides: dict[int, str]) -> dict:
    """Metrics of one TSE run; ``sides`` maps track ids to the half of the frame they were on."""
    tgt_at = r.target_at()
    stems = (r.stems["A"], r.stems["B"])
    W, H = run.window, run.hop
    n = len(run.out)
    covered = np.zeros(n, bool)
    rows = []
    for w in run.windows:
        a, b = w.start, w.start + W
        if w.status == "ok":
            covered[a : min(b, n)] = True
        who = int(tgt_at[min(a + W // 2, r.n - 1)])
        tgt, oth = stems[who][a:b], stems[1 - who][a:b]
        scored = w.status == "ok" and active(tgt)
        ok = window_correct(w.y, w.streams, tgt, oth) if scored else None
        muted = w.status == "ok" and not np.any(w.y)
        # the other person played: their voice is the largest part of the output (any window they talk in)
        other_played = False
        if w.status == "ok" and not muted and active(oth):
            e_t, e_o, e_n = ls_shares(w.y, [tgt, oth, r.stems["noise"][a:b]])
            other_played = e_o > max(e_t, e_n)
        rows.append({"k": w.k, "t": a / SR, "status": w.status, "reason": w.reason, "who": who,
                     "scored": scored, "correct": ok, "selected": w.selected,
                     "sims": [round(s, 3) for s in (w.similarities or [])],
                     "scores": [round(s, 3) for s in (w.scores or [])],
                     "tgt_active": active(tgt), "oth_active": active(oth), "muted": muted, "paused": w.paused,
                     "p_target": w.p_target, "tgt_share": round(_energy(tgt) / (LEVEL**2 * len(tgt)), 3),
                     "other_played": other_played,
                     "rms": [round(_rms(s), 4) for s in w.streams] if w.streams is not None else None,
                     # some stream would have been correct: misses with False are the separator's, not ECAPA's
                     "oracle_ok": any(window_correct(s, w.streams, tgt, oth) for s in w.streams)
                     if scored and w.streams is not None else None})
    scored = [x for x in rows if x["scored"]]
    acc = sum(x["correct"] for x in scored) / len(scored) if scored else None

    # wrong-target audio on the extracted output
    wrong_frames = focus_frames = 0
    for f0 in range(0, n - WRONG_FRAME + 1, WRONG_FRAME):
        sl = slice(f0, f0 + WRONG_FRAME)
        if not covered[sl].all():
            continue
        focus_frames += 1
        who = int(tgt_at[f0 + WRONG_FRAME // 2])
        tgt, oth = stems[who][sl], stems[1 - who][sl]
        if not active(oth):
            continue
        e_t, e_o, _ = ls_shares(run.out[sl], [tgt, oth, r.stems["noise"][sl]])
        if e_o > e_t and e_o > _energy(oth) * 10 ** (-LEAK_DB / 10):
            wrong_frames += 1
    wrong_s = wrong_frames * WRONG_FRAME / SR
    focus_s = focus_frames * WRONG_FRAME / SR

    # lock time and false switches per target segment
    locks, switches, locked_s = [], 0, 0.0
    for t0, t1, _who in target_segments(r.scene):
        seg = [x for x in scored if t0 <= x["t"] and x["t"] + W / SR <= t1 + H / SR]
        first = next((x for x in seg if x["correct"]), None)
        if first is None:
            locks.append(None)
            continue
        lock_t = first["t"] + W / SR
        locks.append(round(lock_t - t0, 2))
        after = [x for x in seg if x["t"] >= first["t"]]
        switches += sum(1 for p, q in itertools.pairwise(after) if p["correct"] and not q["correct"])
        locked_s += max(0.0, t1 - lock_t)
    reacq = []
    for te in return_events(r.scene):
        hit = next((x for x in scored if x["t"] >= te and x["correct"]), None)
        reacq.append(round(hit["t"] + W / SR - te, 2) if hit else None)

    # enrollment purity over the chunks of the clips that were finished
    who_side = {r.side_of("A"): 0, r.side_of("B"): 1}
    pure = total = 0
    for tid, spans in run.clips:
        who = who_side.get(sides.get(tid, ""))
        for pos, ln in spans:
            total += 1
            if who is not None and _energy(stems[who][pos : pos + ln]) > _energy(stems[1 - who][pos : pos + ln]):
                pure += 1
    purity = pure / total if total else None
    last = run.clips[0][1][-1] if run.clips else None  # last chunk of the first clip
    first_ready = round((last[0] + last[1]) / SR, 2) if last else None
    return {
        "scene": r.scene.name, "variant": r.variant,
        "acc": acc, "n_scored": len(scored), "n_correct": sum(bool(x["correct"]) for x in scored),
        "n_muted": sum(x["muted"] for x in scored), "n_other": sum(x["other_played"] for x in rows),
        "windows_ok": sum(x["status"] == "ok" for x in rows), "windows": len(rows),
        "lock_s": locks, "switches": switches, "locked_s": round(locked_s, 2),
        "sw_per_min": switches / (locked_s / 60) if locked_s > 0 else 0.0,
        "wrong_s": round(wrong_s, 2), "focus_s": round(focus_s, 2),
        "wrong_per_10s": wrong_s / focus_s * 10 if focus_s > 0 else 0.0,
        "unfocused_s": round((n - int(covered.sum())) / SR, 2),
        "reacq_s": reacq, "purity": purity, "enroll_chunks": total,
        "enroll_ready_s": first_ready, "clips": len(run.clips),
        "rows": rows,
    }


def summarize(per: list[dict]) -> dict:
    def vals(key):
        return [p[key] for p in per if p[key] is not None]

    def flat(key):
        return [v for p in per for v in p[key]]

    locks, reacq = flat("lock_s"), flat("reacq_s")
    n_sc = sum(p["n_scored"] for p in per)
    focus = sum(p["focus_s"] for p in per)
    locked = sum(p["locked_s"] for p in per)
    out = {
        "scenes": len(per),
        "acc_pooled": sum(p["n_correct"] for p in per) / n_sc if n_sc else None,
        "muted_pooled": sum(p["n_muted"] for p in per) / n_sc if n_sc else None,
        "other_windows": sum(p["n_other"] for p in per),
        "windows_ok": sum(p["windows_ok"] for p in per),
        "targets_locked": sum(x is not None for x in locks), "targets": len(locks),
        "acc_mean": float(np.mean(vals("acc"))) if vals("acc") else None,
        "acc_worst": min(vals("acc")) if vals("acc") else None,
        "wrong_s_total": round(sum(p["wrong_s"] for p in per), 2),
        "wrong_per_10s_pooled": sum(p["wrong_s"] for p in per) / focus * 10 if focus else 0.0,
        "wrong_per_10s_mean": float(np.mean([p["wrong_per_10s"] for p in per])),
        "wrong_per_10s_worst": max(p["wrong_per_10s"] for p in per),
        "sw_per_min_pooled": sum(p["switches"] for p in per) / (locked / 60) if locked else 0.0,
        "sw_per_min_worst": max(p["sw_per_min"] for p in per),
        "lock_s_mean": float(np.mean([x for x in locks if x is not None])) if any(x is not None for x in locks) else None,
        "lock_s_worst": max((x for x in locks if x is not None), default=None),
        "lock_never": sum(x is None for x in locks),
        "reacq_s_mean": float(np.mean([x for x in reacq if x is not None])) if any(x is not None for x in reacq) else None,
        "reacq_s_worst": max((x for x in reacq if x is not None), default=None),
        "reacq_never": sum(x is None for x in reacq),
        "purity_mean": float(np.mean(vals("purity"))) if vals("purity") else None,
        "purity_worst": min(vals("purity")) if vals("purity") else None,
        "unfocused_s_total": round(sum(p["unfocused_s"] for p in per), 2),
    }
    out["go"] = bool(out["acc_pooled"] is not None and out["acc_pooled"] >= GO_ACC
                     and out["wrong_per_10s_pooled"] < GO_WRONG_PER_10S)
    return out


# ------------------------------------------------------------------------------------- report
def _f(v, fmt="{:.2f}") -> str:
    if v is None:
        return "never"
    if isinstance(v, list):
        return ",".join(_f(x, fmt) for x in v) or "-"
    return fmt.format(v)


def table(per: list[dict], summ: dict, cfg: str) -> str:
    hdr = (f"{'config':<10}{'scene':<20}{'var':<4}{'acc':>6}{'n':>4}{'lock s':>10}{'sw/min':>7}"
           f"{'wrong s':>8}{'wrong/10s':>10}{'unfoc s':>8}{'reacq s':>9}{'purity':>7}{'ready s':>8}")
    lines = [hdr, "-" * len(hdr)]
    for p in per:
        lines.append(
            f"{cfg:<10}{p['scene']:<20}{p['variant']:<4}{_f(p['acc']):>6}{p['n_scored']:>4}{_f(p['lock_s']):>10}"
            f"{p['sw_per_min']:>7.2f}{p['wrong_s']:>8.2f}{p['wrong_per_10s']:>10.2f}{p['unfocused_s']:>8.2f}"
            f"{_f(p['reacq_s']):>9}{_f(p['purity']):>7}{_f(p['enroll_ready_s']):>8}")
    s = summ
    lines.append("-" * len(hdr))
    lines.append(
        f"{cfg:<10}{'MEAN':<24}{_f(s['acc_mean']):>6}{'':>4}{_f(s['lock_s_mean']):>10}{s['sw_per_min_pooled']:>7.2f}"
        f"{s['wrong_s_total']:>8.2f}{s['wrong_per_10s_mean']:>10.2f}{s['unfocused_s_total']:>8.2f}"
        f"{_f(s['reacq_s_mean']):>9}{_f(s['purity_mean']):>7}")
    lines.append(
        f"{cfg:<10}{'WORST':<24}{_f(s['acc_worst']):>6}{'':>4}{_f(s['lock_s_worst']):>10}{s['sw_per_min_worst']:>7.2f}"
        f"{'':>8}{s['wrong_per_10s_worst']:>10.2f}{'':>8}{_f(s['reacq_s_worst']):>9}{_f(s['purity_worst']):>7}")
    lines.append(
        f"{cfg:<10}POOLED acc {_f(s['acc_pooled'], '{:.3f}')}  wrong/10s {s['wrong_per_10s_pooled']:.2f}  "
        f"never locked {s['lock_never']}  never re-acquired {s['reacq_never']}  -> {'GO' if s['go'] else 'NO-GO'}")
    lines.append(
        f"{cfg:<10}misses: target muted in {_f(s['muted_pooled'], '{:.3f}')} of scored windows; other person played "
        f"in {s['other_windows']} of {s['windows_ok']} extracted windows; {s['targets_locked']}/{s['targets']} targets locked")
    return "\n".join(lines)


def _results_dir() -> Path:
    env = os.environ.get("HEARING_RESULTS_DIR")
    return Path(env) if env else ROOT / "eval" / "results"


def out_path(out: Path | None, pair: str = "mf") -> Path:
    """``--out``: a ``.json`` file as given, or a directory (existing, or a path without ``.json``)
    that gets the default ``association_<stamp>.json`` name; none -> the results dir."""
    name = f"association_{'' if pair == 'mf' else pair + '_'}{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json"
    if out is None:
        return _results_dir() / name
    if out.is_dir() or out.suffix != ".json":
        out.mkdir(parents=True, exist_ok=True)
        return out / name
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help=f"scenes {', '.join(FAST_SCENES)}, variant ab")
    ap.add_argument("--scenes", default="", help="comma-separated scene names (default: all)")
    ap.add_argument("--variants", default="", help="ab, ba or ab,ba (default: ab,ba; --fast: ab)")
    ap.add_argument("--configs", default="baseline,final", help=f"comma-separated, from {', '.join(CONFIGS)}")
    ap.add_argument("--pair", choices=tuple(PAIRS), default="mf",
                    help="speaker pair: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in PAIRS.items()))
    ap.add_argument("--fetch", action="store_true", help="download missing GRID clips of the pair (network)")
    ap.add_argument("--snr", type=float, default=5.0, help="speech level / noise, dB")
    ap.add_argument("--noise", choices=("babble", "pink"), default="babble")
    ap.add_argument("--asd", default="light_asd")
    ap.add_argument("--faces", default="mediapipe")
    ap.add_argument("--candidate", choices=("enrolled", "av_mossformer2"), default="enrolled",
                    help="enrolled: the Enrolled tier per --configs; av_mossformer2: the AV tier on the "
                         "vision pass's lip crops (--configs ignored, window 2 s / hop 1 s by default)")
    ap.add_argument("--window", type=float, default=None, help="seconds (default 1, AV 2)")
    ap.add_argument("--hop", type=float, default=None, help="seconds (default 0.5, AV 1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid", type=Path, default=None, help="GRID source cache (default: auto)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rows", action="store_true", help="print per-window rows")
    ap.add_argument("--define", action="append", default=[], metavar="NAME=JSON",
                    help='extra config, e.g. x={"candidate":{"floor":0.2},"enroller":{"other_hold_s":0.2}}')
    ap.add_argument("--vision-cache", type=Path, default=None,
                    help="directory for per-scene vision results (tracks, p_speaking, lock); reused when present")
    args = ap.parse_args(argv)
    if args.out is not None and args.out.exists() and not args.out.is_dir() and args.out.suffix != ".json":
        ap.error(f"--out {args.out}: not a directory and not a .json file")
    av = args.candidate != "enrolled"
    if args.window is None:
        args.window = 2.0 if av else 1.0
    if args.hop is None:
        args.hop = 1.0 if av else 0.5

    from hearing import device, models
    from hearing.tse.blind import BlindSeparator
    from hearing.tse.enrolled import MODEL as ECAPA

    scenes = load_scenes()
    wanted = [s for s in args.scenes.split(",") if s] or (list(FAST_SCENES) if args.fast else [])
    if wanted:
        unknown = set(wanted) - {s.name for s in scenes}
        if unknown:
            ap.error(f"unknown scenes {sorted(unknown)}")
        scenes = [s for s in scenes if s.name in wanted]
    variants = [v for v in args.variants.split(",") if v] or (["ab"] if args.fast else ["ab", "ba"])
    for d in args.define:
        name, _, spec = d.partition("=")
        spec = json.loads(spec)
        CONFIGS[name] = Config(name, spec.get("candidate", {}), spec.get("enroller", {}))
    configs = [args.candidate] if av else [c for c in args.configs.split(",") if c]
    for c in configs if not av else ():
        if c not in CONFIGS:
            ap.error(f"unknown config {c!r}; known: {', '.join(CONFIGS)}")

    wall = time.perf_counter()
    out = out_path(args.out, args.pair)
    talkers = {}
    for spk in sorted({s for v in variants for s in speakers(v, args.pair)}):
        if args.fetch:
            fetch_talker(spk, args.grid)
        talkers[spk] = load_talker(spk, args.grid)
    separator = CachedSeparator(BlindSeparator()) if not av else None
    embedder = CachedEmbedder(models.load(ECAPA)) if not av else None
    results: dict[str, list[dict]] = {c: [] for c in configs}
    vision_info = {}
    for si, scene in enumerate(scenes):
        for variant in variants:
            r = render(scene, variant, talkers, args.snr, args.noise, args.seed + si, args.pair)
            spec = [dataclasses.asdict(scene), variant, args.snr, args.noise, args.seed + si, args.asd, args.faces]
            if args.pair != "mf":  # mf keys stay as T-061 wrote them, so its caches stay valid
                spec.append(PAIRS[args.pair])
            if av:
                spec.append("lips")
            key = hashlib.sha1(json.dumps(spec).encode()).hexdigest()[:16]
            tag = "" if args.pair == "mf" else f"{args.pair}_"
            cached = args.vision_cache / f"{tag}{scene.name}_{variant}_{key}.pkl" if args.vision_cache else None
            if cached is not None and cached.exists():
                states, info = pickle.loads(cached.read_bytes())
            else:
                states, info = run_vision(r, args.faces, args.asd, lips=av)
                if cached is not None:
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(pickle.dumps((states, info)))
            vision_info = info
            gate = asd_stats(r, states)
            print(f"[asd] {scene.name}/{variant}: " + "  ".join(
                f"{k} {v['s']:.1f}s p_t {v['p_target']:.2f} p_o {v['p_other']:.2f} gate {v['gate_s']:.1f}s"
                for k, v in sorted(gate.items())), file=sys.stderr, flush=True)
            sides: dict[int, str] = {}
            for st in states:
                sides.update(st.sides)
            for c in configs:
                run = (run_av(r, states, c, args.window, args.hop) if av else
                       run_tse(r, states, CONFIGS[c], separator, embedder, args.window, args.hop))
                p = score(r, run, sides)
                p["tse_s"] = round(run.compute_s, 2)
                p["vision_s"] = info["vision_s"]
                p["asd"] = gate
                results[c].append(p)
                print(f"[{c}] {scene.name}/{variant}: acc {_f(p['acc'])} wrong {p['wrong_s']:.2f}s "
                      f"purity {_f(p['purity'])} lock {_f(p['lock_s'])} (vision {info['vision_s']}s, "
                      f"tse {p['tse_s']}s)", file=sys.stderr, flush=True)
                if args.rows:
                    for x in p["rows"]:
                        print("   ", x, file=sys.stderr)
            write_results(out, args, vision_info, wall, results, partial=True)
    summaries = {c: summarize(results[c]) for c in configs}
    print()
    for c in configs:
        print(table(results[c], summaries[c], c))
        print()
    print(f"device {device.pick()}, asd {vision_info.get('asd')}, faces {vision_info.get('faces')}, "
          f"snr {args.snr:g} dB {args.noise}, window {args.window:g}s hop {args.hop:g}s, "
          f"separations {separator.calls if separator else 'n/a'}, wall {time.perf_counter() - wall:.0f}s")
    write_results(out, args, vision_info, wall, results)
    print(f"wrote {out}")
    return 0


def write_results(out: Path, args, vision_info: dict, wall: float, results: dict[str, list[dict]],
                  partial: bool = False) -> None:
    """The result JSON; rewritten after every scene with ``"partial": true`` so a killed run keeps its numbers."""
    from hearing import device

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": device.describe(), "args": {k: str(v) for k, v in vars(args).items()},
        "pair": {"name": args.pair, "speakers": PAIRS[args.pair]}, "partial": partial,
        "vision": vision_info, "wall_s": round(time.perf_counter() - wall, 1),
        "summary": {c: summarize(per) for c, per in results.items() if per},
        "per_scene": {c: [{k: v for k, v in p.items() if k != "rows"} for p in per] for c, per in results.items()},
        "rows": {c: {f"{p['scene']}/{p['variant']}": p["rows"] for p in per} for c, per in results.items()},
    }
    tmp = out.with_suffix(".part")
    tmp.write_text(json.dumps(payload, indent=1, default=float))
    tmp.replace(out)


if __name__ == "__main__":
    # Re-enter through the module so the vision-cache pickles name eval.association_eval.FrameState, not
    # __main__.FrameState (which no other process can unpickle).
    from eval.association_eval import main as _main

    sys.exit(_main())
