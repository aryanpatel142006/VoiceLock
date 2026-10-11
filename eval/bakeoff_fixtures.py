"""Audio-visual fixtures for the Focus HQ bake-off.

A fixture directory follows the hearing-stack layout::

    <set>/<id>/mix.wav target.wav interferer.wav noise.wav manifest.json [video.mp4] [lips.npy]

``manifest.json`` keys used here (all optional except what the metric needs):

* ``transcript``: target transcript, a string or ``{"target": ..., "interferer": ...}``.
* ``lips_fps``: frame rate of ``lips.npy`` / ``video.mp4`` after resampling (default 25).
* ``target_box``: target face box ``[x, y, w, h]`` in the video, normalised to [0, 1] like
  ``hearing.types.FaceTrack.box`` (pixels if any value is > 1).
* ``mouth_box``: explicit mouth box in the same format; overrides the one derived from ``target_box``.
* ``enroll_wav``: path (relative to the fixture) of target-only enrollment audio.

Inputs derived for every candidate:

* ``lips``: ``lips.npy`` (``T x H x W`` uint8) if present, else a **fixed** mouth crop taken from
  ``video.mp4`` at the manifest's box. A fixed crop is only valid for frontal, static talkers
  (GRID, our synthetic AV mixtures); real scenes need the face tracker's per-frame landmarks.
  Else None.
* ``p_speaking``: an **oracle proxy** from the clean target's per-video-frame energy, until active
  speaker detection (Light-ASD) lands.
* ``enroll``: ``enroll_wav`` if present, else the first 3 s of the clean target. The fallback overlaps
  the evaluated clip, so it is optimistic for enrolled candidates.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np

from hearing.audio import io
from hearing.testing import make_test_signal
from hearing.tse.base import LIPS_FPS, LIPS_SIZE, SR

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
GENERATED = Path(os.environ["HEARING_FIXTURES_DIR"]) if os.environ.get("HEARING_FIXTURES_DIR") else ROOT / "fixtures" / "generated"
TINY_DIR = GENERATED / "bakeoff_tiny"
ENROLL_S = 3.0
_SYNTH_VERSION = 1

_WORDS = [
    *(
        "bin",
        "lay",
        "place",
        "set",
        "blue",
        "green",
        "red",
        "white",
        "at",
        "by",
        "in",
        "with",
    ),
    *(
        "f",
        "l",
        "s",
        "x",
        "one",
        "two",
        "three",
        "four",
        "now",
        "please",
        "again",
        "soon",
    ),
]


@dataclass
class AVFixture:
    id: str
    path: Path
    sr: int
    mix: np.ndarray
    target: np.ndarray
    transcript: str | None
    lips: np.ndarray | None
    lips_fps: float
    lips_source: str | None  # "lips.npy" | "video.mp4@box" | None
    p_speaking: np.ndarray
    enroll: np.ndarray
    enroll_source: str  # "enroll_wav" | "target[:3s]"
    manifest: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.mix.shape[0] / self.sr


# --- derived inputs ----------------------------------------------------------------------


def oracle_p_speaking(
    target: np.ndarray, sr: int = SR, fps: float = LIPS_FPS
) -> np.ndarray:
    """Per-video-frame p(target speaking) from clean-target energy (an oracle proxy for ASD).

    Frame RMS in dB through a soft threshold 30 dB below the loudest frame (floor -70 dBFS).
    """
    n_frames = max(1, math.ceil(target.shape[0] / sr * fps))
    edges = np.round(np.arange(n_frames + 1) * sr / fps).astype(int)
    x = np.asarray(target, dtype=np.float64)
    rms = np.array(
        [
            np.sqrt(np.mean(x[a:b] ** 2)) if b > a and a < x.shape[0] else 0.0
            for a, b in pairwise(edges)
        ]
    )
    db = 20 * np.log10(rms + 1e-10)
    thr = max(float(db.max()) - 30.0, -70.0)
    return (1.0 / (1.0 + np.exp(-(db - thr) / 3.0))).astype(np.float32)


def _box_px(
    box: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    x, y, w, h = (float(v) for v in box)
    if max(x, y, w, h) <= 1.0:
        return x * width, y * height, w * width, h * height
    return x, y, w, h


def mouth_box_from_face(face: list[float]) -> list[float]:
    """Square mouth box (same units as ``face``): centred horizontally, 78 % down the face, 50 % wide."""
    x, y, w, h = (float(v) for v in face)
    side = 0.5 * w
    cx, cy = x + w / 2, y + 0.78 * h
    return [cx - side / 2, cy - side / 2, side, side]


def lips_from_video(
    video: Path, mouth_box: list[float], fps: float = LIPS_FPS, size: int = LIPS_SIZE
) -> np.ndarray | None:
    """Fixed mouth crop from every frame of ``video``, resampled to ``fps``; None without OpenCV."""
    try:
        import cv2
    except ImportError:
        log.warning("opencv missing; no lip crops from %s", video)
        return None
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None
    hgt, wid = frames[0].shape[:2]
    x, y, w, h = _box_px(mouth_box, wid, hgt)
    x0, y0 = max(0, round(x)), max(0, round(y))
    x1, y1 = min(wid, round(x + w)), min(hgt, round(y + h))
    n_out = max(1, round(len(frames) / src_fps * fps))
    idx = np.minimum(
        np.round(np.arange(n_out) * src_fps / fps).astype(int), len(frames) - 1
    )
    out = np.empty((n_out, size, size), np.uint8)
    for i, j in enumerate(idx):
        gray = cv2.cvtColor(frames[j][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        out[i] = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return out


# --- loading -----------------------------------------------------------------------------


def _transcript(manifest: dict) -> str | None:
    t = manifest.get("transcript", manifest.get("transcripts"))
    if isinstance(t, dict):
        t = t.get("target")
    return t if isinstance(t, str) else None


def load_fixture(path: str | Path) -> AVFixture:
    d = Path(path)
    manifest = json.loads((d / "manifest.json").read_text())
    mix, sr = io.load(d / "mix.wav", SR)
    target, _ = io.load(d / "target.wav", SR)
    n = min(mix.shape[0], target.shape[0])
    mix, target = mix[:n], target[:n]
    fps = float(manifest.get("lips_fps", LIPS_FPS))

    lips, lips_source = None, None
    if (d / "lips.npy").exists():
        lips, lips_source = np.load(d / "lips.npy"), "lips.npy"
        if lips.ndim != 3 or lips.dtype != np.uint8:
            raise ValueError(
                f"{d}/lips.npy must be T x H x W uint8, got {lips.shape} {lips.dtype}"
            )
    elif (d / "video.mp4").exists() and (
        "mouth_box" in manifest or "target_box" in manifest
    ):
        box = manifest.get("mouth_box") or mouth_box_from_face(manifest["target_box"])
        lips = lips_from_video(d / "video.mp4", box, fps)
        lips_source = "video.mp4@box" if lips is not None else None

    if manifest.get("enroll_wav"):
        enroll, _ = io.load(d / manifest["enroll_wav"], SR)
        enroll_source = "enroll_wav"
    else:
        enroll, enroll_source = target[: int(ENROLL_S * sr)].copy(), "target[:3s]"

    return AVFixture(
        id=str(manifest.get("id", d.name)),
        path=d,
        sr=sr,
        mix=mix,
        target=target,
        transcript=_transcript(manifest),
        lips=lips,
        lips_fps=fps,
        lips_source=lips_source,
        p_speaking=oracle_p_speaking(target, sr, fps),
        enroll=enroll,
        enroll_source=enroll_source,
        manifest=manifest,
    )


def load_set(path: str | Path) -> list[AVFixture]:
    """One fixture directory, or a set directory whose subdirectories are fixtures."""
    d = Path(path)
    if (d / "manifest.json").exists():
        return [load_fixture(d)]
    dirs = (
        sorted(p for p in d.iterdir() if (p / "manifest.json").exists())
        if d.is_dir()
        else []
    )
    if not dirs:
        raise FileNotFoundError(f"no fixtures (dirs with manifest.json) under {d}")
    return [load_fixture(p) for p in dirs]


def resolve_set(name: str) -> Path:
    """``tiny`` → the synthetic AV set (generated on demand); anything else is a path."""
    if name == "tiny":
        return synthetic_av_set()
    return Path(name)


# --- synthetic set -----------------------------------------------------------------------


def _scale_to(x: np.ndarray, ref: np.ndarray, db: float) -> np.ndarray:
    """Scale ``x`` so that 20·log10(rms(ref)/rms(x)) == db."""
    rx, rr = np.sqrt(np.mean(x**2)), np.sqrt(np.mean(ref**2))
    return x * (rr / (rx + 1e-12)) * 10 ** (-db / 20)


def synthetic_lips(
    target: np.ndarray, sr: int = SR, fps: float = LIPS_FPS, seed: int = 0
) -> np.ndarray:
    """``T x 88 x 88`` uint8 frames: a dark mouth ellipse whose opening follows target energy."""
    rng = np.random.default_rng(seed)
    p = oracle_p_speaking(target, sr, fps)
    n = p.shape[0]
    edges = np.round(np.arange(n + 1) * sr / fps).astype(int)
    x = target.astype(np.float64)
    rms = np.array(
        [np.sqrt(np.mean(x[a:b] ** 2)) if b > a else 0.0 for a, b in pairwise(edges)]
    )
    env = rms / (rms.max() + 1e-12)
    yy, xx = np.mgrid[0:LIPS_SIZE, 0:LIPS_SIZE].astype(np.float32)
    c = LIPS_SIZE / 2
    out = np.empty((n, LIPS_SIZE, LIPS_SIZE), np.uint8)
    for i in range(n):
        b = 1.5 + 18.0 * env[i]  # vertical half-opening in px
        mouth = ((xx - c) / 22.0) ** 2 + ((yy - c) / b) ** 2 <= 1.0
        frame = 150.0 + rng.normal(0.0, 4.0, (LIPS_SIZE, LIPS_SIZE))
        frame[mouth] = 40.0
        out[i] = np.clip(frame, 0, 255).astype(np.uint8)
    return out


def synthetic_av_set(
    n: int = 4, seed: int = 0, out: str | Path | None = None, duration_s: float = 4.0
) -> Path:
    """Write ``n`` two-talker + noise AV fixtures with synthetic lip arrays; return the set dir.

    Target: speech-like bursts (~120 Hz). Interferer: another speech-like signal pitch-shifted up
    by 1.4x, 0 dB SIR. Noise: white, 10 dB SNR re the target. ``lips.npy``: mouth opening driven
    by target energy. Cached: regenerated only when the parameters change.
    """
    root = Path(out) if out is not None else TINY_DIR
    params = {"version": _SYNTH_VERSION, "n": n, "seed": seed, "duration_s": duration_s}
    stamp = root / "params.json"
    if stamp.exists() and json.loads(stamp.read_text()) == params:
        return root
    root.mkdir(parents=True, exist_ok=True)
    for old in root.glob("syn[0-9]*"):
        shutil.rmtree(old)
    for k in range(n):
        s = seed * 1000 + k
        rng = np.random.default_rng(s)
        target = make_test_signal(duration_s, SR, "speechlike", seed=s)
        shift = 1.4
        inter = make_test_signal(duration_s * shift, SR, "speechlike", seed=s + 500)
        inter = io.resample(inter, round(SR * shift), SR)[: target.shape[0]]
        inter = np.pad(inter, (0, target.shape[0] - inter.shape[0]))
        inter = _scale_to(inter, target, 0.0)
        noise = _scale_to(
            make_test_signal(duration_s, SR, "noise", seed=s + 900), target, 10.0
        )
        mix = target + inter + noise
        g = 0.9 / max(float(np.abs(mix).max()), 1e-6)
        target, inter, noise, mix = (a * g for a in (target, inter, noise, mix))
        d = root / f"syn{k:02d}"
        d.mkdir(exist_ok=True)
        for name, x in (
            ("mix", mix),
            ("target", target),
            ("interferer", inter),
            ("noise", noise),
        ):
            io.save(d / f"{name}.wav", x.astype(np.float32), SR)
        np.save(
            d / "lips.npy",
            synthetic_lips(target.astype(np.float32), SR, LIPS_FPS, seed=s),
        )
        manifest = {
            "id": d.name,
            "sr": SR,
            "duration_s": duration_s,
            "source": "synthetic (hearing.testing.make_test_signal); not real speech",
            "transcript": {
                "target": " ".join(rng.choice(_WORDS, 6)),
                "interferer": " ".join(rng.choice(_WORDS, 6)),
            },
            "sir_db": 0.0,
            "snr_db": 10.0,
            "lips_fps": LIPS_FPS,
            "target_box": [0.35, 0.2, 0.3, 0.5],
        }
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    stamp.write_text(json.dumps(params))
    return root
