#!/usr/bin/env python3
"""Record a teammate's solo take (mic + webcam) where ``make_fixtures --demo-team`` looks for it (T-126).

    python scripts/record_take.py --name aryan --kind profile            # 30 s enrollment take
    python scripts/record_take.py --name aryan --kind fixture            # 60 s scene take
    python scripts/record_take.py --venue-noise --seconds 60              # the room, nobody talking
    python scripts/record_take.py --list-devices

Writes ``$HEARING_FIXTURES_DIR/team/<name>/<kind>-NN.wav`` (16 kHz mono PCM16, the lossless mic
take) and ``<kind>-NN.mp4`` (the webcam at 25 fps, <= 640x360, with the same audio), the next free
NN; ``--venue-noise`` writes ``team/venue_noise.wav`` (no video). ``--bucket`` stores the voice
bucket (``male_adult`` ...) in ``team/<name>/meta.json`` so ``demo-team-same`` is built when both
teammates share one. Before recording: a 3 s level check (speak normally; exit 3 when too quiet or
clipping, ``--force`` records anyway), then a countdown, then the take, then the take's speech
level and seconds of speech. Profile and fixture takes are always separate files.

Opens the microphone and the camera: run it by hand in a terminal that has both permissions,
never from a test or a headless agent (tests pass fake devices to :func:`record`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import demo_team
from eval.fixtures import video as fxvideo
from hearing.audio import io as aio

SR = 16000
FPS = 25
KINDS = ("profile", "fixture")
DEFAULT_SECONDS = {"profile": 30.0, "fixture": 60.0, "venue": 60.0}
CHECK_S = 3.0
COUNTDOWN = 3
TOO_QUIET_DB = -40.0  # active speech level below this: move closer / raise the input gain
CLIP_PEAK = 0.99  # a peak at or above this: the take clipped, lower the gain
MIN_SPEECH_S = {"profile": 10.0, "fixture": 20.0}
BUCKETS = ("male_adult", "female_adult", "male_child", "female_child", "neutral")
# Harvard sentences (public domain), read one per line with a short pause between them
SCRIPT = (
    "The birch canoe slid on the smooth planks.", "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.", "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.", "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.", "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.", "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.", "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.", "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.", "A pot of tea helps to pass the evening.",
    "Smoky fires lack flame and heat.", "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.", "The girl at the booth sold fifty bonds.",
)


def fixtures_dir() -> Path:
    env = os.environ.get("HEARING_FIXTURES_DIR")
    if env:
        return Path(env)
    from eval import make_fixtures

    return make_fixtures.DEFAULT_OUT


# --- devices (real ones only here; tests pass fakes) ---------------------------------------


class SoundDeviceMic:
    """Blocking 16 kHz mono capture through sounddevice."""

    def __init__(self, device: str | int | None = None):
        import sounddevice as sd

        self.sd = sd
        self.device = int(device) if isinstance(device, str) and device.isdigit() else device

    def record(self, seconds: float) -> np.ndarray:
        x = self.sd.rec(round(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=self.device)
        self.sd.wait()
        return np.asarray(x[:, 0], np.float32)


class OpenCVCamera:
    """Webcam frames with their capture times, read on a thread while the mic records."""

    def __init__(self, index: int = 0):
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {index} did not open (camera permission? another app using it?)")
        self.frames: list[tuple[float, np.ndarray]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.frames, t0 = [], time.monotonic()
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                ok, bgr = self.cap.read()
                if ok:
                    self.frames.append((time.monotonic() - t0, self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)))

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, np.ndarray]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.cap.release()
        return self.frames


# --- pure helpers ----------------------------------------------------------------------------


def to_fps(stamped: list[tuple[float, np.ndarray]], seconds: float, fps: int = FPS) -> np.ndarray:
    """Timestamped frames -> exactly ``seconds * fps`` frames (the latest frame at each tick), fitted to 640x360."""
    if not stamped:
        raise RuntimeError("the camera delivered no frames")
    times = np.array([t for t, _ in stamped])
    n = round(seconds * fps)
    idx = np.clip(np.searchsorted(times, np.arange(n) / fps, side="right") - 1, 0, len(stamped) - 1)
    frames = np.stack([stamped[i][1] for i in idx])
    return fxvideo.fit_resize(frames)


def levels(x: np.ndarray) -> dict:
    """Peak, speech level (dBFS over the VAD's speech frames) and seconds of speech."""
    db = demo_team.frame_db(np.asarray(x, np.float32))
    runs = demo_team.speech_runs(db)
    speech = np.concatenate([db[a:b] for a, b in runs]) if runs else db
    active = 10 * np.log10(np.mean(10 ** (speech / 10)) + 1e-12) if len(speech) else -120.0
    return {"peak": round(float(np.max(np.abs(x))) if len(x) else 0.0, 3), "speech_dbfs": round(float(active), 1),
            "speech_s": round(sum(b - a for a, b in runs) / FPS, 1)}


def verdict(lv: dict) -> str | None:
    if lv["peak"] >= CLIP_PEAK:
        return f"clipping (peak {lv['peak']}): lower the input gain or move back"
    if lv["speech_dbfs"] < TOO_QUIET_DB or lv["speech_s"] < 0.5:
        return f"too quiet (speech {lv['speech_dbfs']} dBFS < {TOO_QUIET_DB}): move closer or raise the gain"
    return None


def next_path(d: Path, kind: str) -> Path:
    used = {int(p.stem.split("-")[-1]) for p in d.glob(f"{kind}-*") if p.stem.split("-")[-1].isdigit()}
    return d / f"{kind}-{min(set(range(1, len(used) + 2)) - used):02d}"


def record(name: str | None, kind: str, seconds: float, root: Path, mic, camera=None, *,
           check: bool = True, force: bool = False, bucket: str | None = None,
           sleep: Callable[[float], None] = time.sleep, say: Callable[[str], None] = print) -> dict:
    """Level check, countdown, take; writes the files and returns ``{"wav", "mp4", "levels", ...}``.

    ``kind`` is ``profile``, ``fixture`` or ``venue`` (``team/venue_noise.wav``, no name, no camera).
    Raises ``LevelError`` when the level check fails and ``force`` is False."""
    if kind not in (*KINDS, "venue"):
        raise ValueError(f"kind must be one of {KINDS} or 'venue'")
    venue = kind == "venue"
    if not venue and not name:
        raise ValueError("--name is required for profile/fixture takes")
    team = Path(root) / "team"
    if check and not venue:
        say(f"Level check: talk normally for {CHECK_S:g} s, e.g. read '{SCRIPT[0]}'")
        lv = levels(mic.record(CHECK_S))
        problem = verdict(lv)
        say(f"  peak {lv['peak']}, speech {lv['speech_dbfs']} dBFS: {problem or 'OK'}")
        if problem and not force:
            raise LevelError(problem)
    if venue:
        say(f"Venue noise: nobody near the mic talks for {seconds:g} s; let the room be the room.")
    else:
        say(f"{kind} take for {name}, {seconds:g} s. Face the camera, centred, about an arm's length away."
            " Read these, one per line, with a short pause between them:")
        for line in SCRIPT[: max(4, int(seconds / 3))]:
            say(f"    {line}")
    for k in range(COUNTDOWN, 0, -1):
        say(f"  {k}...")
        sleep(1.0)
    say("  recording")
    if camera is not None and not venue:
        camera.start()
    try:
        audio = np.asarray(mic.record(seconds), np.float32)
    finally:
        stamped = camera.stop() if camera is not None and not venue else None
    lv = levels(audio)
    problem = verdict(lv)
    if venue:
        team.mkdir(parents=True, exist_ok=True)
        wav = team / "venue_noise.wav"
        aio.save(wav, audio, SR)
        say(f"wrote {wav} (peak {lv['peak']})")
        return {"wav": wav, "mp4": None, "levels": lv, "problem": None if lv["peak"] < CLIP_PEAK else problem}
    d = team / name
    d.mkdir(parents=True, exist_ok=True)
    base = next_path(d, kind)
    wav, mp4 = base.with_suffix(".wav"), None
    aio.save(wav, audio, SR)
    if stamped is not None:
        frames = to_fps(stamped, len(audio) / SR)
        mp4 = base.with_suffix(".mp4")
        with tempfile.TemporaryDirectory(prefix="record_take_") as tmp:
            fxvideo.mux_audio(fxvideo.write_video(Path(tmp) / "v.mp4", frames, FPS), wav, mp4)
    if bucket:
        meta = d / "meta.json"
        m = json.loads(meta.read_text()) if meta.exists() else {}
        meta.write_text(json.dumps({**m, "bucket": bucket}, indent=2) + "\n")
    short = lv["speech_s"] < MIN_SPEECH_S[kind]
    say(f"wrote {wav}" + (f" and {mp4}" if mp4 else "") + f": peak {lv['peak']}, speech {lv['speech_dbfs']} dBFS, "
        f"{lv['speech_s']} s of speech" + (f" (want >= {MIN_SPEECH_S[kind]:g} s: record another)" if short else ""))
    if problem:
        say(f"  WARNING: {problem}; record this take again")
    return {"wav": wav, "mp4": mp4, "levels": lv, "problem": problem or ("too little speech" if short else None)}


class LevelError(RuntimeError):
    pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", help="teammate (lowercase, one word): the folder under team/")
    ap.add_argument("--kind", choices=KINDS, default="fixture")
    ap.add_argument("--venue-noise", action="store_true", help="record team/venue_noise.wav (no camera, no name)")
    ap.add_argument("--seconds", type=float, default=None, help="take length (profile 30, fixture 60, venue 60)")
    ap.add_argument("--bucket", choices=BUCKETS, default=None, help="store this teammate's voice bucket in meta.json")
    ap.add_argument("--mic", default=None, help="input device index or name (default: the system input)")
    ap.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    ap.add_argument("--no-video", action="store_true", help="audio only (the scene shows a grey card for this take)")
    ap.add_argument("--root", type=Path, default=None, help="fixtures root (default $HEARING_FIXTURES_DIR)")
    ap.add_argument("--no-check", action="store_true", help="skip the level check")
    ap.add_argument("--force", action="store_true", help="record even when the level check fails")
    ap.add_argument("--list-devices", action="store_true", help="print the audio devices and exit")
    args = ap.parse_args(argv)
    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    kind = "venue" if args.venue_noise else args.kind
    if kind != "venue" and not args.name:
        ap.error("--name is required (or --venue-noise)")
    name = args.name.strip().lower() if args.name else None
    if name and not name.replace("-", "").replace("_", "").isalnum():
        ap.error("--name: letters, digits, - and _ only")
    mic = SoundDeviceMic(args.mic)
    camera = None if args.no_video or kind == "venue" else OpenCVCamera(args.camera)
    try:
        out = record(name, kind, args.seconds or DEFAULT_SECONDS[kind], args.root or fixtures_dir(), mic, camera,
                     check=not args.no_check, force=args.force, bucket=args.bucket)
    except LevelError as e:
        print(f"level check failed: {e} (fix it, or --force)", file=sys.stderr)
        return 3
    return 1 if out["problem"] else 0


if __name__ == "__main__":
    sys.exit(main())
