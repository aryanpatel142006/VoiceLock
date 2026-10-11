"""Demo noise kit: play café babble and/or a competing talker through a chosen speaker.

    .venv/bin/python scripts/demo_noise.py --list
    .venv/bin/python scripts/demo_noise.py --device "JBL" --calibrate
    .venv/bin/python scripts/demo_noise.py --device "JBL" [--engine-output "MacBook Pro Speakers"]

Keys: 1 babble, 2 talker, 3 both, + / - level (2 dB), space pause, q quit.
Refuses to play on the engine's headphones output (``--engine-output``, else the system default
output, which is what ``hearing live`` uses). Sources come from WAVs under HEARING_FIXTURES_DIR
(babble = several clips summed, talker = one clip); with none cached the talker is Kokoro TTS and
the babble is a synthetic murmur. Never imported by the engine; no engine code is touched.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np

SR = 16000
CALIBRATE_TEXT = (
    "Calibration: pink noise plays at a fixed digital level. Hold a phone SPL app at the listener's "
    "head and use +/- until it reads about 65 dB(A). Quit with q; the level is printed."
)
PARAGRAPH = ("I was telling her about the trip last week. We drove all night, stopped for coffee "
             "near the river, and then the road closed, so we had to turn around and go home.")


class DeviceError(SystemExit):
    pass


def resolve_device(spec, devices: list[dict]) -> int:
    """Index of the output device matching ``spec`` (an index or a name substring)."""
    outs = [(i, d) for i, d in enumerate(devices) if d.get("max_output_channels", 0) > 0]
    if str(spec).isdigit():
        for i, _ in outs:
            if i == int(spec):
                return i
        raise DeviceError(f"no output device with index {spec}")
    hits = [i for i, d in outs if str(spec).lower() in d["name"].lower()]
    if not hits:
        raise DeviceError(f"no output device matching {spec!r}; use --list")
    if len(hits) > 1:
        names = ", ".join(f"{i}:{devices[i]['name']}" for i in hits)
        raise DeviceError(f"{spec!r} is ambiguous ({names}); give the index")
    return hits[0]


def choose_devices(device, engine_output, devices: list[dict], default_out: int | None) -> int:
    """Validate the noise device and refuse the engine's output device."""
    if device is None:
        raise DeviceError("--device is required (see --list)")
    noise = resolve_device(device, devices)
    engine = resolve_device(engine_output, devices) if engine_output is not None else default_out
    if engine is not None and noise == engine:
        raise DeviceError(f"refusing: {devices[noise]['name']!r} is the engine's headphones output; "
                          "pick the Bluetooth speaker (or pass --engine-output)")
    return noise


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spec = np.fft.rfft(rng.standard_normal(n))
    spec[1:] /= np.sqrt(np.arange(1, len(spec)))
    x = np.fft.irfft(spec, n).astype(np.float32)
    return x / (np.sqrt(np.mean(x**2)) + 1e-9)


def _rms_norm(x: np.ndarray, rms: float = 0.1) -> np.ndarray:
    return (x * (rms / (np.sqrt(np.mean(x**2)) + 1e-9))).astype(np.float32)


def find_clips(root: Path | None) -> list[Path]:
    if root is None or not root.is_dir():
        return []
    return sorted(root.rglob("*.wav"))[:200]


def _load(path: Path) -> np.ndarray:
    from hearing.audio.io import load
    return load(path)[0]


def build_babble(clips: list[Path], seconds: float = 30.0, seed: int = 0) -> np.ndarray:
    n = int(seconds * SR)
    rng = np.random.default_rng(seed)
    if len(clips) >= 3:
        mix = np.zeros(n, np.float32)
        for p in rng.permutation(len(clips))[:6]:
            x = _load(clips[p])
            if len(x) < 100:
                continue
            x = np.tile(x, n // len(x) + 1)[:n]
            mix += np.roll(_rms_norm(x), int(rng.integers(n)))
        return _rms_norm(mix)
    # synthetic murmur: pink noise with syllable-rate (2-6 Hz) modulation, several "voices"
    t = np.arange(n) / SR
    mix = np.zeros(n, np.float32)
    for _ in range(6):
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * t + rng.uniform(0, 6.3))
        mix += pink_noise(n, rng) * mod.astype(np.float32)
    return _rms_norm(mix)


def build_talker(clips: list[Path], seconds: float = 30.0) -> np.ndarray:
    n = int(seconds * SR)
    x = None
    if clips:
        x = _load(clips[0])
    else:
        try:
            from hearing.synth.tts import KokoroTTS
            x = KokoroTTS().synthesize(PARAGRAPH, None)
        except Exception as e:  # noqa: BLE001  weights missing: say so, fall back to a murmur
            print(f"no clips and Kokoro unavailable ({e}); talker is synthetic noise", file=sys.stderr)
    if x is None or len(x) < 100:
        return build_babble([], seconds, seed=1)
    return _rms_norm(np.tile(x, n // len(x) + 1)[:n])


class Player:
    """Loops the selected source at a level (dB re full scale RMS); keys are handled by the caller."""

    def __init__(self, sources: dict[str, np.ndarray]):
        self.sources, self.mode, self.gain_db, self.paused = sources, "3", -20.0, False
        self.pos = 0
        self.lock = threading.Lock()

    def callback(self, outdata, frames, time_info, status):
        with self.lock:
            if self.paused:
                outdata[:] = 0
                return
            src = self.sources[self.mode]
            idx = (self.pos + np.arange(frames)) % len(src)
            self.pos = (self.pos + frames) % len(src)
            outdata[:, 0] = np.clip(src[idx] * 10 ** (self.gain_db / 20), -1, 1)

    def key(self, ch: str) -> bool:
        """Apply a key; False means quit."""
        with self.lock:
            if ch in self.sources and ch in "123":
                self.mode = ch
            elif ch == "+" or ch == "=":
                self.gain_db = min(self.gain_db + 2, -3)
            elif ch in "-_":
                self.gain_db -= 2
            elif ch == " ":
                self.paused = not self.paused
            elif ch in "qQ":
                return False
        print(f"\rsource={self.mode} level={self.gain_db:+.0f} dBFS {'PAUSED' if self.paused else '      '}",
              end="", flush=True)
        return True


def read_keys():
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            yield sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list output devices and exit")
    ap.add_argument("--device", help="noise speaker: output device index or name substring")
    ap.add_argument("--engine-output", default=os.environ.get("HEARING_OUTPUT_DEVICE"),
                    help="the engine's headphones output (default: the system default output)")
    ap.add_argument("--calibrate", action="store_true", help="play pink noise to set the level (~65 dB)")
    ap.add_argument("--source", choices=["1", "2", "3"], default="3", help="start source")
    ap.add_argument("--level", type=float, default=_preset_level(-20.0),
                    help="start level, dBFS RMS (HEARING_DEMO_PRESET=team: demo_noise_level_db of config/demo-team.yaml)")
    ap.add_argument("--fixtures", default=os.environ.get("HEARING_FIXTURES_DIR"))
    return ap.parse_args(argv)


def _preset_level(default: float) -> float:
    """The team demo preset's pinned noise level (T-130), else ``default``."""
    if os.environ.get("HEARING_DEMO_PRESET", "").strip().lower() != "team":
        return default
    import yaml

    preset = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "demo-team.yaml").read_text()) or {}
    return float(preset.get("demo_noise_level_db", default))


def main(argv=None) -> int:
    args = parse_args(argv)
    import sounddevice as sd
    devices = [dict(d) for d in sd.query_devices()]
    default_out = sd.default.device[1] if sd.default.device[1] is not None and sd.default.device[1] >= 0 else None
    if args.list:
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                mark = " (default output)" if i == default_out else ""
                print(f"{i:3d}  {d['name']}  [{d['max_output_channels']} ch]{mark}")
        return 0
    dev = choose_devices(args.device, args.engine_output, devices, default_out)
    if args.calibrate:
        print(CALIBRATE_TEXT)
        pink = pink_noise(30 * SR, np.random.default_rng(0))
        sources = {m: _rms_norm(pink) for m in "123"}
    else:
        clips = find_clips(Path(args.fixtures) if args.fixtures else None)
        sources = {"1": build_babble(clips), "2": build_talker(clips)}
        sources["3"] = _rms_norm(sources["1"] + sources["2"])
    player = Player(sources)
    player.mode, player.gain_db = args.source, args.level
    print(f"playing on {devices[dev]['name']!r}: 1 babble, 2 talker, 3 both, +/- level, space pause, q quit")
    with sd.OutputStream(samplerate=SR, channels=1, device=dev, callback=player.callback, dtype="float32"):
        for ch in read_keys():
            if not player.key(ch):
                break
    print(f"\nfinal level {player.gain_db:+.0f} dBFS (pass --level {player.gain_db:.0f} next time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
