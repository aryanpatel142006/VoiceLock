"""Write a voice profile for the Enrolled tier (T-125): ``$HEARING_PROFILES_DIR/<name>/*.wav``.

    .venv/bin/python scripts/make_profile.py aryan rec1.wav rec2.wav     # 16 kHz mono, silence trimmed
    .venv/bin/python scripts/make_profile.py s1 a.wav b.wav --out /tmp/profiles

Each input (any format soundfile reads) is resampled to
16 kHz mono, peak-normalised to -3 dBFS, and its unvoiced 10 ms hops (``VoicedGate``, the
enroller's -40 dBFS / floor + 9 dB rule) are cut out, keeping 100 ms around speech; the voiced audio
is written as ``<name>/<stem>.wav``. Aim for 10-20 s of clean speech of one person (3-6
sentences, alone, the demo mic). The session cuts it into 2-3 s pieces and embeds each. The files
are voices: gitignored, never committed or logged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from hearing.audio.io import HOP, PIPELINE_SR, load, save
from hearing.tse.enroll import PROFILES_ENV, VoicedGate, level_db, profiles_dir

PAD_HOPS = 10  # 100 ms kept on each side of voiced audio
PEAK = 10 ** (-3 / 20)


def trim_silence(x: np.ndarray, sr: int = PIPELINE_SR) -> np.ndarray:
    """The voiced hops of ``x`` (with ``PAD_HOPS`` around each run), concatenated."""
    x = np.asarray(x, dtype=np.float32)
    n = len(x) // HOP
    gate = VoicedGate(sr, HOP)
    voiced = np.array([gate.voiced(x[i * HOP : (i + 1) * HOP]) for i in range(n)], dtype=bool)
    # a second pass with the floor learned on the whole file (the first second used the absolute gate)
    voiced |= np.array([level_db(x[i * HOP : (i + 1) * HOP]) >= gate.threshold_db for i in range(n)], dtype=bool)
    keep = voiced.copy()
    for d in range(1, PAD_HOPS + 1):
        keep[d:] |= voiced[:-d]
        keep[:-d] |= voiced[d:]
    if not keep.any():
        return np.zeros(0, dtype=np.float32)
    return np.concatenate([x[i * HOP : (i + 1) * HOP] for i in np.flatnonzero(keep)])


def make_profile(name: str, inputs: list[Path], out: Path) -> list[Path]:
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"bad profile name {name!r}")
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    written = []
    for src in inputs:
        x, _ = load(src, PIPELINE_SR)
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        if peak > 1e-6:
            x = (x * (PEAK / peak)).astype(np.float32)
        y = trim_silence(x)
        if len(y) < PIPELINE_SR // 2:
            print(f"skip {src}: {len(y) / PIPELINE_SR:.1f} s of speech", file=sys.stderr)
            continue
        path = d / f"{Path(src).stem}.wav"
        save(path, y, PIPELINE_SR)
        written.append(path)
        print(f"{path}: {len(y) / PIPELINE_SR:.1f} s of speech (from {len(x) / PIPELINE_SR:.1f} s)")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", help="profile name (letters, digits, _ and -)")
    ap.add_argument("inputs", nargs="+", type=Path, help="recordings of that one voice")
    ap.add_argument("--out", type=Path, default=None, help=f"profiles root (default ${PROFILES_ENV})")
    args = ap.parse_args(argv)
    out = args.out or profiles_dir()
    if out is None:
        ap.error(f"set {PROFILES_ENV} (an existing directory) or pass --out")
    written = make_profile(args.name, args.inputs, out)
    total = sum(len(load(p, PIPELINE_SR)[0]) for p in written) / PIPELINE_SR
    print(f"profile {args.name!r}: {len(written)} file(s), {total:.1f} s of speech in {out / args.name}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
