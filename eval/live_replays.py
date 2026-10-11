"""T-392: replay recorded live sessions (real room + mic, no stems) through base vs the fine-tuned separator, for listening.

    .venv/bin/python -m eval.live_replays <session dir> [<session dir> ...] --target aryan --other joseph \
        --out ~/Desktop/selective-hearing-finetune/live-replays

Per session ``<out>/<stamp>/``: ``room.wav`` (the session's ``in.wav``, what the room sounded like at the mic),
``base.wav`` (the shipped MossFormer2 separator) and ``finetuned.wav`` (``HEARING_SEP_CKPT`` = T-374's
fixed-order checkpoint + ``HEARING_PICK=fixed``, T-386's knobs). Both arms run the real ``EnrolledCandidate`` over
1.0 s windows every 0.5 s with the saved profiles (``eval.floor_ab.windows`` / ``render``, standard isolation, slider
100, pre-AGC), each arm in its own process so the separator is built with that arm's env. All three files are
level-matched (``scripts/t386_export_ab.level_match``: active speech at -26 dBFS, peak <= -1 dBFS).
There are no stems, so nothing here is a measurement: listening files only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.contrast_sweep import SR
from eval.floor_ab import render, windows
from hearing.audio import io as aio
from hearing.tse.enroll import load_profile, profile_clip

FT_CKPT = "T-374/fixed_best.pt"  # under $HEARING_MODELS_DIR


def arm_env(arm: str) -> dict:
    env = dict(os.environ)
    for k in ("HEARING_SEP_CKPT", "HEARING_PICK", "HEARING_GATED_FLOOR_DB"):
        env.pop(k, None)
    if arm == "finetuned":
        env["HEARING_SEP_CKPT"] = str(Path(os.environ["HEARING_MODELS_DIR"]) / FT_CKPT)
        env["HEARING_PICK"] = "fixed"
    return env


def run_arm(room: Path, out: Path, profiles: Path, target: str, other: str) -> None:
    """One arm in this process (the env decides the separator)."""
    mix = aio.load(room, SR)[0]
    scene = {"mix": mix}
    # named clips: the fixed pick (T-386) only engages for a profile clip that knows its name
    enroll = profile_clip(load_profile(target, profiles, SR), target)
    neg = profile_clip(load_profile(other, profiles, SR), other)
    hops = windows(scene, enroll, neg, "standard")
    y = render(scene, hops, {}, k5=False)
    gated = sum(h["gated"] for h in hops)
    sf.write(out, y.astype(np.float32), SR)
    print(json.dumps({"out": str(out), "windows": len(hops), "gated": gated}))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", type=Path, nargs="+")
    ap.add_argument("--target", default="aryan")
    ap.add_argument("--other", default="joseph")
    ap.add_argument("--profiles", type=Path,
                    default=Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated")) / "profiles")
    ap.add_argument("--out", type=Path, default=Path("~/Desktop/selective-hearing-finetune/live-replays").expanduser())
    ap.add_argument("--start", type=float, default=0.0, help="seconds into in.wav")
    ap.add_argument("--seconds", type=float, default=120.0, help="at most this much of in.wav")
    ap.add_argument("--arm", choices=["base", "finetuned"], help=argparse.SUPPRESS)  # internal: one arm here
    ap.add_argument("--room", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--dest", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.arm:
        run_arm(args.room, args.dest, args.profiles, args.target, args.other)
        return 0

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from t386_export_ab import level_match

    for sess in args.sessions:
        d = args.out / sess.name
        d.mkdir(parents=True, exist_ok=True)
        x, sr = sf.read(sess / "in.wav", dtype="float32")
        x = aio.resample(x, sr, SR) if sr != SR else x
        x = x[round(args.start * SR):][: round(args.seconds * SR)]
        raw = d / ".room_raw.wav"
        sf.write(raw, x, SR)
        for arm in ("base", "finetuned"):
            cmd = [sys.executable, "-m", "eval.live_replays", str(sess), "--arm", arm, "--room", str(raw),
                   "--dest", str(d / f"{arm}.wav"), "--profiles", str(args.profiles),
                   "--target", args.target, "--other", args.other]
            subprocess.run(cmd, check=True, env=arm_env(arm))
        sf.write(d / "room.wav", level_match(x), SR)
        for arm in ("base", "finetuned"):
            y, _ = sf.read(d / f"{arm}.wav", dtype="float32")
            sf.write(d / f"{arm}.wav", level_match(y), SR)
        raw.unlink()
        print(f"{d}: room/base/finetuned.wav, {len(x) / SR:.1f} s, target {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
