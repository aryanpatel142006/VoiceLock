"""Replay the output AGC offline on recorded sessions and score before/after on the same audio (T-179).

The judge profile's AGC is the last step before playout and has no look-ahead, so ``out_pre_agc.wav``
run through an :class:`~hearing.audio.agc.Agc` is exactly the ``out.wav`` that AGC would have played.
For every session dir (with ``out_pre_agc.wav``) this renders ``out.wav`` with the T-169 AGC
(``before``: the shipped default, +12 dB max, a held block may raise the gain) and the opt-in T-179 floor rule
(``after``, :data:`~hearing.audio.agc.FLOOR_RULE`), scores both with ``session_score`` (same alignment, same
stems; the pre-AGC SI-SDRi on out.wav's own delay-aligned crop) and prints one row each:

    .venv/bin/python scripts/agc_replay.py results/sessions/<dir> ... --manifest eval/fixtures/demo/demo-mf-hard/manifest.json
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_score as ss

from hearing.audio.agc import FLOOR_RULE, Agc

VARIANTS = {"before": {}, "after": FLOOR_RULE}


def write_wav(path: Path, y: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(ss.SR)
        w.writeframes((np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


def render(session: Path, out: Path, **agc_kw) -> Path:
    """A copy of ``session`` whose out.wav is out_pre_agc.wav through ``Agc(**agc_kw)`` (in 10 ms chunks)."""
    out.mkdir(parents=True, exist_ok=True)
    for name in ("in.wav", "events.jsonl", "out_pre_agc.wav"):
        dst = out / name
        if not dst.exists():
            os.symlink((session / name).resolve(), dst)
    pre = ss.read_wav(session / "out_pre_agc.wav")
    agc = Agc(ss.SR, **agc_kw)
    y = np.concatenate([agc.process_array(pre[i:i + 160]) for i in range(0, len(pre), 160)])
    write_wav(out / "out.wav", y)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--keep", type=Path, default=None, help="write the rendered sessions here (default: a temp dir)")
    args = ap.parse_args(argv)
    stems = ss.resolve_stems(ss.main_fallback(args.manifest) if args.manifest else None)
    root = args.keep or Path(tempfile.mkdtemp(prefix="agc-replay-"))
    print("| session | AGC | SI-SDRi | SI-SDRi pre-AGC (same crop) | gap dB | leak dB | true dropouts s | pause lift dB "
          "| speech level mean / p90 dBFS | pre-AGC speech mean dBFS |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in args.sessions:
        if not (d / "out_pre_agc.wav").exists():
            print(f"!! {d}: no out_pre_agc.wav", file=sys.stderr)
            continue
        for name, kw in VARIANTS.items():
            try:
                s = ss.score_session(render(d, root / f"{d.name}-{name}", **kw), stems, dnsmos="none")
            except ValueError as e:  # not this manifest's clip
                print(f"!! {e}", file=sys.stderr)
                break
            pre = s.si_sdri_pre_agc_same_crop_db
            print(f"| {d.name} | {name} | {s.si_sdri_db:+.2f} | {pre:+.2f} | {s.si_sdri_db - pre:+.2f} | {s.leak_db:+.2f} | "
                  f"{s.true_dropout_s:.2f} | {s.pause_lift_db:+.2f} | {s.speech_level_db:.1f} / {s.speech_level_p90_db:.1f} | "
                  f"{s.speech_level_pre_agc_db:.1f} |", flush=True)
    if args.keep is None:
        shutil.rmtree(root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
