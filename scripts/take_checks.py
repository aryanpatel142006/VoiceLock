"""Rehearsal take checks for the re-enrolled profiles (T-172).

    .venv/bin/python scripts/take_checks.py absent results/sessions/<dir> --start 5 --end 30
    .venv/bin/python scripts/take_checks.py listen --takes key2-aryan key2-joseph absent-joseph

``absent``: a recorded session (``HEARING_SESSION_LOG``: ``in.wav``, ``out.wav``, ``out_pre_agc.wav`` when
present) in which the target stays silent and the other talker speaks. Over the stretch, in
``BLOCK_S`` blocks with the output shifted back by ``--delay-ms`` (the tier's delay, default 1500 ms
Mac / pass 950 for the A100): **output level re input** = level of the output in the blocks where the
input is voiced minus the input's level there (dB; more negative = more of the other talker removed),
and **wrong-speaker share** = share of those blocks whose output is voiced (above ``VOICED_DB``) and no
more than ``LEAK_DB`` under the input: the other person playing while the target is silent. Prints one
markdown row. Leveling (AGC) would hide the drop, so ``out_pre_agc.wav`` is used when it exists.

``listen``: asks the listener four yes/no questions per take (wrong person, leak, robotic, delay) and
writes ``results/listening_<stamp>.md``, one line per take.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hearing.audio.io import PIPELINE_SR, load
from hearing.tse.enroll import level_db

BLOCK_S = 0.5
VOICED_DB = -50.0  # a block at or above this level counts as sound (input talker / output leak)
LEAK_DB = 20.0  # an output block within this many dB of the input block is the other talker playing
QUESTIONS = ("wrong person", "leak", "robotic", "delay")


def target_absent(x_in: np.ndarray, x_out: np.ndarray, sr: int = PIPELINE_SR, delay_ms: float = 1500.0,
                  start: float = 0.0, end: float | None = None) -> dict:
    """``{"blocks", "out_re_in_db", "wrong_share"}`` over the voiced input blocks of [start, end)."""
    d = round(delay_ms / 1000 * sr)
    x_out = np.asarray(x_out, np.float32)[d:]
    n = min(len(x_in), len(x_out))
    a, b = round(start * sr), n if end is None else min(n, round(end * sr))
    blk = round(BLOCK_S * sr)
    rows = [(x_in[i : i + blk], x_out[i : i + blk]) for i in range(a, b - blk + 1, blk)]
    voiced = [(i, o) for i, o in rows if level_db(i) >= VOICED_DB]
    if not voiced:
        return {"blocks": 0, "out_re_in_db": None, "wrong_share": None}
    ins, outs = np.concatenate([i for i, _ in voiced]), np.concatenate([o for _, o in voiced])
    wrong = [level_db(o) >= VOICED_DB and level_db(o) >= level_db(i) - LEAK_DB for i, o in voiced]
    return {"blocks": len(voiced), "out_re_in_db": level_db(outs) - level_db(ins), "wrong_share": float(np.mean(wrong))}


def absent_row(r: dict, label: str = "Target absent (other talker only)") -> str:
    if not r["blocks"]:
        return f"| {label} | NOT MEASURED (no voiced input in the stretch) | NOT MEASURED |"
    return (f"| {label} | output {r['out_re_in_db']:+.1f} dB re input over {r['blocks'] * BLOCK_S:.1f} s "
            f"| wrong-speaker share {100 * r['wrong_share']:.0f} % |")


def listening_log(takes: list[str], ask: Callable[[str], str], out: Path, stamp: str | None = None) -> Path:
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    lines = [f"# Listening log {stamp}", "", "| take | " + " | ".join(QUESTIONS) + " |",
             "|---|" + "---|" * len(QUESTIONS)]
    for take in takes:
        cells = []
        for q in QUESTIONS:
            a = ""
            while a not in ("y", "n"):
                a = ask(f"{take}: {q}? [y/n] ").strip().lower()[:1]
            cells.append("yes" if a == "y" else "no")
        lines.append(f"| {take} | " + " | ".join(cells) + " |")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"listening_{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv: list[str] | None = None, *, ask: Callable[[str], str] = input) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("absent", help="output level and wrong-speaker share while the target is silent")
    p.add_argument("session", type=Path, help="a recorded session dir (in.wav, out.wav)")
    p.add_argument("--start", type=float, default=0.0, help="stretch start (s, input clock)")
    p.add_argument("--end", type=float, default=None, help="stretch end (s)")
    p.add_argument("--delay-ms", type=float, default=1500.0, help="output delay to undo (Mac 1500, A100 950)")
    q = sub.add_parser("listen", help="four yes/no questions per take -> results/listening_<stamp>.md")
    q.add_argument("--takes", nargs="+", required=True)
    q.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    if args.cmd == "listen":
        print(listening_log(args.takes, ask, args.out))
        return 0
    out_wav = args.session / "out_pre_agc.wav"
    out_wav = out_wav if out_wav.exists() else args.session / "out.wav"
    x_in, _ = load(args.session / "in.wav", PIPELINE_SR)
    x_out, _ = load(out_wav, PIPELINE_SR)
    r = target_absent(x_in, x_out, delay_ms=args.delay_ms, start=args.start, end=args.end)
    print(f"# {args.session.name} ({out_wav.name}, delay {args.delay_ms:.0f} ms, {args.start:.1f}-"
          f"{'end' if args.end is None else f'{args.end:.1f}'} s)")
    print(absent_row(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
