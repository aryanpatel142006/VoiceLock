"""T-386: level-matched A/B wavs of the fine-tuned separator arms to a listening folder (Desktop by default).

    .venv/bin/python scripts/t386_export_ab.py $HEARING_RESULTS_DIR/T-386_rows.json --scene holdout-sir0-cafe \
        --out ~/Desktop/selective-hearing-finetune

Per target: ``<name>-0-input-mix.wav`` (the session's in.wav) and ``<name>-<i>-<arm>.wav`` (each arm's out.wav), all 16 kHz
mono, active speech at -26 dBFS (20 ms frames within 30 dB of the 95th-percentile frame), peak at or below -1 dBFS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ARMS = ("base", "base+floor20", "ft+fixed", "ft+fixed+floor20")
TARGET_DB, PEAK_DB, FRAME = -26.0, -1.0, 320


def level_match(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64)
    n = len(x) // FRAME * FRAME
    e = np.mean(x[:n].reshape(-1, FRAME) ** 2, axis=1) + 1e-12
    db = 10 * np.log10(e)
    active = e[db >= np.percentile(db, 95) - 30.0]
    gain = 10 ** ((TARGET_DB - 10 * np.log10(np.mean(active))) / 20)
    y = x * gain
    peak = np.max(np.abs(y)) if len(y) else 0.0
    limit = 10 ** (PEAK_DB / 20)
    if peak > limit:
        y *= limit / peak
    return y.astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("rows", type=Path)
    ap.add_argument("--scene", default="holdout-sir0-cafe")
    ap.add_argument("--out", type=Path, default=Path("~/Desktop/selective-hearing-finetune").expanduser())
    args = ap.parse_args(argv)
    rows = [r for r in json.loads(args.rows.read_text())["rows"] if r["scene"] == args.scene and r.get("session")]
    args.out.mkdir(parents=True, exist_ok=True)
    for target in sorted({r["target"] for r in rows}):
        mine = {r["label"]: r for r in rows if r["target"] == target}
        first = next(iter(mine.values()))
        x, sr = sf.read(Path(first["session"]) / "in.wav", dtype="float32")
        sf.write(args.out / f"{target}-0-input-mix.wav", level_match(x), sr)
        for i, arm in enumerate(ARMS, 1):
            if arm not in mine:
                continue
            y, sr = sf.read(Path(mine[arm]["session"]) / "out.wav", dtype="float32")
            name = f"{target}-{i}-{arm.replace('+', '-')}.wav"
            sf.write(args.out / name, level_match(y), sr)
            print(f"{name}: wrong-speaker {mine[arm].get('od_pct')} %, leak {mine[arm].get('leak')} dB")


if __name__ == "__main__":
    main()
