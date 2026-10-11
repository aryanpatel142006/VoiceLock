"""Print the better fine-tuned init (T-374 vs T-387) by their eval tables (T-391).

Score = mean over the team rows (holdout/same, both targets) of the fixed pick's SI-SDRi minus
0.1 x its wrong-speaker % (1 dB per 10 points). Each table holds one fine-tuned label besides
base; prints the winner's checkpoint path. A missing or unreadable table loses.

    python pick_init.py T-374/eval_fixed.json=T-374/run_fixed/best.pt \
        T-374b/eval_fixed2.json=T-374b/run_fixed/best.pt
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def score(table: Path) -> float:
    rows = json.loads(table.read_text())["rows"]
    vals = []
    for name, cols in rows.items():
        if name.startswith(("aryan:joseph", "joseph:aryan")):
            for col, s in cols.items():
                if col.endswith("/fixed") and not col.startswith("base/"):
                    vals.append(s["sdri_db"]["mean"] - 0.1 * s["wrong_pct"]["mean"])
    return sum(vals) / len(vals) if vals else float("-inf")


def main(argv: list[str]) -> None:
    best, best_score = None, float("-inf")
    for spec in argv:
        table, ckpt = spec.split("=", 1)
        try:
            s = score(Path(table))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"{table}: unreadable ({exc})", file=sys.stderr)
            continue
        print(f"{table}: score {s:.2f}", file=sys.stderr)
        if s > best_score and Path(ckpt).exists():
            best, best_score = ckpt, s
    if best is None:
        sys.exit("no usable init")
    print(best)


if __name__ == "__main__":
    main(sys.argv[1:])
