"""Merge eval_sep.py JSONs into one markdown table and a ship verdict (T-374).

Ship rule (task acceptance): a candidate ships only if, on every team row, its wrong-speaker %
is lower and its SI-SDRi higher than base's (both with the ECAPA pick; the fixed-order run
also with its fixed pick), and the generality rows do not drop.

    python experiments/gpu/T-374/table.py eval_base.json eval_fixed.json eval_pit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(s):
    if s["mean"] is None:
        return "- (0)"
    return f"{s['mean']:+.1f} [{s['lo']:+.1f}, {s['hi']:+.1f}] ({s['n']})"


def main(paths):
    rows = {}
    for path in paths:
        for row, cols in json.loads(Path(path).read_text())["rows"].items():
            rows.setdefault(row, {}).update(cols)
    labels = sorted({c for cols in rows.values() for c in cols}, key=lambda c: (c.split("/")[0] != "base", c))
    print("| row | checkpoint/pick | wrong-speaker % [95% CI] (windows) | SI-SDRi dB | leak dB |")
    print("|---|---|---|---|---|")
    for row, cols in rows.items():
        for label in labels:
            if label in cols:
                s = cols[label]
                print(f"| {row} | {label} | {fmt(s['wrong_pct'])} | {fmt(s['sdri_db'])} | {fmt(s['leak_db'])} |")
    verdict = {}
    for label in labels:
        if label.startswith("base/"):
            continue
        wins, total = 0, 0
        for row, cols in rows.items():
            if label not in cols or "base/ecapa" not in cols:
                continue
            base, cand = cols["base/ecapa"], cols[label]
            total += 1
            better = (
                cand["wrong_pct"]["mean"] < base["wrong_pct"]["mean"]
                and cand["sdri_db"]["mean"] > base["sdri_db"]["mean"]
            )
            wins += better
        verdict[label] = f"beats base/ecapa on {wins}/{total} rows" + (" -> SHIP" if wins == total else "")
    print()
    for label, text in verdict.items():
        print(f"- {label}: {text}")


if __name__ == "__main__":
    main(sys.argv[1:])
