"""Does loudness explain wrong-speaker windows? (T-383, MUSE digest N4)

Reads a T-323 wrong_speaker_audit JSON and, per window, computes the target/other stem level ratio (dB, RMS of the
clean ``target.wav`` / ``other.wav`` stems over the window's 1 s span and over its owned last 0.5 s hop), then bins the
wrong windows by it, per speaker x difficulty. A "quiet target" would show wrong windows concentrated at low ratios.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hearing.audio import io as aio

SR, WIN_S, HOP_S = 16000, 1.0, 0.5
EDGES = [-np.inf, -6, 0, 6, np.inf]
LABELS = ["<-6", "-6..0", "0..6", ">6"]


def level_db(x: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(x**2)) + 1e-12)


def bin_of(r: float) -> int:
    return int(np.searchsorted(EDGES, r, side="right") - 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit")
    a = ap.parse_args(argv)
    d = json.loads(Path(a.audit).read_text())
    cells: dict[str, dict] = {}
    for sc in d["scenes"]:
        scene = Path(sc["scene"])
        diff = scene.name.split("-")[-1]
        t = aio.load(scene / "stems" / "target.wav", SR)[0]
        o = aio.load(scene / "stems" / "other.wav", SR)[0]
        c = cells.setdefault(f"{sc['target']} {diff}", {"all": [[], []], "wrong": [[], []], "act": [0, 0], "n": 0, "w": 0})
        for r in sc["rows"]:
            i = round(r["t"] * SR)
            win = slice(i, i + round(WIN_S * SR))
            hop = slice(i + round(HOP_S * SR), i + round(WIN_S * SR))
            ratio = level_db(t[win]) - level_db(o[win])
            ratio_hop = level_db(t[hop]) - level_db(o[hop])
            c["n"] += 1
            c["all"][0].append(ratio)
            if r["wrong"]:
                c["w"] += 1
                c["wrong"][0].append(ratio)
                c["wrong"][1].append(ratio_hop)
                c["act"][0] += bool(r["target_active"])
                c["act"][1] += 1
    print("| cell | windows | wrong | " + " | ".join(LABELS) + " | median wrong (dB) | median all (dB) | target active in wrong |")
    print("|---|---|---|" + "---|" * (len(LABELS) + 3))
    tot = [0] * len(LABELS)
    for k in sorted(cells):
        c = cells[k]
        h = [0] * len(LABELS)
        for r in c["wrong"][0]:
            h[bin_of(r)] += 1
        tot = [x + y for x, y in zip(tot, h)]
        mw = np.median(c["wrong"][0]) if c["wrong"][0] else float("nan")
        print(f"| {k} | {c['n']} | {c['w']} | " + " | ".join(map(str, h)) +
              f" | {mw:+.1f} | {np.median(c['all'][0]):+.1f} | {c['act'][0]}/{c['act'][1]} |")
    allb = [0] * len(LABELS)
    for c in cells.values():
        for r in c["all"][0]:
            allb[bin_of(r)] += 1
    print("| all windows |  | " + str(sum(allb)) + " | " + " | ".join(map(str, allb)) + " | | | |")
    print("| wrong rate | | | " + " | ".join(f"{w / n:.0%}" if n else "-" for w, n in zip(tot, allb)) + " | | | |")
    print("| total | | " + str(sum(c["w"] for c in cells.values())) + " | " + " | ".join(map(str, tot)) + " | | | |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
