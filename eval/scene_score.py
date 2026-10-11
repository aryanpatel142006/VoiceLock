"""Scripted scenes and the target-selection scorer. Plain data, no model code."""
from __future__ import annotations

import argparse
import math
import sys
from itertools import pairwise
from pathlib import Path

import yaml

GRACE_S = 1.0
_EPS = 1e-9


def load_scene(path: str | Path) -> dict:
    scene = yaml.safe_load(Path(path).read_text())
    scene.setdefault("name", Path(path).stem)
    scene.setdefault("fps", 25)
    scene.setdefault("occlusions", [])
    ids = {a["id"] for a in scene["actors"]}
    for seg in scene["expected"]:
        if seg["target"] is not None and seg["target"] not in ids:
            raise ValueError(f"{scene['name']}: unknown expected target {seg['target']}")
    return scene


def _interp(keys: list[list[float]], t: float) -> tuple[float, float, float, float]:
    if t <= keys[0][0]:
        return tuple(keys[0][1:5])
    for k0, k1 in pairwise(keys):
        if t <= k1[0]:
            f = (t - k0[0]) / (k1[0] - k0[0])
            return tuple(a + f * (b - a) for a, b in zip(k0[1:5], k1[1:5]))
    return tuple(keys[-1][1:5])


def _inside(intervals: list[list[float]], t: float) -> bool:
    return any(t0 <= t < t1 for t0, t1 in intervals)


def n_frames(scene: dict) -> int:
    return round(scene["duration_s"] * scene["fps"])


def frame_times(scene: dict) -> list[float]:
    return [i / scene["fps"] for i in range(n_frames(scene))]


def synthesize_tracks(scene: dict) -> list[list[dict]]:
    """Per-frame lists of track dicts (frame i is at t = i / fps)."""
    frames = []
    for t in frame_times(scene):
        tt = t + _EPS
        tracks = []
        for a in scene["actors"]:
            if any(o["actor"] == a["id"] and o["t0"] <= tt < o["t1"] for o in scene["occlusions"]):
                continue
            tracks.append({
                "track_id": a["id"],
                "box": _interp(a["box"], t),
                "p_speaking": 1.0 if _inside(a["speaking"], tt) else 0.0,
                "last_seen_ts": t,
            })
        frames.append(tracks)
    return frames


def expected_target(scene: dict, t: float) -> int | None:
    segs = scene["expected"]
    if t < segs[0]["t0"]:
        return segs[0]["target"]
    for seg in segs:
        if seg["t0"] <= t + _EPS < seg["t1"]:
            return seg["target"]
    return segs[-1]["target"]


def expected_switches(scene: dict) -> list[tuple[float, int | None]]:
    segs = scene["expected"]
    return [(b["t0"], b["target"]) for a, b in pairwise(segs) if a["target"] != b["target"]]


def oracle_predict(scene: dict) -> list[tuple[float, int | None]]:
    """Most central speaking face (distance of box centre to image centre)."""
    preds = []
    for t, tracks in zip(frame_times(scene), synthesize_tracks(scene)):
        best, best_d = None, math.inf
        for tr in tracks:
            if tr["p_speaking"] <= 0.5:
                continue
            x, y, w, h = tr["box"]
            d = math.hypot(x + w / 2 - 0.5, y + h / 2 - 0.5)
            if d < best_d:
                best, best_d = tr["track_id"], d
        preds.append((t, best))
    return preds


def score(scene: dict, predicted: list[tuple[float, int | None]]) -> dict:
    switches = expected_switches(scene)

    def in_grace(t: float) -> bool:
        return any(s <= t + _EPS < s + GRACE_S for s, _ in switches)

    correct = total = 0
    false_switches = 0
    prev = None
    for i, (t, pid) in enumerate(predicted):
        exp = expected_target(scene, t)
        if not in_grace(t):
            total += 1
            correct += pid == exp
            if i > 0 and pid != prev and pid != exp:
                false_switches += 1
        prev = pid

    latencies = []
    bounds = [s for s, _ in switches] + [scene["duration_s"]]
    for (s, target), end in zip(switches, bounds[1:]):
        lat = next((t - s for t, pid in predicted if s - _EPS <= t < end and pid == target), None)
        latencies.append(lat)
    found = [x for x in latencies if x is not None]
    return {
        "scene": scene["name"],
        "accuracy": correct / total if total else 1.0,
        "switch_latency_s": latencies,
        "mean_switch_latency_s": sum(found) / len(found) if found else None,
        "false_switches": false_switches,
        "frames_scored": total,
    }


def score_many(scenes: list[dict], predictor) -> dict:
    per = [score(s, predictor(s)) for s in scenes]
    return {
        "accuracy": sum(p["accuracy"] for p in per) / len(per),
        "false_switches": sum(p["false_switches"] for p in per),
        "per_scene": per,
    }


def load_dir(path: str | Path) -> list[dict]:
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.yaml"))
    return [load_scene(f) for f in files]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--oracle", action="store_true", help="score the most-central-speaking-face oracle")
    args = ap.parse_args(argv)
    if not args.oracle:
        ap.error("only --oracle is supported from the CLI")
    res = score_many(load_dir(args.path), oracle_predict)
    print(f"{'scene':<20}{'accuracy':>10}{'mean latency s':>16}{'false switches':>16}")
    for p in res["per_scene"]:
        lat = p["mean_switch_latency_s"]
        print(f"{p['scene']:<20}{p['accuracy']:>10.3f}{'n/a' if lat is None else f'{lat:.2f}':>16}{p['false_switches']:>16}")
    print(f"{'ALL':<20}{res['accuracy']:>10.3f}{'':>16}{res['false_switches']:>16}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
