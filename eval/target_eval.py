"""Score `hearing.vision.target.TargetSelector` on the scripted scene set.

`eval/scene_score.py` stays free of `hearing` imports; this module bridges the two.
Run `python -m eval.target_eval eval/scenes/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.scene_score import frame_times, load_dir, score_many, synthesize_tracks
from hearing.types import FaceTrack
from hearing.vision.target import ATTENTION_PRESETS, TargetSelector


def to_face_tracks(frame: list[dict]) -> list[FaceTrack]:
    return [
        FaceTrack(
            track_id=d["track_id"],
            box=tuple(d["box"]),
            landmarks=None,
            p_speaking=d["p_speaking"],
            last_seen_ts=d["last_seen_ts"],
        )
        for d in frame
    ]


def make_predictor(**selector_kwargs):
    """Scene -> [(t, track_id | None)] with a fresh `TargetSelector` per scene."""

    def predict(scene: dict) -> list[tuple[float, int | None]]:
        sel = TargetSelector(**selector_kwargs)
        return [
            (t, sel.update(to_face_tracks(frame), t).track_id)
            for t, frame in zip(frame_times(scene), synthesize_tracks(scene))
        ]

    return predict


def evaluate(path: str | Path, **selector_kwargs) -> dict:
    res = score_many(load_dir(path), make_predictor(**selector_kwargs))
    lats = [lat for p in res["per_scene"] for lat in p["switch_latency_s"]]
    found = [lat for lat in lats if lat is not None]
    res["switch_latencies_s"] = lats
    res["missed_switches"] = len(lats) - len(found)
    res["mean_switch_latency_s"] = sum(found) / len(found) if found else None
    res["max_switch_latency_s"] = max(found) if found else None
    return res


def print_table(res: dict) -> None:
    print(f"{'scene':<20}{'accuracy':>10}{'mean latency s':>16}{'false switches':>16}")
    for p in res["per_scene"]:
        lat = p["mean_switch_latency_s"]
        print(f"{p['scene']:<20}{p['accuracy']:>10.3f}{'n/a' if lat is None else f'{lat:.2f}':>16}{p['false_switches']:>16}")
    print(f"{'ALL':<20}{res['accuracy']:>10.3f}{'':>16}{res['false_switches']:>16}")
    mean, worst = res["mean_switch_latency_s"], res["max_switch_latency_s"]
    if mean is not None:
        print(f"switches: {len(res['switch_latencies_s'])}, missed {res['missed_switches']}, "
              f"mean latency {mean:.2f} s, max {worst:.2f} s")
    print(f"overall accuracy: {res['accuracy']:.3f}")


def print_presets(path: str | Path) -> None:
    scenes = [p["scene"] for p in evaluate(path)["per_scene"]]
    print("| preset | dwell_s | hold_s | margin | " + " | ".join(scenes) + " | all | false switches |")
    print("|---|---|---|---|" + "---|" * (len(scenes) + 2))
    for name, v in ATTENTION_PRESETS.items():
        res = evaluate(path, dwell_s=v["dwell_s"], hold_s=v["hold_s"], margin=v["switch_margin"])
        accs = " | ".join(f"{p['accuracy']:.3f}" for p in res["per_scene"])
        print(f"| {name} | {v['dwell_s']} | {v['hold_s']} | {v['switch_margin']} | {accs} | "
              f"{res['accuracy']:.3f} | {res['false_switches']} |")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default=str(Path(__file__).parent / "scenes"))
    ap.add_argument("--min-accuracy", type=float, default=0.9)
    ap.add_argument("--max-latency", type=float, default=1.0, help="fail if any expected switch is missed or slower")
    ap.add_argument("--dwell-s", type=float)
    ap.add_argument("--margin", type=float)
    ap.add_argument("--hold-s", type=float)
    ap.add_argument("--acquire-s", type=float)
    ap.add_argument("--presets", action="store_true", help="score Glance / Normal / Sticky and print a markdown table")
    ap.add_argument("--no-speech-gate", action="store_true", help="pure centre-of-view (speech_memory_s=None)")
    args = ap.parse_args(argv)
    if args.presets:
        print_presets(args.path)
        return 0
    kwargs = {k: v for k, v in {"dwell_s": args.dwell_s, "margin": args.margin, "hold_s": args.hold_s,
                                "acquire_s": args.acquire_s}.items() if v is not None}
    if args.no_speech_gate:
        kwargs["speech_memory_s"] = None
    res = evaluate(args.path, **kwargs)
    print_table(res)
    ok = res["accuracy"] >= args.min_accuracy
    if not ok:
        print(f"FAIL: accuracy {res['accuracy']:.3f} < {args.min_accuracy}")
    worst = res["max_switch_latency_s"]
    if res["missed_switches"] or (worst is not None and worst >= args.max_latency):
        print(f"FAIL: missed {res['missed_switches']} switches, max latency {worst} s (limit {args.max_latency})")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
