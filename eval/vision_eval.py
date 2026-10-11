"""Score detect → track → centre/dwell selection on GRID AV fixture videos.

T-008 labels the audio target for the whole clip, not a gaze trajectory. Speech
intervals are NOT target-switch annotations: silent targets remain expected.
The selector's speech eligibility gate is disabled because this evaluation has
no ASD stage. Slot boxes are approximate fixed GRID placements, used only for
identity mapping, never to steer selection. Static two-face composites may put
the designated target further from centre than the interferer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from eval.fixtures.video import read_video
from eval.scene_score import score
from hearing import device
from hearing.types import VideoFrame
from hearing.vision.faces import FaceBackend, FacePipeline, MediaPipeBackend
from hearing.vision.target import TargetSelector

FPS = 25


def _centre(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _timing(times: list[float]) -> dict:
    return {
        "fps": 1000 * len(times) / sum(times),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
    }


def evaluate_frames(manifest: dict, frames: Iterable[np.ndarray], backend: FaceBackend) -> dict:
    """Offline-injectable scoring path; all detections pass through the real tracker.

    Map each track to the nearest manifest box centre; retain the last mapping
    while the selector holds a missing track. Actor ids differ from utterance ids
    (`target_id`), so the manifest's role establishes the expected actor.
    """
    faces = manifest["faces"]
    target = next(f["actor"] for f in faces if f["role"] == "target")
    duration = float(manifest["duration_s"])
    scene = {
        "name": manifest["id"], "duration_s": duration,
        "expected": [{"t0": 0.0, "t1": duration, "target": target}],
    }
    pipeline = FacePipeline(backend)
    pipeline.reset()
    selector = TargetSelector(speech_memory_s=None)
    identities, predicted, times = {}, [], []
    seen_frames = landmarks_frames = 0
    track_counts = []
    for i, image in enumerate(frames):
        ts = i / FPS
        start = time.perf_counter()
        tracks = pipeline.update(VideoFrame(image, seq=i, capture_ts=ts))
        for track in tracks:
            cx, cy = _centre(track.box)
            nearest = min(faces, key=lambda f: math.dist((cx, cy), _centre(f["box"])))
            identities[track.track_id] = nearest["actor"]
        state = selector.update(tracks, ts)
        predicted.append((ts, identities.get(state.track_id)))
        times.append((time.perf_counter() - start) * 1000)
        # Separate visible tracks from the tracker's retained missing tracks.
        visible = [t for t in tracks if abs(t.last_seen_ts - ts) < 1e-6]
        seen_frames += bool(visible)
        landmarks_frames += any(t.landmarks is not None for t in visible)
        track_counts.append(len(visible))
    expected_count = round(duration * FPS)
    if not predicted or len(predicted) != expected_count:
        raise ValueError(f"{manifest['id']}: decoded frame count {len(predicted)} != expected {expected_count}")
    return {
        **score(scene, predicted), **_timing(times),
        "frames": len(predicted), "frames_with_faces": seen_frames,
        "frames_with_landmarks": landmarks_frames,
        "mean_faces": float(np.mean(track_counts)), "frame_ms": times,
        "target_id": manifest["target_id"], "expected_actor": target,
        "layout": manifest["layout"],
    }


def evaluate_fixture(path: str | Path, backend: FaceBackend) -> dict:
    path = Path(path)
    manifest = json.loads(path.read_text())
    video = manifest.get("paths", {}).get("video", manifest.get("video", "video.mp4"))
    frames, _ = read_video(path.parent / video, fps=FPS, max_size=(640, 360))
    start = time.perf_counter()
    result = evaluate_frames(manifest, frames, backend)
    result["pipeline_wall_s"] = time.perf_counter() - start
    result["manifest"] = str(path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="AV fixture set directory")
    parser.add_argument("--min-accuracy", type=float, default=0.9)
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)
    if not 0 <= args.min_accuracy <= 1:
        parser.error("--min-accuracy must be in [0, 1]")
    paths = sorted(args.path.glob("*/manifest.json"))
    if not paths:
        print(f"vision_eval: no fixture manifests under {args.path}", file=sys.stderr)
        return 2
    per = []
    backend = None
    try:
        backend = MediaPipeBackend()
        for path in paths:
            per.append(evaluate_fixture(path, backend))
    except (OSError, RuntimeError, ValueError, KeyError, StopIteration) as exc:
        print(f"vision_eval: {exc}", file=sys.stderr)
        return 2
    finally:
        if backend is not None and hasattr(backend, "close"):
            backend.close()
    times = [ms for result in per for ms in result.pop("frame_ms")]
    scored = sum(r["frames_scored"] for r in per)
    if scored == 0:
        print("vision_eval: no frames scored", file=sys.stderr)
        return 2
    accuracy = sum(r["accuracy"] * r["frames_scored"] for r in per) / scored
    two_face = [r for r in per if r["layout"] == "two_face"]
    two_scored = sum(r["frames_scored"] for r in two_face)
    two_accuracy = (sum(r["accuracy"] * r["frames_scored"] for r in two_face) / two_scored
                    if two_scored else None)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")
    result = {
        "stamp": stamp, "fixture_set": str(args.path), "device": device.describe(),
        "backend": getattr(backend, "device", "mediapipe"), "video_fps": FPS,
        "accuracy": accuracy, "two_face_accuracy": two_accuracy,
        "false_switches": sum(r["false_switches"] for r in per),
        "switch_latency_s": [lat for r in per for lat in r["switch_latency_s"]],
        "frames_scored": scored, **_timing(times), "per_fixture": per,
        "protocol": "Fixed manifest target; pure centre/dwell (no ASD); nearest-slot identity mapping. "
                    "Compute times include detect/landmark/track/select/mapping, exclude decode; no warmup.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"vision_{stamp}.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{'fixture':<42} {'accuracy':>9} {'latency s':>10} {'false switches':>14} {'fps':>8} {'p50 ms':>8} {'p95 ms':>8}")
    for r in per:
        lat = r["mean_switch_latency_s"]
        latency = "n/a" if lat is None else f"{lat:.3f}"
        print(f"{r['scene']:<42} {r['accuracy']:>9.3f} {latency:>10} {r['false_switches']:>14} "
              f"{r['fps']:>8.1f} {r['p50_ms']:>8.2f} {r['p95_ms']:>8.2f}")
    print(f"ALL accuracy={accuracy:.3f}, two_face_accuracy={two_accuracy}, "
          f"false_switches={result['false_switches']}, fps={result['fps']:.1f}, "
          f"p50/p95={result['p50_ms']:.2f}/{result['p95_ms']:.2f} ms\n{output}")
    # Single-face fixtures must not hide a two-face regression.
    return int(accuracy < args.min_accuracy or (two_accuracy is not None and two_accuracy < args.min_accuracy))


if __name__ == "__main__":
    raise SystemExit(main())
