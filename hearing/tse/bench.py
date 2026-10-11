"""Focus HQ bench: the streaming ``tse`` stage on an AV fixture set, fed like the live session.

    .venv/bin/python -m hearing.tse.bench --set av_tiny                      # auto pick + identity
    .venv/bin/python -m hearing.tse.bench --candidates dolphin,av_mossformer2,identity --lips mediapipe

Per fixture the video is decoded at 25 fps; each frame's target face gives a lip crop
(:func:`hearing.tse.lips.lip_crop`) that goes into a :class:`~hearing.tse.lips.LipFeed`, and the
mix is streamed through :class:`~hearing.tse.stage.FocusHQ` in 10 ms chunks with the target in
``ctx`` exactly as the session does. ``--lips box`` crops at the manifest's target face box
(fixed; GRID talkers are frontal and static), ``--lips mediapipe`` runs the face detector +
tracker per frame and crops at the landmarks of the track over the target box (the live path).

Windows run inline (``threaded=false``) so the numbers are deterministic; the reported delay is
the live, threaded one (window + hop). Prints one table (SI-SDRi of the target estimate vs the
mixture, compute per window, RTF = compute / hop, skipped windows, device) overall and per layout
and writes ``eval/results/focus_hq_<stamp>.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hearing.audio import io
from hearing.tse.lips import FEED_KEY, LipFeed, lip_crop
from hearing.tse.stage import FocusHQ, resolve_candidate
from hearing.types import AudioChunk, FaceTrack, StageContext, TargetState

log = logging.getLogger("hearing.tse.bench")

ROOT = Path(__file__).resolve().parents[2]
SETS = Path(os.environ["HEARING_FIXTURES_DIR"]) if os.environ.get("HEARING_FIXTURES_DIR") else ROOT / "eval" / "fixtures" / "generated"
RESULTS = Path(os.environ["HEARING_RESULTS_DIR"]) if os.environ.get("HEARING_RESULTS_DIR") else ROOT / "eval" / "results"
HOP = 160
TARGET_ID = 1


def fixture_dirs(set_name: str) -> list[Path]:
    root = Path(set_name) if Path(set_name).is_dir() else SETS / set_name
    dirs = sorted(p for p in root.iterdir() if (p / "manifest.json").exists()) if root.is_dir() else []
    if not dirs:
        raise FileNotFoundError(
            f"no AV fixtures under {root}; run: .venv/bin/python -m eval.make_fixtures --av --tiny"
        )
    return dirs


def target_box(manifest: dict) -> tuple[float, float, float, float]:
    faces = manifest.get("faces") or []
    face = next((f for f in faces if f.get("role") == "target"), None)
    box = face["box"] if face else manifest.get("target_box")
    if box is None:
        raise ValueError(f"{manifest.get('id')}: no target face box in the manifest")
    return tuple(float(v) for v in box)


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    w = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    h = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = w * h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def lip_frames(frames: np.ndarray, fps: float, box, mode: str) -> list[tuple[float, np.ndarray | None]]:
    """``(ts, crop or None)`` per video frame for the target face."""
    if mode == "box":
        return [(i / fps, lip_crop(f, box)) for i, f in enumerate(frames)]
    from hearing.server.vision import get_face_detector
    from hearing.vision.tracker import IouTracker

    detector = get_face_detector("mediapipe")
    if detector is None:
        raise RuntimeError("mediapipe or its weights missing: python -m hearing.vision.faces --download")
    tracker, out = IouTracker(), []
    try:
        for i, f in enumerate(frames):
            ts = i / fps
            tracks = tracker.update(detector.detect(f, ts), ts)
            best = max(tracks, key=lambda t: _iou(t.box, box), default=None)
            if best is None or _iou(best.box, box) < 0.1 or best.last_seen_ts != ts:
                out.append((ts, None))
            else:
                out.append((ts, lip_crop(f, best.box, best.landmarks)))
    finally:
        detector.close()
    return out


def run_fixture(stage: FocusHQ, mix: np.ndarray, lips: list, box) -> tuple[np.ndarray, list]:
    """Stream ``mix`` through ``stage`` with the target's lips in the feed; aligned output + windows."""
    stage.reset()
    ctx = StageContext.default()
    feed = ctx.telemetry.extras[FEED_KEY] = LipFeed()
    ctx.target = TargetState(TARGET_ID, 1.0, True, 0.0)
    ctx.tracks = [FaceTrack(TARGET_ID, box, None, 1.0, 0.0)]
    pad = stage.delay + stage.hop + HOP
    x = np.concatenate([mix.astype(np.float32), np.zeros(pad, np.float32)])
    outs, vi = [], 0
    for i in range(0, len(x) - HOP + 1, HOP):
        ts = i / io.PIPELINE_SR
        while vi < len(lips) and lips[vi][0] < ts + HOP / io.PIPELINE_SR:
            if lips[vi][1] is not None:
                feed.append(lips[vi][0], TARGET_ID, lips[vi][1], 1.0)
            vi += 1
        outs.append(stage.process(AudioChunk(x[i : i + HOP], io.PIPELINE_SR, i // HOP, ts), ctx).samples)
    y = np.concatenate(outs)[stage.delay : stage.delay + len(mix)]
    return y, list(stage._recent)


def bench(candidates: list[str], set_name: str, window_s: float, hop_s: float, lips_mode: str,
          limit: int | None = None) -> dict:
    from eval.fixtures.video import read_video
    from eval.metrics import si_sdr, si_sdri

    dirs = fixture_dirs(set_name)[:limit]
    fixtures = []
    for d in dirs:
        m = json.loads((d / "manifest.json").read_text())
        mix, _ = io.load(d / "mix.wav", io.PIPELINE_SR)
        target, _ = io.load(d / "target.wav", io.PIPELINE_SR)
        n = min(len(mix), len(target))
        box = target_box(m)
        frames, fps = read_video(d / (m.get("video") or "video.mp4"))
        lips = lip_frames(frames, fps, box, lips_mode)
        fixtures.append({"id": d.name, "layout": m.get("layout", "?"), "mix": mix[:n],
                         "target": target[:n], "box": box, "lips": lips,
                         "lips_found": float(np.mean([c is not None for _, c in lips]))})
    rows = []
    for name in candidates:
        resolved = resolve_candidate(name)
        row = {"candidate": resolved, "requested": name, "window_s": window_s, "hop_s": hop_s,
               "lips": lips_mode, "error": None, "per_fixture": []}
        try:
            stage = FocusHQ(candidate=resolved, window_s=window_s, hop_s=hop_s, threaded=False)
            stage.load()
            if stage.error:
                raise RuntimeError(stage.error)
        except Exception as e:  # noqa: BLE001  (a broken candidate is a row, not a crash)
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        row["delay_ms"] = 1000.0 * (stage.window + stage.hop) / io.PIPELINE_SR  # live, threaded
        row["load_ms"] = stage.load_ms
        times, skipped, total, reasons = [], 0, 0, {}
        for fx in fixtures:
            t0 = time.perf_counter()
            y, windows = run_fixture(stage, fx["mix"], fx["lips"], fx["box"])
            wall = time.perf_counter() - t0
            times += [w.ms for w in windows if math.isfinite(w.ms)]
            skipped += sum(w.status != "ok" for w in windows)
            for w in windows:
                if w.status != "ok":
                    reasons[w.reason or w.status] = reasons.get(w.reason or w.status, 0) + 1
            total += len(windows)
            row["per_fixture"].append({
                "id": fx["id"], "layout": fx["layout"], "lips_found": fx["lips_found"],
                "si_sdri_db": si_sdri(y, fx["mix"], fx["target"]), "si_sdr_db": si_sdr(y, fx["target"]),
                "mix_si_sdr_db": si_sdr(fx["mix"], fx["target"]), "wall_s": wall,
            })
        row["device"] = stage.device
        stage.close()
        p50, p95 = np.percentile(times, [50, 95]).tolist() if times else (math.nan, math.nan)
        row.update(compute_p50_ms=p50, compute_p95_ms=p95, rtf=p50 / (1000.0 * hop_s),
                   realtime=bool(p95 < 1000.0 * hop_s), windows=total, skipped=skipped,
                   skip_reasons=reasons)
        rows.append(row)
    return {"set": set_name, "n_fixtures": len(fixtures), "rows": rows}


def _mean(items: list[dict], key: str, layout: str | None = None) -> float:
    vals = [i[key] for i in items if layout is None or i["layout"] == layout]
    return float(np.mean(vals)) if vals else math.nan


def format_table(report: dict) -> str:
    head = ("candidate", "layout", "n", "SI-SDRi_dB", "SI-SDR_dB", "comp_p50_ms", "comp_p95_ms",
            "RTF", "delay_ms", "skipped", "device", "error")
    lines = [" | ".join(head)]
    for r in report["rows"]:
        per = r["per_fixture"]
        layouts = [None, *sorted({p["layout"] for p in per})]
        for lay in layouts:
            n = sum(1 for p in per if lay is None or p["layout"] == lay)
            cells = [r["candidate"], lay or "all", str(n), f"{_mean(per, 'si_sdri_db', lay):.2f}",
                     f"{_mean(per, 'si_sdr_db', lay):.2f}"]
            if lay is None:
                cells += [f"{r.get('compute_p50_ms', math.nan):.1f}", f"{r.get('compute_p95_ms', math.nan):.1f}",
                          f"{r.get('rtf', math.nan):.3f}", f"{r.get('delay_ms', math.nan):.0f}",
                          f"{r.get('skipped', 0)}/{r.get('windows', 0)}", str(r.get("device")),
                          r["error"] or ""]
            else:
                cells += [""] * 7
            lines.append(" | ".join(cells))
    for r in report["rows"]:
        if r.get("skip_reasons"):
            why = ", ".join(f"{k}: {v}" for k, v in sorted(r["skip_reasons"].items()))
            lines.append(f"skipped windows, {r['candidate']}: {why}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="set_name", default="av_tiny", help="AV set under eval/fixtures/generated or a path")
    ap.add_argument("--candidates", default="auto,identity", help="comma-separated (auto = the Focus HQ pick)")
    ap.add_argument("--window", type=float, default=1.0, help="window seconds")
    ap.add_argument("--hop", type=float, default=0.5, help="hop seconds")
    ap.add_argument("--lips", choices=("box", "mediapipe"), default="box")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    report = bench([c.strip() for c in args.candidates.split(",") if c.strip()], args.set_name,
                   args.window, args.hop, args.lips, args.limit)
    from hearing import device as hdevice

    report["machine"] = hdevice.describe()
    report["stamp"] = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    print(format_table(report))
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"focus_hq_{report['stamp']}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {path}  (host {report['machine'].get('hostname')}, device {hdevice.pick()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
