"""Real-room clips: a phone video of two people → a replayable clip + manifest (T-090).

    .venv/bin/python scripts/ingest_clip.py phone.mp4 --target left [--name kitchen]
    .venv/bin/python -m eval.real_clip --synth /tmp/two-talkers.mp4   # stand-in clip from av_full

``ingest`` normalises any video to the replay format (640x360 @ 25 fps H.264, mono 16 kHz AAC,
speech level matched to the GRID demo clip), runs the vision stack once offline (face detector →
IoU tracker → ASD, the same classes the server's ``VisionEngine`` uses) and writes
``eval/fixtures/demo/<name>/{clip.mp4, manifest.json}``. The manifest's per-face speaking timeline
is the stand-in for clean stems: ``eval.session_nr`` scores a recorded session against it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from eval.fixtures import video
from hearing.audio import io as aio

SR = aio.PIPELINE_SR
FPS = 25
SIZE = (640, 360)
DEMO_DIR = Path(__file__).resolve().parent / "fixtures" / "demo"
# Active speech level of eval/fixtures/demo/grid-two-face.mp4 (mean power of the louder half of its
# 20 ms frames), measured on laptop-a with speech_level_db(): -18.5 dBFS. Real clips are matched to it
# so the slider and the gate thresholds see the same input level as on the GRID clip.
GRID_LEVEL_DB = -18.5
PEAK_DB = -1.0  # never clip: the gain is lowered so the peak stays under -1 dBFS
P_SPEAKING = 0.5  # ASD p_speaking >= 0.5 → speaking (the gate's threshold)
MIN_SEGMENT_S = 0.2  # shorter speaking runs are dropped, shorter pauses are bridged
MIN_FACE_SHARE = 0.2  # a track seen in < 20 % of frames is a detector glitch, not a person
FRAME_S = 0.02  # scoring frame (20 ms)


# --- audio ----------------------------------------------------------------------------------


def speech_level_db(x: np.ndarray, sr: int = SR) -> float:
    """Mean power of the louder half of the 20 ms frames, in dBFS (a cheap active-speech level)."""
    hop = int(sr * FRAME_S)
    n = len(x) // hop
    if n == 0:
        return -120.0
    p = np.mean(np.square(x[: n * hop].reshape(n, hop), dtype=np.float64), axis=1)
    return float(10 * np.log10(p[p >= np.median(p)].mean() + 1e-12))


def normalise_level(x: np.ndarray, target_db: float = GRID_LEVEL_DB) -> tuple[np.ndarray, dict]:
    level = speech_level_db(x)
    gain_db = target_db - level
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0:
        gain_db = min(gain_db, PEAK_DB - 20 * np.log10(peak))
    y = (x * 10 ** (gain_db / 20)).astype(np.float32)
    return y, {"input_db": round(level, 2), "target_db": target_db, "gain_db": round(float(gain_db), 2),
               "output_db": round(speech_level_db(y), 2)}


# --- video ----------------------------------------------------------------------------------


def transcode(src: Path, audio: np.ndarray, out: Path, max_s: float | None = None) -> Path:
    """``src`` video (autorotated, fitted and padded to 640x360, 25 fps) + ``audio`` → H.264/AAC MP4."""
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = SIZE
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        aio.save(wav, audio, SR)
        cmd = [video.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-i", str(wav),
               "-map", "0:v:0", "-map", "1:a:0",
               "-vf", (f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=decrease,"
                       f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "veryfast",
               "-c:a", "aac", "-b:a", "128k", "-ac", "1", "-ar", str(SR), "-shortest",
               "-movflags", "+faststart", "-map_metadata", "-1"]
        if max_s:
            cmd += ["-t", f"{max_s:g}"]
        subprocess.run([*cmd, str(out)], check=True, capture_output=True)
    return out


# --- vision ---------------------------------------------------------------------------------


def analyse(frames: np.ndarray, audio: np.ndarray, detector: str = "auto", asd: str = "auto") -> dict:
    """Run detector → IoU tracker → ASD over every frame in order (offline, no frame drops).

    Returns ``{"detector", "asd", "tracks": {id: {"frames": [...], "boxes": [...], "p": [...]}}}``.
    """
    from hearing.server.vision import get_face_detector
    from hearing.types import AudioChunk, VideoFrame
    from hearing.vision.asd import get_asd
    from hearing.vision.tracker import IouTracker

    det = get_face_detector(detector)
    if det is None:
        raise RuntimeError("no face detector: install mediapipe weights (python -m hearing.vision.faces --download)")
    try:
        speaker = get_asd(asd)
    except (ImportError, KeyError, OSError, RuntimeError, ValueError):
        speaker = get_asd("lipmotion")
    tracker = IouTracker()
    tracks: dict[int, dict] = {}
    hop = SR // FPS
    try:
        for i, image in enumerate(frames):
            ts = i / FPS
            chunk = audio[max(0, (i - 1) * hop): i * hop] if i else audio[:0]
            found = tracker.update(det.detect(image, ts), ts)
            speaker.update(found, ts, AudioChunk(chunk.astype(np.float32), SR, i, ts) if len(chunk) else None,
                           VideoFrame(image, i, ts))
            for t in found:
                if t.last_seen_ts != ts:
                    continue  # coasting, not seen in this frame
                row = tracks.setdefault(t.track_id, {"frames": [], "boxes": [], "p": []})
                row["frames"].append(i)
                row["boxes"].append([round(float(v), 4) for v in t.box])
                row["p"].append(round(float(t.p_speaking), 3))
    finally:
        det.close()
    return {"detector": getattr(det, "name", detector), "asd": getattr(speaker, "name", asd), "tracks": tracks}


def segments(active: np.ndarray, rate: float = FPS, min_s: float = MIN_SEGMENT_S) -> list[list[float]]:
    """Boolean per-frame activity → ``[[start_s, end_s], ...]``: pauses < ``min_s`` bridged, runs < ``min_s`` dropped."""
    a = np.asarray(active, dtype=bool).copy()
    gap = max(1, round(min_s * rate))
    edges = np.flatnonzero(np.diff(np.r_[0, a.astype(int), 0]))
    starts, ends = edges[::2], edges[1::2]
    for s, e in zip(ends[:-1], starts[1:], strict=True):
        if e - s < gap:
            a[s:e] = True
    edges = np.flatnonzero(np.diff(np.r_[0, a.astype(int), 0]))
    return [[round(s / rate, 2), round(e / rate, 2)] for s, e in zip(edges[::2], edges[1::2], strict=True)
            if e - s >= gap]


def speaking_timeline(frames: list[int], p: list[float], n_frames: int) -> list[list[float]]:
    full = np.zeros(n_frames)
    full[np.asarray(frames, dtype=int)] = p
    if n_frames >= 5:  # 5-frame (200 ms) median: one noisy ASD frame is not a turn
        pad = np.pad(full, 2, mode="edge")
        full = np.median(np.lib.stride_tricks.sliding_window_view(pad, 5), axis=1)
    return segments(full >= P_SPEAKING)


def pick_target(faces: list[dict], rule: str) -> dict:
    """``left`` / ``right`` (by mean box centre), ``largest`` (mean box area) or ``track:<id>``."""
    if not faces:
        raise ValueError("no face was seen in enough frames to be a target")
    if rule.startswith("track:"):
        tid = int(rule.split(":", 1)[1])
        hit = [f for f in faces if f["track_id"] == tid]
        if not hit:
            raise ValueError(f"no track {tid}; tracks: {[f['track_id'] for f in faces]}")
        return hit[0]
    if rule == "left":
        return min(faces, key=lambda f: f["center_x"])
    if rule == "right":
        return max(faces, key=lambda f: f["center_x"])
    if rule == "largest":
        return max(faces, key=lambda f: f["mean_box"][2] * f["mean_box"][3])
    raise ValueError(f"target must be left, right, largest or track:<id>, got {rule!r}")


def build_manifest(name: str, source: str, analysis: dict, n_frames: int, target_rule: str,
                   loudness: dict) -> dict:
    faces = []
    for tid, row in sorted(analysis["tracks"].items()):
        if len(row["frames"]) < MIN_FACE_SHARE * n_frames:
            continue
        boxes = np.asarray(row["boxes"])
        mean = [round(float(v), 4) for v in boxes.mean(axis=0)]
        timeline = speaking_timeline(row["frames"], row["p"], n_frames)
        faces.append({"track_id": int(tid), "center_x": round(mean[0] + mean[2] / 2, 4), "mean_box": mean,
                      "frames_seen": len(row["frames"]),
                      "speaking": timeline, "speaking_s": round(sum(e - s for s, e in timeline), 2)})
    order = sorted(faces, key=lambda f: f["center_x"])
    for i, f in enumerate(order):
        f["side"] = "left" if i == 0 and len(order) > 1 else "right" if i == len(order) - 1 and len(order) > 1 else "center"
    target = pick_target(faces, target_rule)
    return {
        "version": 1, "name": name, "source": source, "clip": "clip.mp4",
        "duration_s": round(n_frames / FPS, 2), "fps": FPS, "sample_rate": SR, "frame_size": list(SIZE),
        "loudness": loudness, "detector": analysis["detector"], "asd": analysis["asd"],
        "p_speaking_threshold": P_SPEAKING,
        "target": {"rule": target_rule, "track_id": target["track_id"], "side": target["side"],
                   "box": target["mean_box"]},
        "faces": faces,
        "boxes": {str(tid): [[i, *b] for i, b in zip(row["frames"], row["boxes"], strict=True)]
                  for tid, row in sorted(analysis["tracks"].items())
                  if any(f["track_id"] == tid for f in faces)},
    }


def dump_manifest(manifest: dict) -> str:
    """Pretty JSON, except the per-frame boxes: one line per face so a 30 s manifest stays small."""
    head = {k: v for k, v in manifest.items() if k != "boxes"}
    text = json.dumps(head, indent=2)[:-2]
    boxes = ",\n".join(f'    "{k}": {json.dumps(v, separators=(",", ":"))}' for k, v in manifest["boxes"].items())
    return text + ',\n  "boxes": {\n' + boxes + "\n  }\n}\n"


def ingest(src: Path, name: str | None = None, target: str = "largest", out_root: Path = DEMO_DIR,
           detector: str = "auto", asd: str = "auto", max_s: float | None = None,
           level_db: float = GRID_LEVEL_DB) -> Path:
    """``src`` video → ``out_root/<name>/{clip.mp4, manifest.json}``; returns the manifest path."""
    src = Path(src)
    if not video.available():
        raise RuntimeError("ffmpeg missing (pip install imageio-ffmpeg)")
    name = name or src.stem
    out = Path(out_root) / name
    audio = video.read_audio(src, SR)
    if not len(audio):
        raise ValueError(f"{src} has no audio track")
    if max_s:
        audio = audio[: int(max_s * SR)]
    audio, loudness = normalise_level(audio, level_db)
    clip = transcode(src, audio, out / "clip.mp4", max_s)
    frames, _ = video.read_video(clip, fps=FPS)
    audio = video.read_audio(clip, SR)  # what replay will send (AAC round trip)
    n = min(len(frames), len(audio) * FPS // SR)
    analysis = analyse(frames[:n], audio, detector=detector, asd=asd)
    manifest = build_manifest(name, src.name, analysis, n, target, loudness)
    path = out / "manifest.json"
    path.write_text(dump_manifest(manifest))
    return path


def load_manifest(path: Path) -> dict:
    """A clip directory, its ``manifest.json`` or its ``clip.mp4`` → the manifest dict (with ``_dir``)."""
    path = Path(path)
    if path.is_dir():
        path = path / "manifest.json"
    elif path.suffix.lower() == ".mp4":
        path = path.with_name("manifest.json")
    manifest = json.loads(path.read_text())
    manifest["_dir"] = str(path.parent)
    return manifest


# --- a local stand-in for a phone recording -------------------------------------------------


def synth_phone_clip(out: Path, generated: Path | None = None, each_s: float = 4.0,
                     size: tuple[int, int] = (1280, 720), fps: float = 30.0) -> dict:
    """Two av_full GRID talkers side by side, recorded like the DEMO.md recipe: left alone,
    right alone, then both, written at a phone-like 1280x720 @ 30 fps so ``ingest`` has to resample.

    Faces are looped (GRID utterances are 3 s); audio is each talker's clean ``target.wav`` (no
    noise), so the expected timeline is known: returns ``{"left": [[s, e]], "right": [[s, e]]}``.
    """
    from eval.make_fixtures import DEFAULT_OUT

    root = Path(generated or DEFAULT_OUT) / "av_full"
    picks = []
    for d in sorted(root.glob("*_single_*")):
        m = json.loads((d / "manifest.json").read_text())
        if (d / "video.mp4").exists() and m["source"]["target_speaker"] not in [p[1]["source"]["target_speaker"] for p in picks]:
            picks.append((d, m))
        if len(picks) == 2:
            break
    if len(picks) < 2:
        raise RuntimeError(f"need two av_full single-face fixtures with different speakers under {root}")
    n_audio = int(3 * each_s * SR)
    n_frames = round(3 * each_s * FPS)
    tracks, frames = [], []
    for j, (d, _) in enumerate(picks):
        x, _ = aio.load(d / "target.wav", SR)
        f, _ = video.read_video(d / "video.mp4", fps=FPS)
        shift = j * len(f) // 2  # the right talker's loop starts half an utterance later
        idx = (np.arange(n_frames) + shift) % len(f)
        sample = (np.arange(n_audio) + shift * SR // FPS) % len(x)
        tracks.append(x[sample].astype(np.float32))
        frames.append(f[idx])
    k, kf = int(each_s * SR), round(each_s * FPS)
    left, right = tracks
    # A listener does not move their lips: the silent talker's face is held on a still frame.
    left[k: 2 * k] = 0  # right talks alone in the middle third
    frames[0][kf: 2 * kf] = frames[0][kf]
    right[:k] = 0  # left talks alone first
    frames[1][:kf] = frames[1][0]
    mix = (0.5 * (left + right)).astype(np.float32)
    composite = video.side_by_side(frames[0], frames[1], size)
    if fps != FPS:
        composite = composite[video.resample_indices(n_frames, FPS, fps)]
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "mix.wav"
        aio.save(wav, mix, SR)
        video.write_video(out, composite, fps=fps, audio_wav=wav)
    return {"left": [[0.0, each_s], [2 * each_s, 3 * each_s]], "right": [[each_s, 3 * each_s]],
            "sources": [d.name for d, _ in picks]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synth", type=Path, help="write a two-talker stand-in phone clip from av_full here")
    args = ap.parse_args(argv)
    if args.synth:
        print(json.dumps(synth_phone_clip(args.synth)))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
