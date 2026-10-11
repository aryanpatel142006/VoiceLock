"""Video I/O and frame maths for the audio-visual fixture bank.

Frames are RGB ``uint8`` arrays shaped ``(N, H, W, 3)`` at 25 fps and at most 640x360
(``hearing-stack``). Decoding and encoding go through an ``ffmpeg`` subprocess with raw RGB
pipes: the binary from ``imageio-ffmpeg`` (``eval`` extra) when installed, else ``ffmpeg`` on
``PATH``. OpenCV is not needed (mediapipe already pins ``opencv-contrib-python``; adding the
headless wheel would fight it over the ``cv2`` module).

* :func:`read_video` decodes, fits inside ``max_size`` (never upscales) and resamples to ``fps``.
* :func:`write_video` encodes H.264 (yuv420p, even dimensions) and optionally muxes a WAV as AAC.
* :func:`read_audio` decodes a file's audio track to float32 mono at 16 kHz.
* :func:`resample_fps`, :func:`fit_size`, :func:`fit_resize`, :func:`resize_frames`,
  :func:`crop_to_aspect`, :func:`side_by_side` are pure numpy and used by the tests offline.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hearing.audio import io as aio

FPS = 25
MAX_SIZE = (640, 360)  # (width, height)


# --- ffmpeg ---------------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """Path to an ffmpeg binary: imageio-ffmpeg's bundled one, else ``ffmpeg`` on ``PATH``."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        pass
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg not found: install the eval extra (imageio-ffmpeg) or put ffmpeg on PATH")
    return exe


def available() -> bool:
    try:
        ffmpeg_exe()
    except RuntimeError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def _h264_encoder() -> str:
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True, text=True, check=False)
    return "libx264" if "libx264" in out.stdout else "mpeg4"


def _run(cmd: list[str], data: bytes | None = None) -> bytes:
    proc = subprocess.run(cmd, input=data, capture_output=True, check=False)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return proc.stdout


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    duration_s: float
    has_audio: bool


_VIDEO_RE = re.compile(r"Stream #\S+.*?Video: .*?(\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"(\d+(?:\.\d+)?) fps")
_TBR_RE = re.compile(r"(\d+(?:\.\d+)?) tbr")
_DUR_RE = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")


def probe(path: str | os.PathLike) -> VideoInfo:
    """Size, frame rate, duration and audio presence parsed from ``ffmpeg -i`` (no ffprobe needed)."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, check=False)
    err = proc.stderr
    vline = next((ln for ln in err.splitlines() if "Video:" in ln), "")
    m = _VIDEO_RE.search(vline)
    if not m:
        raise RuntimeError(f"no video stream in {path}:\n{err[-800:]}")
    fm = _FPS_RE.search(vline) or _TBR_RE.search(vline)
    dm = _DUR_RE.search(err)
    dur = int(dm[1]) * 3600 + int(dm[2]) * 60 + float(dm[3]) if dm else 0.0
    return VideoInfo(
        width=int(m[1]),
        height=int(m[2]),
        fps=float(fm[1]) if fm else float(FPS),
        duration_s=dur,
        has_audio=any("Audio:" in ln for ln in err.splitlines() if "Stream #" in ln),
    )


# --- pure frame maths -----------------------------------------------------------------------


def fit_size(width: int, height: int, max_size: tuple[int, int] = MAX_SIZE) -> tuple[int, int]:
    """Largest even ``(w, h)`` with the input aspect inside ``max_size``; never upscales."""
    mw, mh = max_size
    scale = min(1.0, mw / width, mh / height)
    w = max(2, int(width * scale) // 2 * 2)
    h = max(2, int(height * scale) // 2 * 2)
    return w, h


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of ``(N, H, W, 3)`` uint8 frames to ``size = (w, h)``; area-averaged when shrinking."""
    frames = np.asarray(frames)
    w, h = size
    n, H, W = frames.shape[:3]
    if (W, H) == (w, h):
        return frames.copy()
    x = frames.astype(np.float32)
    # Box-filter down by an integer factor first so large shrinks do not alias.
    fy, fx = max(1, H // h), max(1, W // w)
    if fy > 1 or fx > 1:
        Hc, Wc = H // fy * fy, W // fx * fx
        x = x[:, :Hc, :Wc].reshape(n, Hc // fy, fy, Wc // fx, fx, x.shape[-1]).mean(axis=(2, 4))
        H, W = x.shape[1:3]
    ys = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, H - 1), np.minimum(x0 + 1, W - 1)
    wy = (ys - y0)[None, :, None, None]
    wx = (xs - x0)[None, None, :, None]
    top = x[:, y0][:, :, x0] * (1 - wx) + x[:, y0][:, :, x1] * wx
    bot = x[:, y1][:, :, x0] * (1 - wx) + x[:, y1][:, :, x1] * wx
    return np.clip(np.rint(top * (1 - wy) + bot * wy), 0, 255).astype(np.uint8)


def fit_resize(frames: np.ndarray, max_size: tuple[int, int] = MAX_SIZE) -> np.ndarray:
    """Shrink frames (keeping aspect, even dims) to fit inside ``max_size``; never upscales."""
    frames = np.asarray(frames)
    return resize_frames(frames, fit_size(frames.shape[2], frames.shape[1], max_size))


def resample_indices(n: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """Source frame index for each output frame (nearest-earlier frame at each output timestamp)."""
    if n <= 0:
        return np.zeros(0, dtype=int)
    n_out = max(1, round(n * dst_fps / src_fps))
    idx = np.floor(np.arange(n_out) * (src_fps / dst_fps) + 1e-6).astype(int)
    return np.minimum(idx, n - 1)


def resample_fps(frames: np.ndarray, src_fps: float, dst_fps: float = FPS) -> np.ndarray:
    """Drop or repeat frames so ``src_fps`` becomes ``dst_fps`` (30 -> 25 keeps 5 of every 6)."""
    frames = np.asarray(frames)
    if abs(src_fps - dst_fps) < 1e-6:
        return frames
    return frames[resample_indices(len(frames), src_fps, dst_fps)]


def crop_box_to_aspect(width: int, height: int, aspect: float) -> tuple[int, int, int, int]:
    """Centred ``(x, y, w, h)`` crop of a ``width x height`` frame with ``w / h == aspect``."""
    if width / height > aspect:
        w = round(height * aspect)
        return (width - w) // 2, 0, w, height
    h = round(width / aspect)
    return 0, (height - h) // 2, width, h


def crop_to_aspect(frames: np.ndarray, aspect: float) -> np.ndarray:
    frames = np.asarray(frames)
    x, y, w, h = crop_box_to_aspect(frames.shape[2], frames.shape[1], aspect)
    return frames[:, y : y + h, x : x + w]


def fit_frames(frames: np.ndarray, n: int) -> np.ndarray:
    """Trim to ``n`` frames or loop (tile) until ``n`` long: the video twin of ``mix.fit_length``."""
    frames = np.asarray(frames)
    if len(frames) == 0:
        raise ValueError("cannot fit an empty clip")
    if len(frames) >= n:
        return frames[:n]
    reps = -(-n // len(frames))
    return np.concatenate([frames] * reps)[:n]


def side_by_side(left: np.ndarray, right: np.ndarray, size: tuple[int, int] = MAX_SIZE) -> np.ndarray:
    """Composite two clips into one ``size`` frame: each centre-cropped to a half's aspect and resized.

    Both clips must already have the same frame count (use :func:`fit_frames`).
    """
    if len(left) != len(right):
        raise ValueError(f"clips differ in length: {len(left)} vs {len(right)}")
    w, h = size
    half = (w // 2, h)
    aspect = half[0] / half[1]
    a = resize_frames(crop_to_aspect(left, aspect), half)
    b = resize_frames(crop_to_aspect(right, aspect), half)
    return np.concatenate([a, b], axis=2)


def map_box(
    box: tuple[float, float, float, float],
    src_size: tuple[int, int],
    crop: tuple[int, int, int, int],
    dst_rect: tuple[int, int, int, int],
    frame_size: tuple[int, int],
) -> list[float]:
    """Carry a normalised ``(x, y, w, h)`` box through a crop and a resize into a composite frame.

    ``src_size`` is the source ``(w, h)``, ``crop`` the ``(x, y, w, h)`` pixel crop taken from it,
    ``dst_rect`` where that crop lands in the output frame of ``frame_size``. The result is
    normalised to the output frame and clipped to the destination rectangle.
    """
    sw, sh = src_size
    cx, cy, cw, ch = crop
    dx, dy, dw, dh = dst_rect
    fw, fh = frame_size
    x0 = dx + (box[0] * sw - cx) * dw / cw
    y0 = dy + (box[1] * sh - cy) * dh / ch
    x1 = dx + ((box[0] + box[2]) * sw - cx) * dw / cw
    y1 = dy + ((box[1] + box[3]) * sh - cy) * dh / ch
    x0, x1 = np.clip([x0, x1], dx, dx + dw)
    y0, y1 = np.clip([y0, y1], dy, dy + dh)
    return [round(float(x0) / fw, 4), round(float(y0) / fh, 4), round(float(x1 - x0) / fw, 4), round(float(y1 - y0) / fh, 4)]


def side_by_side_boxes(
    left_box: tuple[float, float, float, float],
    left_size: tuple[int, int],
    right_box: tuple[float, float, float, float],
    right_size: tuple[int, int],
    size: tuple[int, int] = MAX_SIZE,
) -> tuple[list[float], list[float]]:
    """Where each clip's face box ends up in the :func:`side_by_side` composite (normalised)."""
    w, h = size
    aspect = (w // 2) / h
    out = []
    for box, src, dx in ((left_box, left_size, 0), (right_box, right_size, w // 2)):
        crop = crop_box_to_aspect(src[0], src[1], aspect)
        out.append(map_box(box, src, crop, (dx, 0, w // 2, h), size))
    return out[0], out[1]


# --- file I/O -------------------------------------------------------------------------------


def read_video(
    path: str | os.PathLike, fps: float = FPS, max_size: tuple[int, int] | None = MAX_SIZE,
    duration_s: float | None = None,
) -> tuple[np.ndarray, float]:
    """Decode ``path`` to RGB uint8 ``(N, H, W, 3)`` frames at ``fps``, fitted inside ``max_size``.

    ``duration_s`` decodes only the first that many seconds (a long take, bounded memory).

    ffmpeg decodes at the native rate (scaling down in the decoder when needed); the frame-rate
    change is :func:`resample_fps`, so tests can check it without a file.
    """
    info = probe(path)
    w, h = fit_size(info.width, info.height, max_size) if max_size else (info.width, info.height)
    vf = ["-vf", f"scale={w}:{h}:flags=area"] if (w, h) != (info.width, info.height) else []
    raw = _run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path), *vf,
         *(["-t", f"{duration_s:.3f}"] if duration_s else []),
         "-vsync", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    )
    frames = np.frombuffer(raw, dtype=np.uint8)
    n = frames.size // (w * h * 3)
    frames = frames[: n * w * h * 3].reshape(n, h, w, 3)
    if fps and abs(info.fps - fps) > 1e-3:
        frames = resample_fps(frames, info.fps, fps)
    else:
        fps = info.fps
    return np.ascontiguousarray(frames), float(fps)


def write_video(
    path: str | os.PathLike,
    frames: np.ndarray,
    fps: float = FPS,
    audio_wav: str | os.PathLike | None = None,
    crf: int = 23,
) -> Path:
    """Encode RGB uint8 frames as H.264/yuv420p MP4, muxing ``audio_wav`` as AAC when given."""
    frames = np.ascontiguousarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"frames must be non-empty (N, H, W, 3) uint8, got {frames.shape}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _, h, w, _ = frames.shape
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "-",
    ]
    if audio_wav is not None:
        cmd += ["-i", str(audio_wav)]
    enc = _h264_encoder()
    cmd += ["-map", "0:v:0", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", enc, "-pix_fmt", "yuv420p"]
    cmd += ["-crf", str(crf), "-preset", "veryfast"] if enc == "libx264" else ["-q:v", "4"]
    if audio_wav is not None:
        cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k"]
    cmd += ["-movflags", "+faststart", "-map_metadata", "-1", str(out)]
    _run(cmd, frames.tobytes())
    return out


def read_audio(path: str | os.PathLike, sr: int = aio.PIPELINE_SR) -> np.ndarray:
    """Decode the first audio track of any media file to float32 mono at ``sr``."""
    raw = _run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    )
    return np.frombuffer(raw, dtype="<f4").astype(np.float32)


def mux_audio(video_in: str | os.PathLike, audio_wav: str | os.PathLike, out: str | os.PathLike) -> Path:
    """Copy the video stream of ``video_in`` and add ``audio_wav`` as AAC (no re-encode of video)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(video_in), "-i", str(audio_wav),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart", "-map_metadata", "-1", str(out)]
    )
    return out
