"""Light-ASD active-speaker detector (Liao et al., CVPR 2023; MIT, ~1.0 M params).

Implements :class:`hearing.vision.asd.SpeakerDetector` as ``get_asd("light_asd")``.
The network (visual encoder, audio encoder, bidirectional-GRU head and the AV/V
classifier layers) is vendored below from github.com/Junhua-Liao/Light-ASD at
``REVISION``; parameter names match the released state dict.

Per video frame and track:

* a 112×112 grayscale crop is cut from the frame (:func:`crop_geometry`): a
  square of side ``crop_scale`` × face size centred on the mouth landmarks, or,
  without landmarks, at ``mouth_offset`` × face size below the face-box centre
  (the lower half of the box). With the defaults this reproduces Light-ASD's own
  preprocessing (``Columbia_test.py``: face box +40 %, resized to 224, centre
  112 → side 0.7 × face, centred 0.2 × face below the box centre). Pixels outside
  the frame read as 110, the upstream padding value;
* the last ``window`` crops (default 12 = 0.48 s, the T-051 operating point; upstream
  scores whole tracks) and the matching last
  ``4 × window`` 13-dim MFCC frames (100 fps, ``python_speech_features`` exactly as
  upstream, computed incrementally) are scored; ``logit = logit_speaking -
  logit_silent`` of the last frame and ``p = sigmoid(scale * logit + bias)`` with the
  backend's ``calibration`` (:data:`TORCH_CALIBRATION` at the nearest fitted window for the
  real network, identity for mocks), then an optional EMA with time constant ``ema_s`` (off by default: on
  ``av_full`` any smoothing lag cost more than it removed in flicker).

Until audio arrives, the visual-only head (``lossV.FC``) is used. Without a
frame (``frame=None``) or with a stale track, ``p_speaking`` holds its last value.

Weights: call :func:`download_model` once (pinned URL and SHA-256); processing
never downloads. ``available()`` says whether weights and torch are present.
``backend="auto"`` (the default) uses torch when available, else the
deterministic :class:`MockBackend` (score from crop motion, no torch);
``backend="torch"`` raises ``RuntimeError`` if unavailable. ``get_asd("auto")``
picks Light-ASD only when ``available()``.

``python -m hearing.vision.light_asd [--download]`` prints device, weights path
and per-frame compute for two tracks on each device.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from hearing import device, models
from hearing.types import AudioChunk, FaceTrack, VideoFrame
from hearing.vision.asd import MP_LOWER_INNER_LIP, MP_UPPER_INNER_LIP, register_asd

REVISION = "ed38c232de5efe0261dbd68627c0ade7cdfe14eb"
# finetuning_TalkSet: the upstream checkpoint for in-the-wild video (Columbia demo).
MODEL_URL = (
    f"https://raw.githubusercontent.com/Junhua-Liao/Light-ASD/{REVISION}/"
    "weight/finetuning_TalkSet.model"
)
MODEL_SHA256 = "efc375833887eefa9d209dc92810e18519b04c3c73ea35a549f2a7f40b7d94d5"
CROP = 112
SR = 16000
FPS = 25
MFCC_PER_FRAME = 4  # 100 fps MFCC / 25 fps video.
MFCC_HOP = 160  # 10 ms.
MFCC_WIN = 400  # 25 ms.
PREEMPH = 0.97
PAD_VALUE = 110.0
# MediaPipe FaceMesh mouth corners; used with the inner-lip points when present.
MP_MOUTH_LEFT = 61
MP_MOUTH_RIGHT = 291
MOUTH_POINTS = (MP_UPPER_INNER_LIP, MP_LOWER_INNER_LIP, MP_MOUTH_LEFT, MP_MOUTH_RIGHT)
# Platt scaling of the released checkpoint's last-frame logits, p = sigmoid(scale * logit + bias), per
# window length (frames), fitted on av_full (GRID two-face, 80 clips, 12 000 face-frames; T-051) with
# `eval.asd_eval --fit`. Raw logits sit around +5 even on silent faces (the uncalibrated sigmoid said
# "speaking" 80 % of the time at window 25); the fit moves the decision point up and keeps p = 0.5 at
# even odds. Leave-one-pair-out frame accuracy in the comments; DEFAULT_WINDOW is the best.
TORCH_CALIBRATION: dict[int, tuple[float, float]] = {
    6: (1.377, -1.714),  # 0.853
    8: (1.296, -1.703),  # 0.873
    12: (0.937, -1.826),  # 0.905
    16: (0.744, -1.924),  # 0.883
    20: (0.663, -2.060),  # 0.855
    25: (0.619, -2.244),  # 0.818
    50: (0.425, -2.004),  # 0.715
}
DEFAULT_WINDOW = 12


def calibration_for(table: dict[int, tuple[float, float]], window: int) -> tuple[float, float]:
    """The ``(scale, bias)`` fitted at the window closest to ``window`` (ties: the shorter)."""
    return table[min(table, key=lambda w: (abs(w - window), w))]


# --------------------------------------------------------------------- weights
def model_path() -> Path:
    return models.cache_dir() / "light_asd" / "finetuning_TalkSet.model"


def _verify(data: bytes) -> None:
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("Light-ASD checkpoint checksum mismatch; re-provision the pinned weights")


def download_model() -> Path:
    """Explicit network adapter; pinned revision/checksum and atomic cache install."""
    path = model_path()
    if path.exists():
        _verify(path.read_bytes())
        return path
    import httpx

    response = httpx.get(MODEL_URL, timeout=60, follow_redirects=True)
    response.raise_for_status()
    data = response.content
    _verify(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(data)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def available() -> bool:
    """Weights on disk and torch + python_speech_features importable (no checksum, no import)."""
    return (
        model_path().exists()
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("python_speech_features") is not None
    )


# --------------------------------------------------------------------- crops
def crop_geometry(
    box: tuple[float, float, float, float],
    landmarks: np.ndarray | None,
    width: int,
    height: int,
    crop_scale: float = 0.7,
    mouth_offset: float = 0.2,
) -> tuple[float, float, float, bool]:
    """``(cx, cy, side, used_landmarks)`` in pixels for one track's lip crop.

    ``box`` is normalised ``x, y, w, h``. Landmarks whose x/y all lie in
    [-0.5, 1.5] are taken as normalised, otherwise as pixels.
    """
    bx, by, bw, bh = (float(v) for v in box)
    face = max(bw * width, bh * height)
    side = max(crop_scale * face, 1.0)
    if landmarks is not None:
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.ndim == 2 and lm.shape[1] >= 2 and lm.shape[0] > MP_LOWER_INNER_LIP:
            idx = [i for i in MOUTH_POINTS if i < lm.shape[0]]
            pts = lm[idx, :2]
            if np.isfinite(pts).all():
                if np.abs(lm[:, :2] - 0.5).max() <= 1.0:
                    pts = pts * (width, height)
                cx, cy = pts.mean(axis=0)
                return float(cx), float(cy), side, True
    cx = (bx + bw / 2) * width
    cy = (by + bh / 2) * height + mouth_offset * face
    return cx, cy, side, False


def _bilinear_sample(gray: np.ndarray, x0: float, y0: float, step: float, size: int) -> np.ndarray:
    """Sample ``size×size`` points at ``x0 + (i + 0.5) * step - 0.5`` (and y), padding with 110."""
    h, w = gray.shape
    grid = (np.arange(size) + 0.5) * step - 0.5
    xs, ys = x0 + grid, y0 + grid
    xf, yf = np.floor(xs), np.floor(ys)
    ax, ay = (xs - xf)[None, :], (ys - yf)[:, None]
    xi, yi = xf.astype(np.int64), yf.astype(np.int64)
    padded = np.pad(gray, 1, constant_values=PAD_VALUE)  # Edge neighbours fall into the pad.
    xc = np.clip(xi + 1, 0, w + 1)
    yc = np.clip(yi + 1, 0, h + 1)
    xc1 = np.clip(xi + 2, 0, w + 1)
    yc1 = np.clip(yi + 2, 0, h + 1)
    out = (
        padded[yc[:, None], xc[None, :]] * (1 - ax) * (1 - ay)
        + padded[yc[:, None], xc1[None, :]] * ax * (1 - ay)
        + padded[yc1[:, None], xc[None, :]] * (1 - ax) * ay
        + padded[yc1[:, None], xc1[None, :]] * ax * ay
    )
    outside = ((xs < -1) | (xs > w))[None, :] | ((ys < -1) | (ys > h))[:, None]
    out[outside] = PAD_VALUE
    return out


def cut_crop(image: np.ndarray, cx: float, cy: float, side: float, size: int = CROP) -> np.ndarray:
    """Grayscale ``size×size`` uint8 crop of an RGB uint8 frame, bilinear, padded with 110."""
    h, w = image.shape[:2]
    left, top = cx - side / 2, cy - side / 2
    # Convert only the region the crop touches (+1 px for interpolation).
    x1, y1 = max(math.floor(left) - 1, 0), max(math.floor(top) - 1, 0)
    x2, y2 = min(math.ceil(left + side) + 2, w), min(math.ceil(top + side) + 2, h)
    if x2 <= x1 or y2 <= y1:
        return np.full((size, size), PAD_VALUE, np.uint8)
    rgb = image[y1:y2, x1:x2, :3].astype(np.float32)
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)  # == cv2 RGB2GRAY weights.
    out = _bilinear_sample(gray, left - x1, top - y1, side / size, size)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------- MFCC
class StreamingMFCC:
    """13-dim MFCC at 100 fps, chunk-invariant and equal to upstream's whole-clip call.

    Upstream: ``python_speech_features.mfcc(int16_audio, 16000, numcep=13,
    winlen=0.025, winstep=0.010)``. Pre-emphasis runs here with carried state, so
    each call hands whole, independent 25 ms frames to the library with
    ``preemph=0``.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, np.float64)
        self._last: float | None = None  # Last raw sample, for pre-emphasis.

    def push(self, samples: np.ndarray) -> np.ndarray:
        """Feed float32 [-1, 1] samples at 16 kHz; returns new ``(n, 13)`` frames."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1) * 32768.0
        if x.size == 0:
            return np.zeros((0, 13), np.float32)
        prev = np.concatenate(([x[0] if self._last is None else self._last], x[:-1]))
        emph = x - PREEMPH * prev
        if self._last is None:
            emph[0] = x[0]
        self._last = float(x[-1])
        buf = np.concatenate((self._buf, emph))
        if buf.size < MFCC_WIN:
            self._buf = buf
            return np.zeros((0, 13), np.float32)
        n = 1 + (buf.size - MFCC_WIN) // MFCC_HOP
        import python_speech_features as psf

        used = (n - 1) * MFCC_HOP + MFCC_WIN
        feats = psf.mfcc(buf[:used], SR, winlen=0.025, winstep=0.010, numcep=13, preemph=0.0)
        self._buf = buf[n * MFCC_HOP :].copy()
        return feats.astype(np.float32)


# --------------------------------------------------------------------- backends
class Backend(Protocol):
    device: str
    needs_audio: bool

    def scores(self, video: np.ndarray, mfcc: np.ndarray | None) -> np.ndarray:
        """``video`` (B, T, 112, 112) uint8, ``mfcc`` (4T, 13) float32 or None → (B,) last-frame logits."""
        ...


class MockBackend:
    """Deterministic, torch-free: logit = (mean |crop_t − crop_{t−1}| − threshold) × gain."""

    device = "mock"
    needs_audio = False

    def __init__(self, threshold: float = 2.0, gain: float = 1.0) -> None:
        self.threshold = threshold
        self.gain = gain

    def scores(self, video: np.ndarray, mfcc: np.ndarray | None) -> np.ndarray:
        v = np.asarray(video, dtype=np.float32)
        if v.shape[1] < 2:
            change = np.zeros(v.shape[0], np.float32)
        else:
            change = np.abs(np.diff(v, axis=1)).mean(axis=(1, 2, 3))
        return (change - self.threshold) * self.gain


def _build_net():
    """The vendored Light-ASD network (needs torch)."""
    import torch
    from torch import nn

    def bn2(c):
        return nn.BatchNorm2d(c, momentum=0.01, eps=0.001)

    def bn3(c):
        return nn.BatchNorm3d(c, momentum=0.01, eps=0.001)

    class AudioBlock(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.relu = nn.ReLU()
            self.m_3 = nn.Conv2d(cin, cout, (3, 1), padding=(1, 0), bias=False)
            self.bn_m_3 = bn2(cout)
            self.t_3 = nn.Conv2d(cout, cout, (1, 3), padding=(0, 1), bias=False)
            self.bn_t_3 = bn2(cout)
            self.m_5 = nn.Conv2d(cin, cout, (5, 1), padding=(2, 0), bias=False)
            self.bn_m_5 = bn2(cout)
            self.t_5 = nn.Conv2d(cout, cout, (1, 5), padding=(0, 2), bias=False)
            self.bn_t_5 = bn2(cout)
            self.last = nn.Conv2d(cout, cout, (1, 1), bias=False)
            self.bn_last = bn2(cout)

        def forward(self, x):
            x3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_m_3(self.m_3(x))))))
            x5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_m_5(self.m_5(x))))))
            return self.relu(self.bn_last(self.last(x3 + x5)))

    class VisualBlock(nn.Module):
        def __init__(self, cin, cout, is_down=False):
            super().__init__()
            stride = (1, 2, 2) if is_down else (1, 1, 1)
            self.relu = nn.ReLU()
            self.s_3 = nn.Conv3d(cin, cout, (1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
            self.bn_s_3 = bn3(cout)
            self.t_3 = nn.Conv3d(cout, cout, (3, 1, 1), padding=(1, 0, 0), bias=False)
            self.bn_t_3 = bn3(cout)
            self.s_5 = nn.Conv3d(cin, cout, (1, 5, 5), stride=stride, padding=(0, 2, 2), bias=False)
            self.bn_s_5 = bn3(cout)
            self.t_5 = nn.Conv3d(cout, cout, (5, 1, 1), padding=(2, 0, 0), bias=False)
            self.bn_t_5 = bn3(cout)
            self.last = nn.Conv3d(cout, cout, (1, 1, 1), bias=False)
            self.bn_last = bn3(cout)

        def forward(self, x):
            x3 = self.relu(self.bn_t_3(self.t_3(self.relu(self.bn_s_3(self.s_3(x))))))
            x5 = self.relu(self.bn_t_5(self.t_5(self.relu(self.bn_s_5(self.s_5(x))))))
            return self.relu(self.bn_last(self.last(x3 + x5)))

    class VisualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = VisualBlock(1, 32, is_down=True)
            self.pool1 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block2 = VisualBlock(32, 64)
            self.pool2 = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
            self.block3 = VisualBlock(64, 128)

        def forward(self, x):  # (B, 1, T, 112, 112) → (B, T, 128)
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            return x.amax(dim=(3, 4)).transpose(1, 2)  # == AdaptiveMaxPool2d((1, 1)).

    class AudioEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = AudioBlock(1, 32)
            self.pool1 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block2 = AudioBlock(32, 64)
            self.pool2 = nn.MaxPool3d((1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1))
            self.block3 = AudioBlock(64, 128)

        def forward(self, x):  # (B, 1, 13, 4T) → (B, T, 128)
            x = self.block3(self.pool2(self.block2(self.pool1(self.block1(x)))))
            return x.mean(dim=2).transpose(1, 2)

    class BGRU(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.gru_forward = nn.GRU(c, c, num_layers=1, batch_first=True)
            self.gru_backward = nn.GRU(c, c, num_layers=1, batch_first=True)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.gelu(self.gru_forward(x)[0])
            x = torch.flip(self.gru_backward(torch.flip(x, dims=[1]))[0], dims=[1])
            return self.gelu(x)

    class ASDModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.visualEncoder = VisualEncoder()
            self.audioEncoder = AudioEncoder()
            self.GRU = BGRU(128)

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.FC = nn.Linear(128, 2)

    class LightASDNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = ASDModel()
            self.lossAV = Head()
            self.lossV = Head()

        def visual(self, video):  # (B, T, 112, 112) float 0-255
            x = (video / 255 - 0.4161) / 0.1688
            return self.model.visualEncoder(x.unsqueeze(1))

        def forward(self, video, mfcc):
            """Last-frame logit difference (speaking − silent); ``mfcc`` (B, 4T, 13) or None."""
            v = self.visual(video)
            if mfcc is None:
                out = self.lossV.FC(v[:, -1])
            else:
                a = self.model.audioEncoder(mfcc.unsqueeze(1).transpose(2, 3))
                out = self.lossAV.FC(self.model.GRU(a + v)[:, -1])
            return out[:, 1] - out[:, 0]

    return LightASDNet()


class TorchBackend:
    """Shared, stateless Light-ASD network (eval mode, no grad).

    ``calibration`` maps window length -> the ``(scale, bias)`` :class:`LightASD` applies to its logits.
    """

    needs_audio = True
    calibration = TORCH_CALIBRATION

    def __init__(self, device_name: str | None = None) -> None:
        path = model_path()
        if not path.exists():
            raise FileNotFoundError(f"{path}: provision with hearing.vision.light_asd.download_model()")
        _verify(path.read_bytes())
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Only effective before torch loads.
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Light-ASD requires torch; install hearing[ml]") from exc
        self._torch = torch
        net = _build_net()
        state = torch.load(path, map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        net.eval()
        self.net = net
        self.device = "cpu"
        requested = device_name or device.pick()
        if requested != "cpu":
            try:
                self.net.to(requested)
                self.device = requested
                self.scores(np.full((1, 5, CROP, CROP), 128, np.uint8), np.zeros((20, 13), np.float32))
            except (RuntimeError, NotImplementedError, TypeError):
                self.net.to("cpu")  # An op the accelerator lacks: CPU fallback for the whole net.
                self.device = "cpu"

    def scores(self, video: np.ndarray, mfcc: np.ndarray | None) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            v = torch.from_numpy(np.ascontiguousarray(video)).to(self.device).float()
            a = None
            if mfcc is not None:
                a = torch.from_numpy(np.ascontiguousarray(mfcc, dtype=np.float32)).to(self.device)
                a = a.unsqueeze(0).expand(v.shape[0], -1, -1)
            return self.net(v, a).float().cpu().numpy()


@models.register_loader("light_asd")
def _load_backend() -> TorchBackend:
    return TorchBackend()


# --------------------------------------------------------------------- detector
def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


@dataclass
class _Track:
    crops: deque
    crop_ts: float | None = None
    p: float | None = None
    last_ts: float | None = None


@register_asd("light_asd")
class LightASD:
    """Light-ASD over a sliding window of ``window`` frames per track (see module doc).

    The model runs on every ``every``-th ``update`` call (crops are still cut every
    frame); in between, ``p_speaking`` holds. ``scores[track_id]`` is the last raw (uncalibrated) logit, ``crop_info[track_id]`` the last
    crop geometry, ``last_compute_ms`` the model time of the last ``update`` (0 when skipped).
    """

    name = "light_asd"

    def __init__(
        self,
        window: int = DEFAULT_WINDOW,
        ema_s: float = 0.0,
        backend: Backend | str | None = "auto",
        crop_scale: float = 0.7,
        mouth_offset: float = 0.2,
        stale_s: float = 0.1,
        max_gap_s: float = 0.2,
        min_frames: int = 1,
        every: int = 1,
        calibration: tuple[float, float] | None = None,
    ) -> None:
        if window < 1 or every < 1:
            raise ValueError("window and every must be >= 1 frame")
        self.every = int(every)
        self.window = int(window)
        self.ema_s = ema_s
        self.crop_scale = crop_scale
        self.mouth_offset = mouth_offset
        self.stale_s = stale_s
        self.max_gap_s = max_gap_s
        self.min_frames = max(1, int(min_frames))
        self._calibration = calibration
        if backend in (None, "auto"):
            backend = "torch" if available() else MockBackend()
        if backend == "mock":
            backend = MockBackend()
        elif backend == "torch":
            if not available():
                raise RuntimeError(
                    f"Light-ASD weights or torch missing ({model_path()}); "
                    "run hearing.vision.light_asd.download_model()"
                )
            backend = None  # Loaded lazily through hearing.models on first use.
        elif isinstance(backend, str):
            raise ValueError(f"backend must be 'auto', 'torch', 'mock' or a Backend, not {backend!r}")
        self._backend: Backend | None = backend
        self.scores: dict[int, float] = {}
        self.crop_info: dict[int, tuple[float, float, float, bool]] = {}
        self.last_compute_ms = 0.0
        self.reset()

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = models.load("light_asd")
        return self._backend

    @property
    def calibration(self) -> tuple[float, float]:
        """``(scale, bias)`` for ``p = sigmoid(scale * logit + bias)``: the constructor's, else the backend's.

        A backend may give one pair or a ``{window: pair}`` table (the nearest window is used).
        """
        cal = self._calibration if self._calibration is not None else getattr(self.backend, "calibration", (1.0, 0.0))
        if isinstance(cal, dict):
            cal = calibration_for(cal, self.window)
        scale, bias = cal
        return float(scale), float(bias)

    @property
    def is_mock(self) -> bool:
        return isinstance(self._backend, MockBackend)

    def reset(self) -> None:
        self._tracks: dict[int, _Track] = {}
        self._mfcc_stream = StreamingMFCC()
        self._mfcc: deque[np.ndarray] = deque(maxlen=MFCC_PER_FRAME * self.window)
        self._audio_samples = 0  # Total samples received, for bookkeeping/tests.
        self._updates = 0
        self.scores.clear()
        self.crop_info.clear()
        self.last_compute_ms = 0.0

    @property
    def mfcc_frames(self) -> int:
        """MFCC frames currently in the audio window (≤ 4 × window)."""
        return len(self._mfcc)

    def window_frames(self, track_id: int) -> int:
        st = self._tracks.get(track_id)
        return len(st.crops) if st else 0

    def push_audio(self, chunk: AudioChunk) -> None:
        if chunk.sample_rate != SR:
            raise ValueError(f"Light-ASD expects {SR} Hz audio, got {chunk.sample_rate}")
        x = np.asarray(chunk.samples, dtype=np.float32).reshape(-1)
        self._audio_samples += x.size
        if self._backend is not None and not self._backend.needs_audio:
            n_new = (self._audio_samples // MFCC_HOP) - (self._audio_samples - x.size) // MFCC_HOP
            self._mfcc.extend(np.zeros((n_new, 13), np.float32))  # Bookkeeping only.
            return
        self._mfcc.extend(self._mfcc_stream.push(x))

    def update(
        self,
        tracks: list[FaceTrack],
        frame_ts: float,
        audio: AudioChunk | None,
        frame: VideoFrame | None = None,
    ) -> list[FaceTrack]:
        self.last_compute_ms = 0.0
        if audio is not None:
            self.push_audio(audio)
        live = {t.track_id for t in tracks}
        for tid in [tid for tid in self._tracks if tid not in live]:
            del self._tracks[tid]
            self.scores.pop(tid, None)
            self.crop_info.pop(tid, None)

        ready: list[tuple[FaceTrack, _Track]] = []
        for track in tracks:
            st = self._tracks.get(track.track_id)
            if st is None:
                st = self._tracks[track.track_id] = _Track(deque(maxlen=self.window))
            fresh = track.last_seen_ts + self.stale_s >= frame_ts
            if frame is None or not fresh:
                track.p_speaking = float(st.p or 0.0)
                continue
            if st.crop_ts is not None and frame_ts - st.crop_ts > self.max_gap_s:
                st.crops.clear()  # The window must be contiguous in time.
            h, w = frame.image.shape[:2]
            geom = crop_geometry(track.box, track.landmarks, w, h, self.crop_scale, self.mouth_offset)
            self.crop_info[track.track_id] = geom
            st.crops.append(cut_crop(frame.image, *geom[:3]))
            st.crop_ts = frame_ts
            if len(st.crops) >= self.min_frames:
                ready.append((track, st))
            else:
                track.p_speaking = float(st.p or 0.0)

        run = self._updates % self.every == 0
        self._updates += 1
        if ready and not run:
            for track, st in ready:
                track.p_speaking = float(st.p or 0.0)
        elif ready:
            backend = self.backend
            n_audio = len(self._mfcc) // MFCC_PER_FRAME
            groups: dict[int, list[tuple[FaceTrack, _Track]]] = {}
            for track, st in ready:
                t = min(len(st.crops), n_audio) if n_audio else len(st.crops)
                groups.setdefault(t, []).append((track, st))
            mfcc_all = np.stack(self._mfcc) if n_audio else None
            scale, bias = self.calibration
            start = time.perf_counter()
            for t, group in groups.items():
                video = np.stack([np.stack(list(st.crops)[-t:]) for _, st in group])
                mfcc = mfcc_all[-MFCC_PER_FRAME * t :] if mfcc_all is not None else None
                logits = backend.scores(video, mfcc)
                for (track, st), logit in zip(group, np.asarray(logits).tolist(), strict=True):
                    self.scores[track.track_id] = float(logit)
                    raw = _sigmoid(scale * float(logit) + bias)
                    if st.p is None or st.last_ts is None:
                        st.p = raw
                    else:
                        dt = max(frame_ts - st.last_ts, 0.0)
                        alpha = 1.0 - math.exp(-dt / self.ema_s) if self.ema_s > 0 else 1.0
                        st.p += alpha * (raw - st.p)
                    st.last_ts = frame_ts
                    track.p_speaking = float(min(max(st.p, 0.0), 1.0))
            self.last_compute_ms = (time.perf_counter() - start) * 1000.0
        return tracks


# --------------------------------------------------------------------- synthetic input
def synthetic_face_frame(
    boxes: list[tuple[float, float, float, float]],
    openings: list[float],
    size: tuple[int, int] = (360, 640),
) -> np.ndarray:
    """RGB cartoon faces (skin ellipse, eyes, dark mouth of height ∝ opening in [0, 1])."""
    h, w = size
    img = np.full((h, w, 3), (90, 100, 110), np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    for (bx, by, bw, bh), opening in zip(boxes, openings, strict=True):
        cx, cy, rx, ry = (bx + bw / 2) * w, (by + bh / 2) * h, bw * w / 2, bh * h / 2
        img[((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1] = (205, 160, 135)
        for ex in (cx - 0.35 * rx, cx + 0.35 * rx):
            img[((xx - ex) / (0.12 * rx)) ** 2 + ((yy - (cy - 0.25 * ry)) / (0.07 * ry)) ** 2 <= 1] = (40, 30, 30)
        my, mrx = cy + 0.5 * ry, 0.35 * rx
        mry = max(0.02 + 0.16 * float(np.clip(opening, 0, 1)), 0.02) * ry
        img[((xx - cx) / mrx) ** 2 + ((yy - my) / mry) ** 2 <= 1] = (80, 20, 30)
    return img


# --------------------------------------------------------------------- CLI
def benchmark(
    backend: Backend, n_tracks: int = 2, window: int = DEFAULT_WINDOW, frames: int = 50, every: int = 1
) -> dict:
    """Per-frame ``update`` time for ``n_tracks`` tracks once the window is full."""
    det = LightASD(window=window, backend=backend, every=every)
    rng = np.random.default_rng(0)
    timings, model_ms = [], []
    for i in range(window + frames):
        boxes = [(0.05 + 0.9 * k / n_tracks, 0.2, 0.8 / n_tracks, 0.5) for k in range(n_tracks)]
        image = synthetic_face_frame(boxes, [0.5 + 0.5 * math.sin(i)] * n_tracks)
        tracks = [FaceTrack(k, box, None, 0.0, i / FPS) for k, box in enumerate(boxes)]
        chunk = AudioChunk((rng.normal(0, 0.05, SR // FPS)).astype(np.float32), SR, i, i / FPS)
        before = time.perf_counter()
        det.update(tracks, i / FPS, chunk, VideoFrame(image, i, i / FPS))
        if i >= window:
            timings.append((time.perf_counter() - before) * 1000.0)
            model_ms.append(det.last_compute_ms)
    return {
        "device": getattr(backend, "device", "?"),
        "tracks": n_tracks,
        "window": window,
        "every": every,
        "update_ms_mean": float(np.mean(timings)),
        "update_ms_p50": float(np.percentile(timings, 50)),
        "update_ms_p95": float(np.percentile(timings, 95)),
        "model_ms_mean": float(np.mean(model_ms)),
    }


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json
    import platform

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true", help="fetch the pinned checkpoint first")
    parser.add_argument("--tracks", type=int, default=2)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--every", type=int, default=1, help="run the model every N frames")
    args = parser.parse_args(argv)
    if args.download:
        download_model()
    out: dict = {
        "picked_device": device.pick(),
        "weights": str(model_path()),
        "weights_present": model_path().exists(),
        "available": available(),
        "machine": f"{platform.machine()} {platform.platform()}",
        "timing": [],
    }
    if available():
        devices = ["cpu"] + [d for d in {device.pick()} if d != "cpu"]
        for name in devices:
            backend = TorchBackend(name)
            out["timing"].append(benchmark(backend, args.tracks, args.window, every=args.every))
    else:
        out["hint"] = "weights missing: python -m hearing.vision.light_asd --download"
        out["timing"].append(benchmark(MockBackend(), args.tracks, args.window, every=args.every))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
