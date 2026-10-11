"""Apparent gender/age from a face crop: InsightFace ``genderage.onnx`` behind ``FaceEstimator``.

Model: the ``buffalo_l`` pack's attribute head (96x96 RGB crop, float 0-255 NCHW, output
``[female, male, age / 100]``), pinned from the Hugging Face mirror ``public-data/insightface``
and checksum-verified. InsightFace models are for **non-commercial research use only**.

Weights: ``download_model()`` installs them into ``hearing.models.cache_dir()/apparent_face/``;
the ``apparent_face`` loader calls it, so a first ``estimate`` without weights downloads
1.3 MB. ``hearing.synth.voice_match.get_face_estimator`` checks ``available()`` first and
returns None instead when the weights are absent. Importing this module never touches the
network or onnxruntime.

Crop: square, centred on the (frame-clamped) face box, side ``CROP_SCALE`` x its longer
edge as in InsightFace's ``Attribute.get``; pixels outside the frame are zero, matching
the model's training-time ``warpAffine``. Faces whose visible box is under ``MIN_FACE_PX``
on a side give None.

Privacy: estimates are returned to the caller and nothing else. Never log, print, store or
send them; this module emits no log records.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from hearing import device, models
from hearing.synth.voice_match import AGES, ApparentEstimate

REVISION = "33c1063c49c785b7652d3fd529f86fa4f149392b"
MODEL_URL = (
    f"https://huggingface.co/public-data/insightface/resolve/{REVISION}/"
    "models/buffalo_l/genderage.onnx"
)
MODEL_SHA256 = "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb"
INPUT_SIZE = 96
CROP_SCALE = 1.5  # crop side / longer face-box side (InsightFace Attribute.get)
MIN_FACE_PX = 32
AGE_BOUNDS = (25.0, 60.0)  # young < 25 <= adult <= 60 < senior
AGE_SIGMA = 8.0  # years; spreads the point estimate into soft bins (genderage MAE is ~this)


def model_path() -> Path:
    return models.cache_dir() / "apparent_face" / "genderage.onnx"


def weights_available() -> bool:
    """True when the pinned weights are in the cache dir (size check only, no hashing)."""
    path = model_path()
    return path.is_file() and path.stat().st_size > 0


def available() -> bool:
    """``voice_match._load_adapter`` hook: weights present and onnxruntime importable."""
    if not weights_available():
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def _verify(data: bytes) -> None:
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError("genderage.onnx checksum mismatch; re-provision the pinned weights")


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


def _providers(ort) -> list[str]:
    # ORT has no MPS provider, and CoreML is slower than CPU for this 1.3 MB graph.
    providers = ["CPUExecutionProvider"]
    if device.pick() == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    return providers


@models.register_loader("apparent_face")
def _load_session():
    path = download_model()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("apparent_face requires onnxruntime; install hearing[ml]") from exc
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=_providers(ort))


def face_pixels(
    shape: tuple[int, ...], box: tuple[float, float, float, float]
) -> tuple[float, float, float, float] | None:
    """Normalised (x, y, w, h) -> pixel (x0, y0, x1, y1) clamped to the frame, or None if empty."""
    height, width = shape[:2]
    x, y, w, h = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        return None
    x0, x1 = np.clip([x * width, (x + w) * width], 0, width)
    y0, y1 = np.clip([y * height, (y + h) * height], 0, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def crop(image: np.ndarray, face: tuple[float, float, float, float]) -> np.ndarray:
    """Square ``INPUT_SIZE`` RGB float32 crop around a pixel face box, zero outside the frame.

    Bilinear sampling, as InsightFace's ``warpAffine`` at the same scale and centre.
    """
    x0, y0, x1, y1 = face
    side = CROP_SCALE * max(x1 - x0, y1 - y0)
    step = side / INPUT_SIZE
    # Output pixel i samples source coordinate (centre - side/2) + (i + 0.5) * step - 0.5.
    grid = (np.arange(INPUT_SIZE) + 0.5) * step - 0.5 - side / 2
    xs, xw = _taps((x0 + x1) / 2 + grid, image.shape[1])
    ys, yw = _taps((y0 + y1) / 2 + grid, image.shape[0])
    out = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.float32)
    for yi, wy in zip(ys, yw, strict=True):
        rows = image[yi, :, :3]  # only the sampled rows, never a copy of the frame
        for xi, wx in zip(xs, xw, strict=True):
            out += rows[:, xi] * (wy[:, None, None] * wx[None, :, None])
    return out


def _taps(coords: np.ndarray, size: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Bilinear neighbour indices (clipped) and weights (0 outside ``[0, size)``)."""
    low = np.floor(coords).astype(int)
    frac = (coords - low).astype(np.float32)
    indices, weights = [], []
    for index, weight in ((low, 1 - frac), (low + 1, frac)):
        inside = (index >= 0) & (index < size)
        indices.append(np.clip(index, 0, size - 1))
        weights.append(np.where(inside, weight, 0).astype(np.float32))
    return indices, weights


def _age_bins(age: float) -> tuple[float, float, float]:
    """P(true age in young/adult/senior) under N(age, AGE_SIGMA)."""

    def cdf(v: float) -> float:
        return 0.5 * (1 + math.erf((v - age) / (AGE_SIGMA * math.sqrt(2))))

    low, high = (cdf(b) for b in AGE_BOUNDS)
    return (low, high - low, 1 - high)


def to_estimate(pred: np.ndarray, face_px: float) -> ApparentEstimate:
    """Model output ``[female, male, age/100]`` + visible face size -> ``ApparentEstimate``."""
    female, male, age = (float(v) for v in np.asarray(pred, np.float64).reshape(-1)[:3])
    p_female = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, male - female))))  # softmax of 2
    size = min(1.0, 0.5 + 0.5 * (face_px - MIN_FACE_PX) / (INPUT_SIZE - MIN_FACE_PX))
    confidence = abs(2 * p_female - 1) * size
    p_age = _age_bins(min(max(age * 100.0, 0.0), 100.0))
    assert len(p_age) == len(AGES)
    return ApparentEstimate(p_female, p_age, float(confidence), "face")


class FaceEstimator:
    """``voice_match.FaceEstimator`` on InsightFace genderage. Construction never loads."""

    def __init__(self, session=None):
        self._session = session
        self._input = None

    def estimate(
        self, image: np.ndarray, box: tuple[float, float, float, float]
    ) -> ApparentEstimate | None:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("FaceEstimator expects an HxWx3 RGB uint8 frame")
        face = face_pixels(image.shape, box)
        if face is None:
            return None
        face_px = min(face[2] - face[0], face[3] - face[1])
        if face_px < MIN_FACE_PX:
            return None
        if self._session is None:
            self._session = models.load("apparent_face")
        if self._input is None:
            self._input = self._session.get_inputs()[0].name
        blob = crop(image, face).transpose(2, 0, 1)[None]  # RGB, 0-255, mean 0 / std 1
        pred = self._session.run(None, {self._input: np.ascontiguousarray(blob)})[0]
        return to_estimate(pred[0], face_px)


def _bench(calls: int = 100) -> dict:
    """Load time and ms per face (p50/p95) on a synthetic 640x360 frame; no estimates printed."""
    import time

    start = time.perf_counter()
    session = models.load("apparent_face")
    load_ms = (time.perf_counter() - start) * 1000
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    box = (0.4, 0.3, 0.15, 0.35)  # ~96x126 px face
    estimator = FaceEstimator(session)
    estimator.estimate(image, box)  # warm-up
    times = []
    for _ in range(calls):
        start = time.perf_counter()
        estimator.estimate(image, box)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "device": device.pick(),
        "providers": session.get_providers(),
        "load_ms": round(load_ms, 1),
        "ms_per_face_p50": round(float(np.percentile(times, 50)), 3),
        "ms_per_face_p95": round(float(np.percentile(times, 95)), 3),
        "calls": calls,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(_bench(), indent=2))
