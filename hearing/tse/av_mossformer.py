"""Direct ClearerVoice AV-MossFormer2 inference on caller-provided mouth crops.

No face detector, media files, or implicit downloads. Call ``download_model``
explicitly before loading real weights. Preprocessing follows ClearerVoice's
utils/video_process.py: 112px gray, /255, mean .4161, std .1688.
Only this direct-network path exists (``method = "direct"``); there is no temp-mp4 path.
Measured Oct 5 2026 on an M4 Pro: native MPS, 1 s window about 215 ms.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np

from hearing import device, models
from hearing.tse.base import conform_output, register_candidate

MODEL = "AV_MossFormer2_TSE_16K"
REPO = f"alibabasglab/{MODEL}"
# Shared weights dir: $HEARING_MODELS_DIR/av_mossformer2/last_best_checkpoint.pt (735 MB).
WEIGHTS_DIR = "av_mossformer2"
log = logging.getLogger(__name__)


def download_model() -> Path:
    """Explicitly fetch only the current checkpoint into the shared model dir."""
    from huggingface_hub import snapshot_download

    root = models.cache_dir() / WEIGHTS_DIR
    snapshot_download(
        REPO,
        local_dir=root,
        allow_patterns=["last_best_checkpoint", "last_best_checkpoint.pt"],
    )
    return root


def available() -> bool:
    return (
        importlib.util.find_spec("clearvoice") is not None
        and (models.cache_dir() / WEIGHTS_DIR / "last_best_checkpoint.pt").is_file()
    )


def prepare_lips(
    lips: np.ndarray | None, n_samples: int, lips_fps: float = 25.0
) -> np.ndarray:
    """Return contiguous float32 [B=1,T,112,112], at 25 fps over the window.

    Frame zero is sample-aligned. Interpolate on the source timestamp grid;
    if the supplied clip duration disagrees, stretch its grid to the window.
    """
    if (
        lips is None
        or lips.ndim != 3
        or len(lips) == 0
        or lips.shape[1:] != (88, 88)
        or lips.dtype != np.uint8
    ):
        raise ValueError("lips must be nonempty T x 88 x 88 uint8 grayscale")
    if not np.isfinite(lips_fps) or lips_fps <= 0 or n_samples <= 0:
        raise ValueError("positive audio length and lips_fps required")
    import cv2

    duration = n_samples / 16000
    count = max(1, int(np.ceil(duration * 25)))
    source_duration = len(lips) / lips_fps
    source_times = np.arange(len(lips)) / lips_fps * duration / source_duration
    positions = np.interp(np.arange(count) / 25, source_times, np.arange(len(lips)))
    low = positions.astype(int)
    high = np.minimum(low + 1, len(lips) - 1)
    resized = np.stack(
        [
            cv2.resize(frame, (112, 112), interpolation=cv2.INTER_LINEAR)
            for frame in lips
        ]
    ).astype(np.float32)
    alpha = (positions - low).astype(np.float32)[:, None, None]
    frames = resized[low] * (1 - alpha) + resized[high] * alpha
    return np.ascontiguousarray(
        ((frames / 255 - 0.4161) / 0.1688)[None], dtype=np.float32
    )


def _decoder_forward(self, mixture_w, est_mask):
    """Equivalent overlap-add using fold on the input device, avoiding upstream .cuda()."""
    from torch.nn import functional

    frames = self.basis_signals((mixture_w * est_mask).transpose(2, 1))
    n = (frames.shape[1] - 1) * (self.L // 2) + self.L
    return functional.fold(
        frames.transpose(1, 2), (1, n), (1, self.L), stride=(1, self.L // 2)
    ).reshape(frames.shape[0], n)


class MockBackend:
    """Explicit offline identity backend; never selected for real bake-off runs."""

    device = "cpu"

    def __call__(self, audio, visual):
        return audio.copy()


class _NetworkBackend:
    def __init__(self):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        root = models.cache_dir() / WEIGHTS_DIR
        checkpoint = root / "last_best_checkpoint.pt"
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Missing {checkpoint}; call download_model() explicitly"
            )
        try:
            import clearvoice
            import torch
            import yaml
            from clearvoice.models.av_mossformer2_tse.av_mossformer2 import (
                AV_MossFormer2_TSE_16K,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install the clearvoice and ml extras before loading"
            ) from exc
        config = (
            Path(clearvoice.__file__).parent / "config" / "inference" / f"{MODEL}.yaml"
        )
        data = yaml.safe_load(config.read_text())
        args = SimpleNamespace(
            **{
                k: SimpleNamespace(**v) if isinstance(v, dict) else v
                for k, v in data.items()
            }
        )
        self.network = AV_MossFormer2_TSE_16K(args).model
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.network.load_state_dict(state, strict=True)
        decoder = self.network.sep_network.decoder
        decoder.forward = MethodType(_decoder_forward, decoder)
        self.device = device.pick()
        self.network.eval().to(self.device)

    def __call__(self, audio, visual):
        import torch

        def forward():
            with torch.inference_mode():
                return (
                    self.network(
                        torch.from_numpy(audio).to(self.device),
                        torch.from_numpy(visual).to(self.device),
                    )
                    .cpu()
                    .numpy()
                )

        try:
            return forward()
        except (NotImplementedError, RuntimeError) as exc:
            if self.device != "mps" or not any(
                word in str(exc).lower() for word in ("mps", "metal")
            ):
                raise
            log.warning(
                "AV-MossFormer2 MPS operation unsupported; retrying CPU: %s", exc
            )
            self.device = "cpu"
            self.network.to("cpu")
            return forward()


@models.register_loader("av_mossformer2")
def _load_backend():
    return _NetworkBackend()


@register_candidate("av_mossformer2")
class AVMossFormer2:
    name = "av_mossformer2"
    licence = "Apache-2.0"
    needs = frozenset({"lips"})
    method = "direct"

    def __init__(self, backend=None):
        self.backend = backend
        self.device = getattr(backend, "device", None)

    def load(self) -> None:
        if self.backend is None:
            self.backend = models.load(self.name)
        self.device = self.backend.device

    def extract(
        self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None
    ):
        mix = np.asarray(mix)
        if (
            sr != 16000
            or mix.ndim != 1
            or mix.dtype != np.float32
            or not np.isfinite(mix).all()
        ):
            raise ValueError("mix must be finite mono float32 at 16000 Hz")
        visual = prepare_lips(lips, len(mix), lips_fps)
        self.load()
        peak = float(np.max(np.abs(mix)))
        scale = peak if peak > 0 else 1.0
        out = self.backend(np.ascontiguousarray(mix[None] / scale), visual)
        self.device = self.backend.device
        return conform_output(np.asarray(out).reshape(-1) * scale, len(mix), self.name)
