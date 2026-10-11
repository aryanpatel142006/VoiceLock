"""Dolphin audio-visual target speaker extraction (JusperLee/Dolphin) as the ``dolphin`` candidate.

The model definition is vendored under ``hearing/tse/_dolphin/`` (Apache-2.0 code; upstream has no
installable package). The released checkpoint is **not** cleared for commercial use (distilled from
AV-HuBERT, trained on LRS2/LRS3/VoxCeleb2; see upstream ``MODEL_PROVENANCE.md``): hackathon and
research only.

Weights never download on load: call :func:`download_model` explicitly first (it writes
``$HEARING_MODELS_DIR/dolphin/{config.json,model.safetensors}``). Inject :class:`MockBackend` for
deterministic offline operation, never for quality claims.

Visual front-end: ``T x 88 x 88`` grayscale mouth crops at 25 fps, scaled to [0, 1] and normalised
with mean 0.421 / std 0.165 (upstream ``get_preprocessing_pipelines()["val"]``), shaped
``(1, 1, T, 88, 88)``; ``T = int(len(mix) / 16000 * 25)`` like upstream ``Inference.py``.
"""

from __future__ import annotations

import json
import logging
import os
from importlib.util import find_spec

import numpy as np

from hearing import device, models
from hearing.tse.base import LIPS_FPS, SR, conform_output, register_candidate

log = logging.getLogger(__name__)

MODEL = "dolphin"
REPO = "JusperLee/Dolphin"
REVISION = "348a88fb05fc2953ee585d5ab7a7f27f1efb9c13"  # HF model repo commit, 2026-10-05
FILES = ("config.json", "model.safetensors")
LICENCE = (
    "non-commercial: checkpoint (HF JusperLee/Dolphin, licence 'other') not cleared for commercial "
    "use (AV-HuBERT distillation, LRS2/LRS3/VoxCeleb2); code Apache-2.0"
)
LIP_SIZE = 88
LIP_MEAN, LIP_STD = 0.421, 0.165
FPS = 25.0
_HF_METADATA = {"model_type", "task", "framework", "license", "tags", "architectures", "auto_map"}
# Weights the released checkpoint omits on purpose: training-only heads of the video encoder.
_TRAINING_ONLY = (
    "video_encoder.semantic_model.distill_out.",
    "video_encoder.recon_model.conv_out.",
    "video_encoder.recon_model.decoder_layers.",
)


def model_dir():
    return models.cache_dir() / MODEL


def download_model():
    """Explicitly acquire the non-commercial checkpoint (~28 MB) in HEARING_MODELS_DIR/dolphin."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=REPO, revision=REVISION, local_dir=model_dir(), allow_patterns=list(FILES)
    )


def _files():
    root = model_dir()
    paths = [root / f for f in FILES]
    if not all(p.is_file() for p in paths):
        raise RuntimeError(
            f"Dolphin weights absent in {root}; call hearing.tse.dolphin.download_model()"
        )
    return paths


def available() -> bool:
    try:
        _files()
    except RuntimeError:
        return False
    return all(
        find_spec(m) is not None
        for m in ("torch", "safetensors", "vector_quantize_pytorch", "taylor_series_linear_attention")
    )


def num_frames(n_samples: int, sr: int = SR) -> int:
    """Video frames Dolphin sees for ``n_samples`` of audio (upstream: ``int(n / sr * 25)``, >= 1)."""
    return max(1, int(n_samples / sr * FPS))


def prepare_lips(lips, n_samples, lips_fps=LIPS_FPS, sr=SR):
    """``T x H x W`` uint8 crops -> normalised float32 ``(1, 1, T', 88, 88)`` at 25 fps.

    Frames are resampled to 25 fps by nearest index (repeating the last frame when the crops end
    early), resized to 88x88 bilinearly when needed, scaled to [0, 1] and normalised.
    """
    lips = np.asarray(lips)
    if lips.ndim != 3 or not lips.shape[0] or not np.isfinite(lips_fps) or lips_fps <= 0:
        raise ValueError(f"dolphin needs T x H x W lip crops and positive lips_fps, got {lips.shape}")
    t_out = num_frames(n_samples, sr)
    idx = np.minimum(np.round(np.arange(t_out) / FPS * lips_fps).astype(int), lips.shape[0] - 1)
    frames = lips[idx].astype(np.float32)
    if frames.shape[1:] != (LIP_SIZE, LIP_SIZE):
        from scipy.ndimage import zoom

        h, w = frames.shape[1:]
        frames = zoom(frames, (1, LIP_SIZE / h, LIP_SIZE / w), order=1, grid_mode=True, mode="nearest")
        frames = frames[:, :LIP_SIZE, :LIP_SIZE]
    frames = (frames / 255.0 - LIP_MEAN) / LIP_STD
    return np.ascontiguousarray(frames[None, None], dtype=np.float32)


class MockBackend:
    """Offline backend: returns ``target`` if given, else the mixture; records the lip tensor."""

    device = "cpu"

    def __init__(self, target=None):
        self.target = target
        self.last_mouth = None

    def extract(self, mix, mouth):
        self.last_mouth = mouth
        return mix.copy() if self.target is None else np.asarray(self.target, dtype=np.float32)


class _TorchBackend:
    """Runs the network; on an MPS op failure it moves to CPU for good and says so in the log."""

    def __init__(self, network, dev):
        self.network = network
        self.device = dev

    def _run(self, mix, mouth):
        import torch

        with torch.inference_mode():
            audio = torch.from_numpy(mix).unsqueeze(0).to(self.device)
            video = torch.from_numpy(mouth).to(self.device)
            return self.network(audio, video).detach().cpu().numpy().reshape(-1)

    def extract(self, mix, mouth):
        if self.device != "mps":
            return self._run(mix, mouth)
        try:
            return self._run(mix, mouth)
        except RuntimeError as e:  # NotImplementedError included
            log.warning("dolphin: MPS op failed (%s); falling back to CPU", e)
            self.network = self.network.to("cpu")
            self.device = "cpu"
            return self._run(mix, mouth)


@models.register_loader(MODEL)
def _load_network():
    config_path, weights_path = _files()
    if device.pick() == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    try:
        from safetensors.torch import load_file

        from hearing.tse._dolphin.model import Dolphin
    except ImportError as exc:
        raise RuntimeError("Dolphin requires torch and the ml extra") from exc
    config = json.loads(config_path.read_text())
    config = {k: v for k, v in config.items() if k not in _HF_METADATA}
    config["is_train"] = False  # True would fetch the video encoder from the Hub
    network = Dolphin(**config)
    result = network.load_state_dict(load_file(str(weights_path), device="cpu"), strict=False)
    missing = [k for k in result.missing_keys if not k.startswith(_TRAINING_ONLY)]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Dolphin checkpoint mismatch: missing={missing[:5]} unexpected={result.unexpected_keys[:5]}"
        )
    dev = device.pick()
    return _TorchBackend(network.to(dev).eval(), dev)


@register_candidate("dolphin")
class DolphinCandidate:
    name = "dolphin"
    licence = LICENCE
    needs = frozenset({"lips"})

    def __init__(self, backend=None):
        self.backend = backend
        self.device = getattr(backend, "device", "")

    def load(self):
        if self.backend is None:
            self.backend = models.load(MODEL)
        self.device = self.backend.device

    def extract(self, mix, sr, *, lips=None, lips_fps=LIPS_FPS, p_speaking=None, enroll=None):
        mix = np.asarray(mix, dtype=np.float32)
        if sr != SR or mix.ndim != 1 or not mix.size or not np.isfinite(mix).all():
            raise ValueError("dolphin expects nonempty finite mono audio at 16000 Hz")
        if lips is None:
            raise ValueError("dolphin needs lips")
        mouth = prepare_lips(lips, len(mix), lips_fps, sr)
        self.load()
        y = conform_output(self.backend.extract(np.ascontiguousarray(mix), mouth), len(mix), MODEL)
        self.device = self.backend.device
        if not np.isfinite(y).all():
            raise ValueError("dolphin returned nonfinite audio")
        return y
