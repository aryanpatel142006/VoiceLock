"""MossFormer2 separation and visual speaking-probability association.

Real backends never download on load; call ``download_model`` explicitly first.
Inject ``MockBackend`` for deterministic offline operation, never for quality claims.
"""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hearing import device, models
from hearing.tse.base import SR, conform_output, register_candidate

MODEL = "MossFormer2_SS_16K"
REPO = f"alibabasglab/{MODEL}"
FRAME_SAMPLES = 640  # 40 ms at 16 kHz

# Alternative backend (T-372): TIGER-speech, vendored in hearing/tse/_tiger. Selected by HEARING_SEPARATOR;
# unset or "mossformer2" keeps MossFormer2. Apache-2.0 weights, MIT code; revisions pinned.
TIGER_REPOS = {
    "tiger": ("TIGER-speech-small", "JusperLee/TIGER-speech-small", "46f86be1a7255790e5c6d3a06d1fc4c94dee74cc"),
    "tiger-small": ("TIGER-speech-small", "JusperLee/TIGER-speech-small", "46f86be1a7255790e5c6d3a06d1fc4c94dee74cc"),
    "tiger-large": ("TIGER-speech", "JusperLee/TIGER-speech", "f0340340b2d9bbf72074edf8c076dcab59a10ba2"),
}


def separator_model() -> str:
    """Registered model name of the selected separator (``HEARING_SEPARATOR``, default MossFormer2)."""
    kind = os.environ.get("HEARING_SEPARATOR", "").strip().lower()
    if kind in ("", "mossformer2"):
        return MODEL
    if kind not in TIGER_REPOS:
        raise ValueError(f"HEARING_SEPARATOR={kind!r}; use mossformer2, tiger, tiger-small or tiger-large")
    return TIGER_REPOS[kind][0]


def download_tiger(kind: str = "tiger"):
    """Explicitly acquire TIGER weights (pinned revision) in HEARING_MODELS_DIR."""
    from huggingface_hub import snapshot_download

    name, repo, revision = TIGER_REPOS[kind]
    return snapshot_download(repo_id=repo, revision=revision, local_dir=models.cache_dir() / name)


def _load_tiger(name):
    root = models.cache_dir() / name
    if not (root / "model.safetensors").is_file():
        raise RuntimeError(f"{name} weights absent; call hearing.tse.blind.download_tiger()")
    import torch
    from safetensors.torch import load_file

    torch.backends.nnpack.set_flags(False)  # NNPACK's tiny 2-D convs ran ~12x slower on CPU (69 s vs 5.7 s per 1 s window)
    from hearing.tse._tiger.model import TIGER

    cfg = json.loads((root / "config.json").read_text())
    network = TIGER(**cfg)
    network.load_state_dict(load_file(root / "model.safetensors"), strict=True)
    dev = device.pick()
    backend = _TorchBackend(network.to(dev).eval(), dev)
    try:  # complex STFT ops are patchy on MPS: fall back to CPU rather than fail
        backend.separate(np.zeros(SR, np.float32))
    except Exception:  # noqa: BLE001
        backend = _TorchBackend(network.to("cpu").eval(), "cpu")
    return backend


def download_model():
    """Explicitly acquire the Apache-2.0 checkpoint in HEARING_MODELS_DIR."""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=REPO, local_dir=models.cache_dir() / MODEL)


def pick_fixed() -> bool:
    """HEARING_PICK=fixed (default off): a fine-tuned fixed-order checkpoint, slot 0 = first
    enrolled speaker (Aryan), slot 1 = second (Joseph); see experiments/gpu/T-374. Another pair's
    checkpoint names its order in ``$HEARING_FIXED_SLOTS`` (T-447)."""
    return os.environ.get("HEARING_PICK", "").strip().lower() == "fixed"


FIXED_SLOTS = ("aryan", "joseph")  # output order the T-374 fixed-order checkpoint was trained on
FIXED_SLOTS_ENV = "HEARING_FIXED_SLOTS"  # T-447: e.g. "joseph,ahmad" for the JA checkpoint


def fixed_slots():
    """Profile names of slot 0 and 1: ``$HEARING_FIXED_SLOTS`` (two distinct comma-separated names,
    T-447), unset/empty = FIXED_SLOTS. A malformed value raises instead of playing the wrong slot."""
    raw = os.environ.get(FIXED_SLOTS_ENV, "").strip()
    if not raw:
        return FIXED_SLOTS
    names = tuple(n.strip().lower() for n in raw.split(","))
    if len(names) != 2 or not all(names) or names[0] == names[1]:
        raise RuntimeError(f"{FIXED_SLOTS_ENV} needs two distinct names, e.g. joseph,ahmad: {raw!r}")
    return names


def fixed_slot(profile_name):
    """Stream index for a voice profile under HEARING_PICK=fixed, else None (pick as usual)."""
    if not pick_fixed():
        return None
    name = (profile_name or "").strip().lower()
    slots = fixed_slots()
    return slots.index(name) if name in slots else None


# T-465 auto-switch (default off): the shipped separator for anyone, the fine-tuned fixed-order checkpoint named here
# only for windows whose target is a saved profile in HEARING_FIXED_SLOTS (and whose partner, when it is a saved
# profile, is the other slot). Both are loaded at start; hearing.tse.enrolled picks one per window.
TEAM_CKPT_ENV = "HEARING_SEP_TEAM_CKPT"
TEAM_MODEL = f"{MODEL}@team"  # hearing.models name of the team checkpoint


def team_checkpoint() -> Path | None:
    """``$HEARING_SEP_TEAM_CKPT`` as a path (None when unset); a missing file is a RuntimeError."""
    raw = os.environ.get(TEAM_CKPT_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise RuntimeError(f"{TEAM_CKPT_ENV} is not a file: {path}")
    return path


def team_slot(target_name, partner_name=None):
    """Slot of the team checkpoint to play for this target, or None (use the stock separator and the ECAPA pick).

    The target must be a saved profile named in ``fixed_slots()``; a partner that is itself a saved profile must be
    the other slot (a live-calibrated partner, ``None``, does not veto). Independent of ``HEARING_PICK``."""
    slots = fixed_slots()
    name = (target_name or "").strip().lower()
    if name not in slots:
        return None
    slot = slots.index(name)
    partner = (partner_name or "").strip().lower()
    if partner and partner != slots[1 - slot]:
        return None
    return slot


SEP_CKPT_ENV = "HEARING_SEP_CKPT"


def _checkpoint():
    override = os.environ.get(SEP_CKPT_ENV, "").strip()
    if override:  # fine-tuned weights, same architecture; unset keeps the shipped base
        path = Path(override).expanduser()
        if not path.is_file():
            raise RuntimeError(f"HEARING_SEP_CKPT is not a file: {path}")
        return path
    root = models.cache_dir() / MODEL
    index = root / "last_best_checkpoint"
    if index.is_file():
        path = (root / index.read_text().strip().splitlines()[0]).resolve()
        if path.is_relative_to(root.resolve()) and path.is_file():
            return path
    raise RuntimeError(
        "MossFormer2 weights absent; call hearing.tse.blind.download_model()"
    )


def available() -> bool:
    try:
        _checkpoint()
    except RuntimeError:
        return False
    return find_spec("clearvoice") is not None and find_spec("torch") is not None


class MockBackend:
    """Offline separator: supplied streams, or mixture and silence (no separation)."""

    device = "cpu"

    def __init__(self, streams=None):
        self.streams = streams

    def separate(self, mix):
        return (
            self.streams
            if self.streams is not None
            else np.stack([mix, np.zeros_like(mix)])
        )


class _TorchBackend:
    def __init__(self, network, dev):
        self.network = network
        self.device = dev

    def separate(self, mix):
        import torch

        with torch.inference_mode():
            outputs = self.network(torch.from_numpy(mix).unsqueeze(0).to(self.device))
        outputs = outputs[0] if getattr(outputs, "ndim", 0) == 3 else outputs  # TIGER: (1, 2, T)
        return np.stack([y.detach().cpu().numpy().reshape(-1) for y in outputs])


@models.register_loader(MODEL)
def _load_separator():
    return _load_mossformer(_checkpoint())


@models.register_loader(TEAM_MODEL)
def _load_team_separator():
    checkpoint = team_checkpoint()
    if checkpoint is None:
        raise RuntimeError(f"{TEAM_CKPT_ENV} is not set")
    return _load_mossformer(checkpoint)


def _load_mossformer(checkpoint):
    try:
        import torch
        from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
    except ImportError as exc:
        raise RuntimeError("MossFormer2 requires clearvoice and the ml extra") from exc
    # ClearVoice 0.1.2 inference configuration, instantiated without its auto-download wrapper.
    args = SimpleNamespace(
        encoder_embedding_dim=512,
        mossformer_sequence_dim=512,
        num_mossformer_layer=24,
        encoder_kernel_size=16,
        num_spks=2,
    )
    network = MossFormer2_SS_16K(args).model
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    network.load_state_dict(state, strict=True)
    dev = device.pick()
    return _TorchBackend(network.to(dev).eval(), dev)


for _name in {v[0] for v in TIGER_REPOS.values()}:
    models.register_loader(_name)(lambda _n=_name: _load_tiger(_n))


def validate_audio(audio, sr=SR):
    audio = np.asarray(audio, dtype=np.float32)
    if sr != SR or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("expected nonempty finite mono audio at 16000 Hz")
    return np.ascontiguousarray(audio)


class BlindSeparator:
    """Two-stream separator shared by enrolled and blind association candidates."""

    def __init__(self, backend=None, model=None):
        self.backend = backend
        self.model = model  # hearing.models name; None = separator_model() (HEARING_SEPARATOR)
        self.device = getattr(backend, "device", "")

    def load(self):
        if self.backend is None:
            self.backend = models.load(self.model or separator_model())
        self.device = self.backend.device

    def separate(self, mix, sr=SR):
        mix = validate_audio(mix, sr)
        self.load()
        streams = self.backend.separate(mix)
        if len(streams) != 2:
            raise ValueError("separator must return exactly two streams")
        result = np.stack([conform_output(y, len(mix), self.model or separator_model()) for y in streams])
        if not np.isfinite(result).all():
            raise ValueError("separator returned nonfinite audio")
        return rescale_to_mix(result, mix)


def rescale_to_mix(streams, mix):
    """Put separated streams on the mixture's scale: one least-squares gain for all of them, so
    their sum matches ``mix`` and their relative levels are kept.

    MossFormer2 returns streams at an arbitrary level (about 30-36 dB above the input on GRID
    windows, T-061); ClearVoice rescales each stream to the input RMS in its own decode loop,
    which the direct network call here skips. A per-stream RMS match would lift a near-silent
    stream to the mixture's loudness, so the gain is shared."""
    total = streams.sum(axis=0, dtype=np.float64)
    energy = float(np.dot(total, total))
    if energy <= 1e-12:
        return streams
    gain = float(np.dot(np.asarray(mix, np.float64), total)) / energy
    return (streams * gain).astype(np.float32)


def energy_envelope(audio):
    """Mean-square energy, including an unpadded final partial 40 ms frame."""
    return np.array(
        [
            np.mean(part.astype(np.float64) ** 2)
            for part in np.array_split(
                audio, np.arange(FRAME_SAMPLES, len(audio), FRAME_SAMPLES)
            )
        ]
    )


def _correlation(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


@register_candidate("blind_assoc")
class BlindAssocCandidate:
    name = "blind_assoc"
    licence = "Apache-2.0"
    needs = frozenset({"p_speaking"})

    def __init__(self, separator=None):
        self.separator = separator if separator is not None else BlindSeparator()
        self.device = self.separator.device
        self.reset()

    def reset(self):
        self.selected_stream = None
        self.correlations = None

    def load(self):
        self.separator.load()
        self.device = self.separator.device

    def extract(
        self, mix, sr, *, lips=None, lips_fps=25.0, p_speaking=None, enroll=None
    ):
        mix = validate_audio(mix, sr)
        if p_speaking is None:
            raise ValueError("blind_assoc needs p_speaking")
        p = np.asarray(p_speaking, dtype=np.float64)
        if (
            p.ndim != 1
            or not p.size
            or not np.isfinite(p).all()
            or not np.isfinite(lips_fps)
            or lips_fps <= 0
        ):
            raise ValueError(
                "p_speaking must be a finite vector with positive lips_fps"
            )
        streams = self.separator.separate(mix, sr)
        envelopes = [energy_envelope(y) for y in streams]
        aligned = np.interp(
            np.arange(len(envelopes[0])) * 0.04, np.arange(len(p)) / lips_fps, p
        )
        self.correlations = tuple(_correlation(e, aligned) for e in envelopes)
        choice = int(np.argmax(self.correlations))
        if (
            self.selected_stream is None
            or abs(self.correlations[0] - self.correlations[1]) >= 0.1
        ):
            self.selected_stream = choice
        return streams[self.selected_stream].copy()
