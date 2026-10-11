"""Apparent gender/age from voice (audeering wav2vec2 age-gender) -> ``ApparentEstimate``.

Implements ``hearing.synth.voice_match.VoiceEstimator``. Model:
``audeering/wav2vec2-large-robust-6-ft-age-gender`` (6 transformer layers; licence
CC BY-NC-SA 4.0, non-commercial). Weights are fetched by the loader into
``$HEARING_MODELS_DIR/hf`` on first use, never at import.

Privacy: estimates are only used to pick a TTS voice. Nothing here logs, prints or stores
them (see ``hearing.synth.voice_match``).

``RollingVoiceEstimator`` keeps the last ``window_s`` of *voiced* target speech from 10 ms
chunks and emits an estimate at most every ``every_s`` of input audio.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from hearing import models
from hearing.audio.io import PIPELINE_SR
from hearing.synth.voice_match import MIN_VOICE_SECONDS, ApparentEstimate
from hearing.types import AudioChunk

MODEL_ID = "audeering/wav2vec2-large-robust-6-ft-age-gender"
FULL_CONFIDENCE_S = 3.0
YOUNG_BELOW = 30.0  # years
SENIOR_FROM = 60.0
AGE_SOFTNESS = 6.0  # years: width of the soft bin edges


def _hf_dir() -> Path:
    return models.cache_dir() / "hf"


def weights_available() -> bool:
    """True when the model's snapshot is already in the cache (no network, no download)."""
    snaps = _hf_dir() / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return False
    return any(
        (s / "config.json").exists()
        and ((s / "model.safetensors").exists() or (s / "pytorch_model.bin").exists())
        for s in snaps.iterdir()
    )


available = weights_available  # name voice_match._load_adapter looks for


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def age_bins(age_years: float) -> tuple[float, float, float]:
    """Soft (young, adult, senior) probabilities from a regressed age in years."""
    young = _sigmoid((YOUNG_BELOW - age_years) / AGE_SOFTNESS)
    senior = _sigmoid((age_years - SENIOR_FROM) / AGE_SOFTNESS)
    adult = max(1.0 - young - senior, 0.0)
    total = young + adult + senior
    return (young / total, adult / total, senior / total)


def to_estimate(
    age_years: float, probs: np.ndarray, n_samples: int
) -> ApparentEstimate:
    """Map model outputs (age in years; softmax over female/male/child) to an estimate."""
    p_f, p_m, p_c = (float(p) for p in probs)
    p_female = p_f / max(p_f + p_m, 1e-6)
    margin = abs(2 * p_female - 1)
    length = min(1.0, n_samples / (FULL_CONFIDENCE_S * PIPELINE_SR))
    confidence = margin * (1.0 - p_c) * (0.5 + 0.5 * length)
    return ApparentEstimate(
        p_female=min(max(p_female, 0.0), 1.0),
        p_age=age_bins(age_years),
        confidence=min(max(confidence, 0.0), 1.0),
        source="voice",
    )


class _AgeGenderModel:
    """The loaded network: ``predict(samples) -> (age_years, probs[female, male, child])``."""

    def __init__(self, net, device: str):
        self.net, self.device = net, device

    def predict(self, samples: np.ndarray) -> tuple[float, np.ndarray]:
        import torch

        x = np.asarray(samples, dtype=np.float32)
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)  # processor do_normalize
        with torch.inference_mode():
            t = torch.from_numpy(x)[None].to(self.device)
            _, age, logits = self.net(t)
            probs = torch.softmax(logits, dim=-1)[0].float().cpu().numpy()
        return float(age.item()) * 100.0, probs


def _build_net_class():
    import torch
    from torch import nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

    class _Head(nn.Module):
        def __init__(self, config, n_out):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, n_out)

        def forward(self, x):
            return self.out_proj(torch.tanh(self.dense(self.dropout(x))))

    class AgeGenderNet(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = _Head(config, 1)
            self.gender = _Head(config, 3)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.age(hidden), self.gender(hidden)

    return AgeGenderNet


@models.register_loader("apparent_voice")
def _load() -> _AgeGenderModel:
    from hearing.device import pick

    net_cls = _build_net_class()
    device = pick()
    net = net_cls.from_pretrained(MODEL_ID, cache_dir=str(_hf_dir())).to(device).eval()
    return _AgeGenderModel(net, device)


class VoiceEstimator:
    """``voice_match.VoiceEstimator`` backed by the audeering model (loaded on first use)."""

    def __init__(self, model=None):
        self._model = model  # anything with predict(samples) -> (age_years, probs3)

    def estimate(self, samples: np.ndarray) -> ApparentEstimate | None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(samples) < MIN_VOICE_SECONDS * PIPELINE_SR:
            return None
        if self._model is None:
            self._model = models.load("apparent_voice")
        age, probs = self._model.predict(samples)
        return to_estimate(age, probs, len(samples))


class RollingVoiceEstimator:
    """Voiced-speech ring over 10 ms chunks; an estimate at most every ``every_s`` once full.

    ``every_s`` is measured in input audio time (all chunks, voiced or not), so results do
    not depend on how fast the caller feeds audio. The window must also be at least
    ``min_voiced_ratio`` voiced over the input time it spans, so speech scattered across a
    long silence is not stitched together.
    """

    def __init__(
        self,
        estimator=None,
        window_s: float = 3.0,
        every_s: float = 2.0,
        min_voiced_ratio: float = 0.5,
    ):
        self.estimator = estimator if estimator is not None else VoiceEstimator()
        self.window = int(window_s * PIPELINE_SR)
        self.every = int(every_s * PIPELINE_SR)
        self.min_voiced_ratio = min_voiced_ratio
        self._ring: deque[tuple[int, np.ndarray]] = (
            deque()
        )  # (input index at arrival, samples)
        self._voiced = 0
        self._total = 0
        self._last_emit: int | None = None

    def reset(self) -> None:
        self._ring.clear()
        self._voiced = self._total = 0
        self._last_emit = None

    def update(
        self, chunk: AudioChunk, speaking: bool = True
    ) -> ApparentEstimate | None:
        n = len(chunk.samples)
        if speaking:
            self._ring.append(
                (self._total, np.asarray(chunk.samples, dtype=np.float32))
            )
            self._voiced += n
            while self._voiced - len(self._ring[0][1]) >= self.window:
                self._voiced -= len(self._ring.popleft()[1])
        self._total += n
        if self._voiced < self.window:
            return None
        if self._last_emit is not None and self._total - self._last_emit < self.every:
            return None
        span = self._total - self._ring[0][0]
        if self._voiced < self.min_voiced_ratio * span:
            return None
        self._last_emit = self._total
        samples = np.concatenate([s for _, s in self._ring])[-self.window :]
        return self.estimator.estimate(samples)
