"""Offline confidence-weighted identity memory; no runtime/audio integration.

Accepts unit-compatible frozen ECAPA or projected embeddings, not waveforms.
All profiles/candidates must use the same encoder and projection revision.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def _unit(vector):
    x = np.asarray(vector, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('expected a finite nonempty embedding vector')
    norm = np.linalg.norm(x)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('embedding norm must be finite and nonzero')
    return x / norm


@dataclass(frozen=True)
class TrackingResult:
    """index addresses this window only; None means no verified target stem."""

    index: int | None
    alpha: float
    confidence: float
    reason: str


class DynamicEMATracker:
    """Track one locked target without trusting permutation-unstable stem indices.

    EMA ranks candidates; immutable enrollment/negative profiles veto identity
    drift. Speech probability describes the current window; laughter is an
    explicit upstream flag, not inferred here. Rejected windows freeze memory
    but NEVER authorize playback of the previous stem index. Reset on explicit
    target change (construct a new tracker for a different enrollment).
    """

    def __init__(self, target, *, negatives=None, alpha_min=0.02, alpha_max=0.35,
                 margin_min=0.05, margin_full=0.30, identity_floor=0.55,
                 negative_margin=0.10, speech_floor=0.60, reference_hop_s=1.0):
        values = (alpha_min, alpha_max, margin_min, margin_full, identity_floor,
                  negative_margin, speech_floor, reference_hop_s)
        if (not all(math.isfinite(v) for v in values)
                or not 0 <= alpha_min <= alpha_max < 1
                or not 0 < margin_min < margin_full <= 2
                or not -1 <= identity_floor <= 1
                or not 0 <= negative_margin <= 2
                or not 0 <= speech_floor <= 1 or reference_hop_s <= 0):
            raise ValueError('invalid EMA tracker configuration')
        self._target = _unit(target)
        self._negatives = [] if negatives is None else [_unit(n) for n in negatives]
        if any(n.shape != self._target.shape for n in self._negatives):
            raise ValueError('negative profile dimension differs from target')
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.margin_min, self.margin_full = margin_min, margin_full
        self.identity_floor, self.negative_margin = identity_floor, negative_margin
        self.speech_floor, self.reference_hop_s = speech_floor, reference_hop_s
        self.reset()

    @property
    def embedding(self):
        """Defensive copy of normalized identity memory."""
        return _unit(self._memory)

    def reset(self):
        self._memory = self._target.copy()

    def update(self, candidates, *, speech_probability, laughter=False, hop_s=1.0):
        """Consume one completed embedding window; hop_s is new audio duration.

        During explicit non-speech, candidates may be None: no encoder call is
        needed. Pause duration is not accumulated into the next update weight.
        EMA stores the unnormalized accumulator and normalizes for comparisons,
        making repeated identical evidence time-consistent across hop sizes.
        """
        if not math.isfinite(speech_probability) or not 0 <= speech_probability <= 1:
            raise ValueError('speech_probability must be finite and in [0, 1]')
        if not math.isfinite(hop_s) or hop_s <= 0:
            raise ValueError('hop_s must be positive and finite')
        if laughter or speech_probability < self.speech_floor:
            return TrackingResult(None, 0., 0., 'laughter' if laughter else 'silence')
        x = np.asarray(candidates, dtype=np.float64)
        if x.ndim != 2 or not len(x) or x.shape[1:] != self._target.shape:
            raise ValueError('expected nonempty candidates [streams, embedding_dim]')
        x = np.stack([_unit(row) for row in x])
        scores = x @ self.embedding
        best = int(np.argmax(scores))
        # A single stream has no competing-stem evidence. The absolute identity
        # floor and any known negatives still apply; use maximal stream margin.
        margin = (float(scores[best] - np.max(np.delete(scores, best)))
                  if len(x) > 1 else self.margin_full)
        anchor_score = float(x[best] @ self._target)
        if anchor_score < self.identity_floor:
            return TrackingResult(None, 0., 0., 'identity_floor')
        if self._negatives:
            contrast = anchor_score - max(float(x[best] @ n) for n in self._negatives)
            if contrast < self.negative_margin:
                return TrackingResult(None, 0., 0., 'negative_margin')
        if margin < self.margin_min:
            return TrackingResult(None, 0., 0., 'stream_margin')
        confidence = min(1., (margin - self.margin_min) / (self.margin_full - self.margin_min))
        base = self.alpha_min + (self.alpha_max - self.alpha_min) * confidence
        alpha = -math.expm1(math.log1p(-base) * hop_s / self.reference_hop_s)
        mixed = (1 - alpha) * self._memory + alpha * x[best]
        # Floor >= -1 is configurable; antipodal/cancelled updates carry no identity.
        if np.linalg.norm(mixed) <= 1e-12:
            return TrackingResult(None, 0., 0., 'cancelled')
        self._memory = mixed
        return TrackingResult(best, alpha, confidence, 'accepted')
