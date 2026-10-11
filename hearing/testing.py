"""Test helpers shared by every stage: synthetic signals and streaming-equivalence checks.

``assert_streaming_equivalent`` is the acceptance test each stage (DFN3, GTCRN, gate,
mixer, …) must pass: processing 10 ms chunks must equal processing the whole signal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from hearing.audio.io import HOP, PIPELINE_SR
from hearing.audio.stream import chunk_stream
from hearing.types import AudioChunk, Stage, StageContext

SignalKind = Literal["speechlike", "tone", "noise"]


def make_test_signal(
    seconds: float, sr: int = PIPELINE_SR, kind: SignalKind = "speechlike", seed: int = 0
) -> np.ndarray:
    """Deterministic float32 mono test signal in [-1, 1].

    * ``speechlike``: amplitude-modulated harmonic bursts (~120 Hz fundamental with
      slow pitch wander, 3-5 Hz syllable envelope) separated by pauses, with a little
      aspiration noise; peaks around 0.8.
    * ``tone``: 1 kHz sine at 0.5 amplitude.
    * ``noise``: white noise at 0.3 RMS, clipped to [-1, 1].
    """
    n = round(seconds * sr)
    t = np.arange(n, dtype=np.float64) / sr
    rng = np.random.default_rng(seed)
    if kind == "tone":
        x = 0.5 * np.sin(2 * np.pi * 1000.0 * t)
    elif kind == "noise":
        x = np.clip(rng.normal(0.0, 0.3, n), -1.0, 1.0)
    elif kind == "speechlike":
        f0 = 120.0 * (1.0 + 0.08 * np.sin(2 * np.pi * 0.7 * t + rng.uniform(0, 2 * np.pi)))
        phase = 2 * np.pi * np.cumsum(f0) / sr
        harmonics = np.zeros(n)
        for k in range(1, 9):
            harmonics += np.sin(k * phase + rng.uniform(0, 2 * np.pi)) / k
        harmonics /= np.max(np.abs(harmonics)) or 1.0
        # Syllable rate 3-5 Hz: a raised cosine, squared to sharpen onsets.
        syllable = 0.5 * (1 + np.sin(2 * np.pi * (4.0 + rng.uniform(-1, 1)) * t)) ** 2
        # Pauses: ~300 ms words gated on with probability 0.7 in a fixed pattern.
        words = rng.random(max(1, int(np.ceil(seconds / 0.3)) + 1)) < 0.7
        gate = np.repeat(words, int(0.3 * sr))[:n].astype(np.float64)
        # Smooth the gate over 10 ms so there are no clicks.
        win = int(0.01 * sr)
        if win > 1:
            gate = np.convolve(gate, np.ones(win) / win, mode="same")
        noise = rng.normal(0.0, 0.02, n)
        x = 0.8 * gate * (syllable * harmonics + noise)
        x = np.clip(x, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown signal kind {kind!r}; use 'speechlike', 'tone' or 'noise'")
    return x.astype(np.float32)


def run_whole(stage: Stage, x: np.ndarray, sr: int, ctx: StageContext | None = None) -> np.ndarray:
    """Process ``x`` as a single chunk and return the float32 output samples."""
    ctx = ctx or StageContext.default()
    chunk = AudioChunk(np.asarray(x, dtype=np.float32), sr, 0, 0.0)
    return np.asarray(stage.process(chunk, ctx).samples, dtype=np.float32)


def run_streaming(
    stage: Stage, x: np.ndarray, sr: int, hop: int = HOP, ctx: StageContext | None = None
) -> np.ndarray:
    """Process ``x`` in ``hop``-sample chunks, concatenate and trim to ``len(x)``."""
    ctx = ctx or StageContext.default()
    outs = [
        np.asarray(stage.process(chunk, ctx).samples, dtype=np.float32)
        for chunk in chunk_stream(x, sr, hop_ms=1000.0 * hop / sr)
    ]
    out = np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)
    # The last chunk may be zero-padded; drop the padding so lengths compare.
    pad = len(outs) * hop - len(x)
    if pad > 0 and len(out) >= pad:
        out = out[: len(out) - pad]
    return out


def _factory(factory: str | Callable[[], Stage]) -> Callable[[], Stage]:
    if isinstance(factory, str):
        from hearing import registry

        return lambda: registry.build_pipeline(factory)
    return factory


def assert_streaming_equivalent(
    factory: str | Callable[[], Stage],
    x: np.ndarray,
    sr: int = PIPELINE_SR,
    hop: int = HOP,
    atol: float = 1e-5,
    ctx: StageContext | None = None,
) -> None:
    """Assert that a fresh stage gives the same output chunked as it does whole-file.

    ``factory`` is a zero-argument callable (a stage class or lambda) or a pipeline spec
    string. It is called twice so the two runs never share state. On failure the
    message shows the max abs error and the index/values of the first differing sample.
    """
    make = _factory(factory)
    whole = run_whole(make(), x, sr, ctx)
    streamed = run_streaming(make(), x, sr, hop, ctx)
    name = getattr(make(), "name", repr(factory))
    if len(whole) != len(streamed):
        raise AssertionError(
            f"{name}: streaming output length {len(streamed)} != whole-file length {len(whole)} "
            f"(input {len(x)} samples, hop {hop})"
        )
    diff = np.abs(whole.astype(np.float64) - streamed.astype(np.float64))
    if not len(diff):
        return
    max_err = float(diff.max())
    if max_err > atol:
        bad = np.flatnonzero(diff > atol)
        i = int(bad[0])
        raise AssertionError(
            f"{name}: chunked (hop={hop}) and whole-file outputs differ: "
            f"max abs error {max_err:.3e} at sample {int(diff.argmax())} (atol {atol:g}); "
            f"first differing sample {i} (t={i / sr:.4f}s, chunk {i // hop}): "
            f"whole={whole[i]:.6g} streamed={streamed[i]:.6g}; {len(bad)} samples differ"
        )
