"""GTCRN DNS3 streaming ONNX adapter, 16 kHz in/out, arbitrary chunk boundaries.

Implements the spectral interface published in Xiaobin-Rong/gtcrn's
stream/gtcrn_stream.py: 512-point sqrt-periodic-Hann STFT, 256-sample hop,
and explicit convolution/attention/GRU caches. No PyTorch dependency.

Import this module before registry lookup. Call ``download_model()`` explicitly
once to provision weights; processing never downloads. Install ``.[ml]`` first.
The default backend fails visibly when unavailable, letting the tier controller
choose a fallback. ``MockBackend`` is an explicit offline identity-spectrum mock.

Output has exactly the input length, with 512 samples (32 ms) causal delay.
Append 512 zeros to drain the tail. Suppression > 0 mixes time-aligned dry/wet;
zero is immediate, exact bypass while internal state continues advancing.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np

from hearing import device, models, registry
from hearing.types import AudioChunk, StageContext

REVISION = "502ebfab64da7c4a9af78dcb9c6ceef1ebb01c73"
MODEL_URL = (
    f"https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/{REVISION}/"
    "stream/onnx_models/gtcrn_simple.onnx"
)
MODEL_SHA256 = "b4718df6228e7bdf1a8a435cf98f838636eb2fd331acabf86ba87c5192ebcb87"
HOP = 256
FFT_SIZE = 512
STATE_SHAPES = ((2, 1, 16, 16, 33), (2, 3, 1, 1, 16), (2, 1, 33, 16))
STATE_NAMES = ("conv_cache", "tra_cache", "inter_cache")


def model_path() -> Path:
    return models.cache_dir() / "gtcrn" / "gtcrn_simple.onnx"


def _verify(data: bytes) -> None:
    if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
        raise ValueError(
            "GTCRN model checksum mismatch; re-provision the pinned weights"
        )


def download_model() -> Path:
    """Explicit network adapter; pinned revision/checksum and atomic cache install."""
    path = model_path()
    if path.exists():
        _verify(path.read_bytes())
        return path
    import httpx  # httpx verifies against certifi; python.org builds lack a CA bundle.

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


class Backend(Protocol):
    device: str

    def infer(
        self, spectrum: np.ndarray, state: list[np.ndarray]
    ) -> tuple[np.ndarray, list[np.ndarray]]: ...


class MockBackend:
    """Deterministic identity spectrum, for offline plumbing tests only."""

    device = "mock"

    def infer(self, spectrum, state):
        return spectrum.copy(), [s.copy() for s in state]


class OnnxBackend:
    """Immutable shared session; all mutable audio/model state belongs to the stage."""

    def __init__(self):
        path = model_path()
        if not path.exists():
            raise FileNotFoundError(
                f"{path}: provision with hearing.enhance.gtcrn.download_model()"
            )
        _verify(path.read_bytes())
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "GTCRN requires onnxruntime; install hearing[ml]"
            ) from exc
        requested = device.pick()
        providers = ["CPUExecutionProvider"]
        if (
            requested == "cuda"
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")
        # ORT has no MPS provider; CPU is the explicit fallback for this small graph.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.device = (
            "cuda"
            if self.session.get_providers()[0] == "CUDAExecutionProvider"
            else "cpu"
        )

    def infer(self, spectrum, state):
        feed = dict(zip(STATE_NAMES, state, strict=True))
        feed["mix"] = spectrum
        output = self.session.run(["enh", *(n + "_out" for n in STATE_NAMES)], feed)
        return output[0], output[1:]


@models.register_loader("gtcrn")
def _load_backend():
    return OnnxBackend()


@registry.register("gtcrn")
class GTCRN:
    name = "gtcrn"
    sample_rate = 16000
    latency_ms = 32.0

    def __init__(self, *, backend: Backend | None = None):
        self._backend = backend
        self._window = np.sqrt(
            0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FFT_SIZE) / FFT_SIZE)
        )
        self.reset()

    def reset(self) -> None:
        self._state = [np.zeros(shape, np.float32) for shape in STATE_SHAPES]
        self._pending = np.zeros(0, np.float32)
        self._previous = np.zeros(HOP, np.float32)
        self._overlap = np.zeros(HOP)
        self._output = np.zeros(HOP, np.float32)
        self._dry = np.zeros(FFT_SIZE, np.float32)
        self._first = True

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(np.concatenate((self._previous, frame)) * self._window)
        packed = np.stack((spec.real, spec.imag), axis=-1)[None, :, None, :].astype(
            np.float32
        )
        enhanced, self._state = self._backend.infer(packed, self._state)
        spectrum = enhanced[0, :, 0, 0] + 1j * enhanced[0, :, 0, 1]
        wave = np.fft.irfft(spectrum, n=FFT_SIZE) * self._window
        out = wave[:HOP] + self._overlap
        self._overlap = wave[HOP:]
        self._previous = frame.copy()
        if self._first:  # Discard the synthesized negative-time half-frame.
            out = np.zeros(HOP)
            self._first = False
        return out.astype(np.float32)

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        x = np.asarray(chunk.samples, dtype=np.float32)
        if (
            chunk.sample_rate != self.sample_rate
            or x.ndim != 1
            or not np.isfinite(x).all()
        ):
            raise ValueError("GTCRN expects finite mono audio at 16000 Hz")
        strength = float(ctx.suppression)
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("suppression must be in [0, 1]")
        if not len(x):
            return replace(chunk, samples=x.copy())
        if self._backend is None:
            self._backend = models.load("gtcrn")
        ctx.telemetry.extras["gtcrn"] = {
            "device": self._backend.device,
            "mock": isinstance(self._backend, MockBackend),
        }
        pending = np.concatenate((self._pending, x))
        count = len(pending) // HOP
        frames = [self._frame(pending[i * HOP : (i + 1) * HOP]) for i in range(count)]
        self._pending = pending[count * HOP :].copy()
        output = np.concatenate((self._output, *frames))
        wet = output[: len(x)]
        self._output = output[len(x) :].copy()
        dry = np.concatenate((self._dry, x))
        self._dry = dry[len(x) :].copy()
        mixed = (
            x.copy()
            if strength == 0
            else (1 - strength) * dry[: len(x)] + strength * wet
        )
        return replace(chunk, samples=np.clip(mixed, -1, 1).astype(np.float32))


def benchmark(
    seconds: float = 3.0, snrs_db: tuple[float, ...] = (0.0, 5.0, 10.0)
) -> dict:
    """Tiny reproducible benchmark: 10 ms streaming at full suppression, one row per SNR.

    Speech-like synthetic target plus white noise; stands in for ``eval.bench --set tiny``
    until that runner exists. Run from the repo root (imports ``eval.metrics``).
    """
    import platform
    import time

    from eval.metrics import LatencyStats, si_sdri
    from hearing.testing import make_test_signal

    ctx = StageContext.default()
    ctx.suppression = 1.0
    stage = registry.get("gtcrn")
    # Warm up so the cold session load is excluded from steady-state RTF.
    stage.process(AudioChunk(np.zeros(1024, np.float32), 16000, 0, 0.0), ctx)
    rows, timings_ms = [], []
    for i, snr in enumerate(snrs_db):
        clean = make_test_signal(seconds, kind="speechlike", seed=7 + i) * 0.5
        noise = np.random.default_rng(42 + i).normal(0, 1, len(clean))
        noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2) / 10 ** (snr / 10))
        mix = np.clip(clean + noise, -1, 1).astype(np.float32)
        # Append FFT_SIZE zeros to drain the adapter delay.
        padded = np.concatenate((mix, np.zeros(FFT_SIZE, np.float32)))
        stage.reset()
        outs, compute = [], 0.0
        for start in range(0, len(padded), 160):
            chunk = AudioChunk(
                padded[start : start + 160], 16000, start // 160, start / 16000
            )
            before = time.perf_counter()
            outs.append(stage.process(chunk, ctx).samples)
            elapsed = time.perf_counter() - before
            compute += elapsed
            timings_ms.append(elapsed * 1000)
        enhanced = np.concatenate(outs)[FFT_SIZE : FFT_SIZE + len(mix)]
        rows.append(
            {
                "snr_db": snr,
                "si_sdri": si_sdri(enhanced, mix, clean),
                "rtf": compute / (len(padded) / 16000),
            }
        )
    stats = LatencyStats.from_ms(timings_ms)
    return {
        "stage": "gtcrn",
        "set": f"synthetic-tiny ({len(snrs_db)} x {seconds:g} s speechlike + white noise)",
        "device": ctx.telemetry.extras["gtcrn"]["device"],
        "machine": f"{platform.machine()} {platform.platform()}",
        "latency_ms": stage.latency_ms,
        "compute_ms_p50": stats.p50,
        "compute_ms_p95": stats.p95,
        "rtf": float(np.mean([r["rtf"] for r in rows])),
        "si_sdri": float(np.mean([r["si_sdri"] for r in rows])),
        "rows": rows,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(benchmark(), indent=2))
