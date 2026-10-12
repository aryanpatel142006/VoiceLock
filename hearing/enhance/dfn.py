"""DeepFilterNet3 streaming stages ``dfn`` and ``dfn_ll``: 16 kHz in/out, 48 kHz inside.

Pipeline per 10 ms hop (one 480-sample frame at 48 kHz):

1. polyphase FIR upsampling 16 kHz -> 48 kHz (explicit filter state, 1.5 ms delay);
2. 960-point vorbis-window STFT with 480 hop (the ``DFState`` of upstream ``libDF``);
3. ERB (32 bands, exponential mean norm) and complex (96 bins, unit norm) features;
4. the ``enc`` / ``erb_dec`` / ``df_dec`` ONNX graphs rewritten for per-hop inference with
   explicit GRU and conv state (:mod:`hearing.enhance.dfn_graphs`), run with onnxruntime;
5. ERB mask on the frame ``lookahead`` hops back, deep filtering of the lowest 96 bins
   over ``df_order`` noisy frames, attenuation limit mixing (slider via
   :func:`hearing.fusion.mixer.strength_to_db`), overlap-add synthesis;
6. polyphase FIR downsampling 48 kHz -> 16 kHz (1.5 ms delay).

The output has exactly the input length; the first ``latency_ms`` of output are the
stage's algorithmic delay (STFT hop + model lookahead + resamplers), reported in
``latency_ms``. Chunk lengths must be multiples of 160 samples (10 ms); that is the
only case: chunks must be whole hops, so output lags input by a constant and chunked and
whole-file processing agree to float precision.

Call ``hearing.enhance.dfn_graphs.provision(variant)`` once to download the pinned ONNX
export (7.9 MB / 36 MB); processing never downloads. ``MockBackend`` is an offline identity
model (unit mask, delta DF taps) for plumbing tests.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from hearing import device, models, registry
from hearing.audio.io import PIPELINE_SR
from hearing.fusion.mixer import SLIDER_RAMP_MS, atten_lim_to_mix, strength_to_db
from hearing.types import AudioChunk, StageContext

MEAN_NORM_INIT = (-60.0, -90.0)
UNIT_NORM_INIT = (0.001, 0.0001)
RESAMPLE_TAPS = 145  # odd, (taps - 1) divisible by 3: integer total delay at 16 kHz
RESAMPLE_PASSBAND_HZ = 6800.0
RESAMPLE_STOPBAND_HZ = 8000.0


@dataclass(frozen=True)
class DfConfig:
    """The ``[df]`` / ``[deepfilternet]`` settings the DSP side depends on."""

    sr: int = 48000
    fft_size: int = 960
    hop_size: int = 480
    nb_erb: int = 32
    nb_df: int = 96
    min_nb_erb_freqs: int = 2
    df_order: int = 5
    df_lookahead: int = 2
    conv_lookahead: int = 2
    norm_tau: float = 1.0

    @property
    def lookahead(self) -> int:
        return max(self.conv_lookahead, self.df_lookahead)

    @property
    def n_freqs(self) -> int:
        return self.fft_size // 2 + 1

    @classmethod
    def from_manifest(cls, manifest: dict) -> DfConfig:
        df, net = manifest["df"], manifest["deepfilternet"]
        return cls(
            sr=int(df["sr"]),
            fft_size=int(df["fft_size"]),
            hop_size=int(df["hop_size"]),
            nb_erb=int(df["nb_erb"]),
            nb_df=int(df["nb_df"]),
            min_nb_erb_freqs=int(df["min_nb_erb_freqs"]),
            df_order=int(df.get("df_order", net.get("df_order", 5))),
            df_lookahead=int(df.get("df_lookahead", net.get("df_lookahead", 0))),
            conv_lookahead=int(net["conv_lookahead"]),
            norm_tau=float(df["norm_tau"]),
        )


# Shipped configs, so the mock backend and latency numbers need no files.
CONFIGS = {
    "dfn": DfConfig(df_lookahead=2, conv_lookahead=2),
    "dfn_ll": DfConfig(df_lookahead=0, conv_lookahead=0),
}


# --- DSP (ports of libDF/src/lib.rs) ------------------------------------------------------


def freq2erb(freq_hz: float) -> float:
    return 9.265 * math.log1p(freq_hz / (24.7 * 9.265))


def erb2freq(n_erb: float) -> float:
    return 24.7 * 9.265 * (math.exp(n_erb / 9.265) - 1.0)


def erb_widths(sr: int, fft_size: int, nb_bands: int, min_nb_freqs: int) -> np.ndarray:
    """Number of FFT bins per ERB band (sums to ``fft_size // 2 + 1``)."""
    freq_width = sr / fft_size
    erb_low, erb_high = freq2erb(0.0), freq2erb(sr / 2)
    step = (erb_high - erb_low) / nb_bands
    widths = np.zeros(nb_bands, dtype=np.int64)
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = erb2freq(erb_low + i * step)
        fb = round(f / freq_width)
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        widths[i - 1] = nb_freqs
        prev_freq = fb
    widths[-1] += 1
    too_large = int(widths.sum()) - (fft_size // 2 + 1)
    if too_large > 0:
        widths[-1] -= too_large
    assert int(widths.sum()) == fft_size // 2 + 1
    return widths


def norm_alpha(sr: int, hop_size: int, tau: float) -> float:
    """Exponential-norm smoothing factor, rounded like upstream (0.99 for 10 ms, tau 1 s)."""
    alpha = math.exp(-(hop_size / sr) / tau)
    a, precision = 1.0, 3
    while a >= 1.0:
        a = round(alpha, precision)
        precision += 1
    return a


def vorbis_window(n: int) -> np.ndarray:
    i = np.arange(n, dtype=np.float64)
    s = np.sin(0.5 * np.pi * (i + 0.5) / (n / 2))
    return np.sin(0.5 * np.pi * s * s).astype(np.float32)


class DfState:
    """STFT analysis/synthesis and feature-normalisation state for one channel."""

    def __init__(self, cfg: DfConfig):
        if cfg.hop_size * 2 != cfg.fft_size:
            raise ValueError(
                "DfState implements the 50 % overlap of the shipped models"
            )
        self.cfg = cfg
        self.window = vorbis_window(cfg.fft_size)
        self.wnorm = np.float32(1.0 / (cfg.fft_size**2 / (2 * cfg.hop_size)))
        self.widths = erb_widths(cfg.sr, cfg.fft_size, cfg.nb_erb, cfg.min_nb_erb_freqs)
        self.band_starts = np.concatenate(([0], np.cumsum(self.widths)[:-1]))
        self.alpha = np.float32(norm_alpha(cfg.sr, cfg.hop_size, cfg.norm_tau))
        self.reset()

    def reset(self) -> None:
        cfg = self.cfg
        self.analysis_mem = np.zeros(cfg.fft_size - cfg.hop_size, dtype=np.float32)
        self.synthesis_mem = np.zeros(cfg.fft_size - cfg.hop_size, dtype=np.float32)
        self.mean_norm_state = np.linspace(
            *MEAN_NORM_INIT, cfg.nb_erb, dtype=np.float32
        )
        self.unit_norm_state = np.linspace(*UNIT_NORM_INIT, cfg.nb_df, dtype=np.float32)

    def analysis(self, frame: np.ndarray) -> np.ndarray:
        """480 new samples -> normalised complex spectrum of the last 960 samples."""
        buf = np.concatenate((self.analysis_mem, frame)) * self.window
        self.analysis_mem = frame.astype(np.float32, copy=True)
        return (np.fft.rfft(buf) * self.wnorm).astype(np.complex64)

    def synthesis(self, spec: np.ndarray) -> np.ndarray:
        """Complex spectrum -> 480 output samples (overlap-add with the previous frame)."""
        x = np.fft.irfft(spec, n=self.cfg.fft_size) * self.cfg.fft_size * self.window
        hop = self.cfg.hop_size
        out = x[:hop] + self.synthesis_mem
        self.synthesis_mem = x[hop:].astype(np.float32)
        return out.astype(np.float32)

    def feat_erb(self, spec: np.ndarray) -> np.ndarray:
        power = spec.real.astype(np.float64) ** 2 + spec.imag.astype(np.float64) ** 2
        bands = np.add.reduceat(power, self.band_starts) / self.widths
        feat = (10.0 * np.log10(bands + 1e-10)).astype(np.float32)
        self.mean_norm_state = (
            feat * (1 - self.alpha) + self.mean_norm_state * self.alpha
        )
        return (feat - self.mean_norm_state) / np.float32(40.0)

    def feat_cplx(self, spec: np.ndarray) -> np.ndarray:
        """Unit-normalised lowest ``nb_df`` bins as a ``[2, nb_df]`` (re, im) array."""
        low = spec[: self.cfg.nb_df]
        mag = np.abs(low).astype(np.float32)
        self.unit_norm_state = (
            mag * (1 - self.alpha) + self.unit_norm_state * self.alpha
        )
        scaled = low / np.sqrt(self.unit_norm_state)
        return np.stack((scaled.real, scaled.imag)).astype(np.float32)

    def expand_gains(self, gains: np.ndarray) -> np.ndarray:
        return np.repeat(gains.astype(np.float32), self.widths)


# --- resampling ---------------------------------------------------------------------------


def design_lowpass(taps: int = RESAMPLE_TAPS, fs: int = 48000) -> np.ndarray:
    """Kaiser-windowed sinc with pass 6.8 kHz / stop 8 kHz at 48 kHz (~60 dB)."""
    from scipy.signal import firwin, kaiser_beta

    width = (RESAMPLE_STOPBAND_HZ - RESAMPLE_PASSBAND_HZ) / (fs / 2)
    attenuation = 2.285 * (taps - 1) * np.pi * width + 7.95
    cutoff = (RESAMPLE_PASSBAND_HZ + RESAMPLE_STOPBAND_HZ) / 2
    h = firwin(taps, cutoff, window=("kaiser", kaiser_beta(attenuation)), fs=fs)
    return h.astype(np.float64)


class Upsampler:
    """Zero-stuff by ``factor`` and low-pass, as a polyphase FIR with explicit history."""

    def __init__(self, factor: int, taps: np.ndarray):
        self.factor = factor
        self.phases = [
            np.pad(
                taps[p::factor],
                (0, math.ceil(len(taps) / factor) - len(taps[p::factor])),
            )
            * factor
            for p in range(factor)
        ]
        self.history = len(self.phases[0]) - 1
        self.delay = (len(taps) - 1) // 2  # samples at the output rate
        self.reset()

    def reset(self) -> None:
        self._mem = np.zeros(self.history, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        if not len(x):
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate((self._mem, x.astype(np.float64)))
        self._mem = buf[len(buf) - self.history :].copy()
        out = np.empty(len(x) * self.factor, dtype=np.float64)
        for p, h in enumerate(self.phases):
            out[p :: self.factor] = np.convolve(buf, h, mode="valid")
        return out.astype(np.float32)


class Downsampler:
    """Low-pass and keep every ``factor``-th sample; input lengths must be multiples."""

    def __init__(self, factor: int, taps: np.ndarray):
        self.factor = factor
        self.taps = np.asarray(taps, dtype=np.float64)
        self.history = len(taps) - 1
        self.delay = (len(taps) - 1) // 2  # samples at the input rate
        self.reset()

    def reset(self) -> None:
        self._mem = np.zeros(self.history, dtype=np.float64)

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) % self.factor:
            raise ValueError(
                f"downsampler input length {len(x)} is not a multiple of {self.factor}"
            )
        if not len(x):
            return np.zeros(0, dtype=np.float32)
        buf = np.concatenate((self._mem, x.astype(np.float64)))
        self._mem = buf[len(buf) - self.history :].copy()
        return np.convolve(buf, self.taps, mode="valid")[:: self.factor].astype(
            np.float32
        )


# --- backends -----------------------------------------------------------------------------


class Backend(Protocol):
    device: str
    cfg: DfConfig

    def initial_state(self) -> dict[str, np.ndarray]: ...

    def infer(
        self, feat_erb: np.ndarray, feat_spec: np.ndarray, state: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, np.ndarray]]:
        """``feat_erb [nb_erb]``, ``feat_spec [2, nb_df]`` -> (gains, coefs, lsnr, state).

        ``gains`` is ``[nb_erb]`` in [0, 1]; ``coefs`` is complex ``[nb_df, df_order]``
        ordered oldest frame first; ``lsnr`` the local SNR estimate in dB.
        """
        ...


class MockBackend:
    """Identity model: unit ERB mask, delta DF taps. For offline plumbing tests only."""

    device = "mock"

    def __init__(self, variant: str = "dfn", gain: float = 1.0):
        self.cfg = CONFIGS[variant]
        self.gain = float(gain)
        self.calls = 0

    def initial_state(self) -> dict[str, np.ndarray]:
        return {"frames": np.zeros(1, dtype=np.float32)}

    def infer(self, feat_erb, feat_spec, state):
        cfg = self.cfg
        self.calls += 1
        gains = np.full(cfg.nb_erb, self.gain, dtype=np.float32)
        coefs = np.zeros((cfg.nb_df, cfg.df_order), dtype=np.complex64)
        coefs[:, cfg.df_order - 1 - cfg.df_lookahead] = self.gain
        return gains, coefs, 35.0, {"frames": state["frames"] + 1}


class OnnxBackend:
    """Three immutable onnxruntime sessions; all recurrent state travels with the stage."""

    def __init__(self, variant: str = "dfn"):
        from hearing.enhance import dfn_graphs

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "DeepFilterNet requires onnxruntime; install hearing[ml]"
            ) from exc
        self.variant = variant
        self.manifest = dfn_graphs.provision(variant, download=False)
        self.cfg = DfConfig.from_manifest(self.manifest)
        if self.cfg.conv_lookahead != self.cfg.df_lookahead:
            raise ValueError("DfTract semantics need conv_lookahead == df_lookahead")
        requested = device.pick()
        providers = ["CPUExecutionProvider"]
        if (
            requested == "cuda"
            and "CUDAExecutionProvider" in ort.get_available_providers()
        ):
            providers.insert(0, "CUDAExecutionProvider")
        # onnxruntime has no MPS provider; the graphs are small enough for one CPU thread.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        paths = dfn_graphs.streaming_paths(variant)
        self.sessions = {
            g: ort.InferenceSession(str(paths[g]), options, providers=providers)
            for g in dfn_graphs.GRAPHS
        }
        self.device = (
            "cuda"
            if self.sessions["enc"].get_providers()[0] == "CUDAExecutionProvider"
            else "cpu"
        )
        self.states = {
            g: [
                (name, out, tuple(shape))
                for name, out, shape in self.manifest["states"][g]
            ]
            for g in dfn_graphs.GRAPHS
        }
        self._enc_outputs = ["e0", "e1", "e2", "e3", "emb", "c0", "lsnr"]

    def initial_state(self) -> dict[str, np.ndarray]:
        return {
            name: np.zeros(shape, dtype=np.float32)
            for specs in self.states.values()
            for name, _, shape in specs
        }

    def _run(
        self, graph: str, feeds: dict, outputs: list[str], state: dict, new_state: dict
    ):
        specs = self.states[graph]
        feeds.update({name: state[name] for name, _, _ in specs})
        result = self.sessions[graph].run(outputs + [out for _, out, _ in specs], feeds)
        for (name, _, _), value in zip(specs, result[len(outputs) :], strict=True):
            new_state[name] = value
        return result[: len(outputs)]

    def infer(self, feat_erb, feat_spec, state):
        cfg = self.cfg
        new_state: dict[str, np.ndarray] = {}
        feeds = {
            "feat_erb": feat_erb.reshape(1, 1, 1, cfg.nb_erb).astype(
                np.float32, copy=False
            ),
            "feat_spec": feat_spec.reshape(1, 2, 1, cfg.nb_df).astype(
                np.float32, copy=False
            ),
        }
        e0, e1, e2, e3, emb, c0, lsnr = self._run(
            "enc", feeds, self._enc_outputs, state, new_state
        )
        (m,) = self._run(
            "erb_dec",
            {"emb": emb, "e3": e3, "e2": e2, "e1": e1, "e0": e0},
            ["m"],
            state,
            new_state,
        )
        (coefs,) = self._run(
            "df_dec", {"emb": emb, "c0": c0}, ["coefs"], state, new_state
        )
        gains = m.reshape(cfg.nb_erb)
        taps = coefs.reshape(cfg.nb_df, cfg.df_order, 2)
        complex_taps = (taps[..., 0] + 1j * taps[..., 1]).astype(np.complex64)
        return gains, complex_taps, float(lsnr.reshape(-1)[0]), new_state


@models.register_loader("dfn")
def _load_dfn():
    return OnnxBackend("dfn")


@models.register_loader("dfn_ll")
def _load_dfn_ll():
    return OnnxBackend("dfn_ll")


# --- the stage ----------------------------------------------------------------------------


class DFN:
    """DeepFilterNet3 streaming stage (see module docstring)."""

    sample_rate = PIPELINE_SR

    def __init__(
        self,
        variant: str = "dfn",
        *,
        backend: Backend | None = None,
        strength: float | None = None,
        post_filter: bool = False,
    ):
        if variant not in CONFIGS:
            raise ValueError(
                f"unknown DeepFilterNet variant {variant!r}; use {sorted(CONFIGS)}"
            )
        self.variant = variant
        self.name = variant
        self._backend = backend
        self.cfg = backend.cfg if backend is not None else CONFIGS[variant]
        self.strength = None if strength is None else float(strength)
        self.post_filter = bool(post_filter)
        if self.cfg.sr % self.sample_rate:
            raise ValueError("internal rate must be an integer multiple of 16 kHz")
        self.factor = self.cfg.sr // self.sample_rate
        if self.cfg.hop_size % self.factor:
            raise ValueError("hop must map to a whole number of 16 kHz samples")
        self.hop = self.cfg.hop_size // self.factor  # 160 samples at 16 kHz
        taps = design_lowpass(RESAMPLE_TAPS, self.cfg.sr)
        self._up = Upsampler(self.factor, taps)
        self._down = Downsampler(self.factor, taps)
        self._state = DfState(self.cfg)
        self.latency_samples = (
            self.cfg.hop_size * (1 + self.cfg.lookahead)
            + self._up.delay
            + self._down.delay
        ) // self.factor
        self.latency_ms = 1000.0 * self.latency_samples / self.sample_rate
        self.reset()

    def reset(self) -> None:
        self._up.reset()
        self._down.reset()
        self._state.reset()
        self._spectra = deque(
            [
                np.zeros(self.cfg.n_freqs, dtype=np.complex64)
                for _ in range(self.cfg.df_order)
            ],
            maxlen=self.cfg.df_order,
        )
        self._model_state = None
        self.lsnr_db = math.nan
        self.lsnr_hops = np.zeros(0, np.float32)
        self._mix: float | None = None  # mix-back in effect, ramped toward the slider's

    def _backend_ready(self) -> Backend:
        if self._backend is None:
            self._backend = models.load(self.variant)
        if self._model_state is None:
            self._model_state = self._backend.initial_state()
        return self._backend

    def _frame(self, frame: np.ndarray, mix: float) -> np.ndarray:
        """One 480-sample hop at 48 kHz -> 480 enhanced samples (delayed by lookahead hops)."""
        cfg = self.cfg
        backend = self._backend_ready()
        spec = self._state.analysis(frame)
        self._spectra.append(spec)
        feat_erb = self._state.feat_erb(spec)
        feat_spec = self._state.feat_cplx(spec)
        gains, coefs, self.lsnr_db, self._model_state = backend.infer(
            feat_erb, feat_spec, self._model_state
        )
        noisy = self._spectra[cfg.df_order - 1 - cfg.lookahead]
        if mix >= 1.0:
            return self._state.synthesis(noisy)
        enhanced = noisy * self._state.expand_gains(gains)
        low = np.zeros(cfg.nb_df, dtype=np.complex64)
        for i, past in enumerate(self._spectra):
            low += past[: cfg.nb_df] * coefs[:, i]
        enhanced[: cfg.nb_df] = low
        if self.post_filter:
            enhanced = _post_filter(noisy, enhanced)
        if mix > 0.0:
            enhanced = mix * noisy + (1.0 - mix) * enhanced
        return self._state.synthesis(enhanced)

    def _ramp_mix(self, goal: float) -> float:
        """Move the mix-back toward ``goal`` by at most one 10 ms hop of a ``SLIDER_RAMP_MS`` swing.

        A slider jump 0 -> 100 then fades over 3 hops (each crossfaded again by the synthesis
        window) instead of switching between two frames, so it never clicks (T-093).
        """
        if self._mix is None:
            self._mix = goal
            return goal
        step = self.cfg.hop_size / self.cfg.sr * 1000.0 / SLIDER_RAMP_MS
        self._mix = min(goal, self._mix + step) if goal > self._mix else max(goal, self._mix - step)
        return self._mix

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        x = np.asarray(chunk.samples, dtype=np.float32)
        if (
            chunk.sample_rate != self.sample_rate
            or x.ndim != 1
            or not np.isfinite(x).all()
        ):
            raise ValueError(
                f"{self.name} expects finite mono audio at {self.sample_rate} Hz"
            )
        strength = ctx.suppression if self.strength is None else self.strength
        limit_db = strength_to_db(strength)
        mix = atten_lim_to_mix(limit_db)
        if not len(x):
            return replace(chunk, samples=x.copy())
        if len(x) % self.hop:
            raise ValueError(
                f"{self.name} needs chunks of whole 10 ms hops ({self.hop} samples), got {len(x)};"
                " re-block with hearing.audio.stream.Chunker"
            )
        backend = self._backend_ready()
        upsampled = self._up.process(x)
        hop = self.cfg.hop_size
        frames, trace = [], []
        for i in range(0, len(upsampled), hop):
            frames.append(self._frame(upsampled[i : i + hop], self._ramp_mix(mix)))
            trace.append(self.lsnr_db)
        self.lsnr_hops = np.asarray(trace, dtype=np.float32)  # the model's local SNR estimate per hop of this chunk
        out = self._down.process(np.concatenate(frames))
        ctx.telemetry.extras[self.name] = {
            "device": backend.device,
            "mock": isinstance(backend, MockBackend),
            "atten_lim_db": limit_db,
            "lsnr_db": None if math.isnan(self.lsnr_db) else round(self.lsnr_db, 1),
        }
        return replace(chunk, samples=np.clip(out, -1.0, 1.0).astype(np.float32))


def _post_filter(
    noisy: np.ndarray, enhanced: np.ndarray, beta: float = 0.02
) -> np.ndarray:
    eps = 1e-12
    g = np.clip(np.abs(enhanced) / (np.abs(noisy) + eps), eps, 1.0)
    g_sin = np.maximum(g * np.sin(np.pi * g / 2), eps)
    pf = (1 + beta) / (1 + beta * (g / g_sin) ** 2)
    return (enhanced * pf).astype(np.complex64)


@registry.register("dfn")
def make_dfn(**kwargs) -> DFN:
    return DFN("dfn", **kwargs)


@registry.register("dfn_ll")
def make_dfn_ll(**kwargs) -> DFN:
    return DFN("dfn_ll", **kwargs)


# --- benchmark ----------------------------------------------------------------------------


def _mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = (
        noise[: len(clean)]
        if len(noise) >= len(clean)
        else np.resize(noise, len(clean))
    )
    noise = noise * np.sqrt(
        np.mean(clean**2) / (np.mean(noise**2) + 1e-12) / 10 ** (snr_db / 10)
    )
    return np.clip(clean + noise, -1, 1).astype(np.float32)


def benchmark(
    variant: str = "dfn",
    clean: list[np.ndarray] | None = None,
    noises: dict[str, np.ndarray] | None = None,
    snrs_db: tuple[float, ...] = (0.0, 5.0, 10.0),
    seconds: float = 3.0,
    strength: float = 1.0,
) -> dict:
    """10 ms streaming benchmark: SI-SDRi, PESQ (wb), STOI, RTF, compute per hop.

    Without ``clean``/``noises`` it uses deterministic speech-like synthetic signals and
    white noise (same recipe as ``hearing.enhance.gtcrn.benchmark``). Pass real 16 kHz
    speech and noise arrays for meaningful quality numbers. Run from the repo root.
    """
    import platform
    import time

    from eval.metrics import LatencyStats, pesq_wb, si_sdri, stoi
    from hearing.testing import make_test_signal

    if clean is None:
        clean = [
            make_test_signal(seconds, kind="speechlike", seed=7 + i) * 0.5
            for i in range(3)
        ]
    if noises is None:
        n = max(len(c) for c in clean)
        noises = {"white": np.random.default_rng(42).normal(0, 1, n).astype(np.float32)}
    ctx = StageContext.default()
    ctx.suppression = strength
    stage = DFN(variant)
    stage.process(
        AudioChunk(np.zeros(1600, np.float32), PIPELINE_SR, 0, 0.0), ctx
    )  # warm up
    hop = 160
    rows, timings_ms = [], []
    for noise_name, noise in noises.items():
        for snr in snrs_db:
            for k, target in enumerate(clean):
                mix = _mix_at_snr(target, noise, snr)
                padded = np.concatenate(
                    (mix, np.zeros(stage.latency_samples, np.float32))
                )
                stage.reset()
                outs, compute = [], 0.0
                for start in range(0, len(padded) - hop + 1, hop):
                    chunk = AudioChunk(
                        padded[start : start + hop], PIPELINE_SR, start // hop, 0.0
                    )
                    before = time.perf_counter()
                    outs.append(stage.process(chunk, ctx).samples)
                    elapsed = time.perf_counter() - before
                    compute += elapsed
                    timings_ms.append(elapsed * 1000)
                enhanced = np.concatenate(outs)[stage.latency_samples :][: len(mix)]
                rows.append(
                    {
                        "noise": noise_name,
                        "snr_db": snr,
                        "utterance": k,
                        "si_sdri": si_sdri(enhanced, mix, target[: len(enhanced)]),
                        "pesq_noisy": pesq_wb(mix, target, PIPELINE_SR),
                        "pesq": pesq_wb(enhanced, target[: len(enhanced)], PIPELINE_SR),
                        "stoi_noisy": stoi(mix, target, PIPELINE_SR),
                        "stoi": stoi(enhanced, target[: len(enhanced)], PIPELINE_SR),
                        "rtf": compute / (len(padded) / PIPELINE_SR),
                    }
                )
    stats = LatencyStats.from_ms(timings_ms)

    def mean(key):
        vals = [
            r[key]
            for r in rows
            if not (isinstance(r[key], float) and math.isnan(r[key]))
        ]
        return float(np.mean(vals)) if vals else math.nan

    by_snr = {
        str(snr): float(np.mean([r["si_sdri"] for r in rows if r["snr_db"] == snr]))
        for snr in snrs_db
    }
    return {
        "stage": variant,
        "set": f"{len(clean)} clean x {list(noises)} x SNR {list(snrs_db)} dB, 10 ms streaming",
        "device": ctx.telemetry.extras[variant]["device"],
        "machine": f"{platform.machine()} {platform.platform()}",
        "latency_ms": stage.latency_ms,
        "compute_ms_p50": stats.p50,
        "compute_ms_p95": stats.p95,
        "rtf": mean("rtf"),
        "si_sdri": mean("si_sdri"),
        "si_sdri_by_snr": by_snr,
        "pesq_noisy": mean("pesq_noisy"),
        "pesq": mean("pesq"),
        "stoi_noisy": mean("stoi_noisy"),
        "stoi": mean("stoi"),
        "rows": rows,
    }


def _load_speech(directory: str, seconds: float, limit: int = 6) -> list[np.ndarray]:
    """Clean 16 kHz utterances (``.flac``/``.wav``, recursive), trimmed to ``seconds``."""
    import glob

    from hearing.audio.io import load

    files = sorted(
        glob.glob(f"{directory}/**/*.flac", recursive=True)
        + glob.glob(f"{directory}/**/*.wav", recursive=True)
    )[:limit]
    clean = []
    for path in files:
        x, _ = load(path, PIPELINE_SR)
        clean.append(x[: int(seconds * PIPELINE_SR)])
    return clean


def tiny_noise_bank(
    directory: str = "eval/fixtures/tiny",
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """The committed tiny talkers and babble, plus seeded white noise: a noise-only bank.

    Unlike ``eval.bench --set tiny`` there is no competing talker, which a noise
    suppressor is not meant to remove.
    """
    from hearing.audio.io import load

    clean = [load(f"{directory}/speech_{k}.wav")[0] for k in ("a", "b")]
    n = max(len(c) for c in clean)
    noises = {
        "babble": load(f"{directory}/noise_babble.wav")[0],
        "white": np.random.default_rng(42).normal(0, 1, n).astype(np.float32),
    }
    return clean, noises


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="DeepFilterNet3 streaming benchmark")
    parser.add_argument("variants", nargs="*", default=["dfn"], choices=sorted(CONFIGS))
    parser.add_argument(
        "--speech", help="directory of clean speech files (else synthetic)"
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="committed eval/fixtures/tiny talkers + babble/white noise, no competing talker",
    )
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--snrs", default="0,5,10")
    parser.add_argument("--strength", type=float, default=1.0, help="suppression slider")
    parser.add_argument("--rows", action="store_true", help="print per-utterance rows")
    args = parser.parse_args()
    snrs = tuple(float(v) for v in args.snrs.split(","))
    clean = noises = None
    if args.tiny:
        clean, noises = tiny_noise_bank()
    elif args.speech:
        utterances = _load_speech(args.speech, args.seconds)
        if len(utterances) < 2:
            raise SystemExit(
                "need at least two speech files (one becomes babble noise)"
            )
        clean = utterances[:-1]
        rng = np.random.default_rng(42)
        n = max(len(c) for c in clean)
        babble = np.zeros(n, np.float32)
        for k in range(3):  # three shifted copies of the held-out talker
            shifted = np.roll(np.resize(utterances[-1], n), k * n // 3)
            babble += shifted / (np.sqrt(np.mean(shifted**2)) + 1e-9)
        noises = {"white": rng.normal(0, 1, n).astype(np.float32), "babble": babble}
    for key in args.variants:
        result = benchmark(
            key,
            clean=clean,
            noises=noises,
            snrs_db=snrs,
            seconds=args.seconds,
            strength=args.strength,
        )
        rows = result.pop("rows")
        print(json.dumps(result, indent=2))
        if args.rows:
            for row in rows:
                print(json.dumps(row))
