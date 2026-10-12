"""Loudest-mode denoise: DeepFilterNet3 on the separator's input + a soft downward expander on the output.

Why (Aryan, Oct 11 live test): with ``HEARING_PICK=loudest`` the stock 2-speaker separator has no noise
output, so café noise lands in whichever stream is picked, and the fixed +22 dB output gain makes it plainly
audible, most of all in speech pauses. In loudest mode the tse stage (:mod:`hearing.tse.stage`) now runs:

1. :class:`LoudestDenoiser` on its INPUT, before the separator: DeepFilterNet3 (``dfn_ll`` by default,
   13 ms algorithmic delay; ``dfn``: 33 ms) at full strength (no attenuation limit). Measured offline
   (café at SNR 5/10 dB on held-out team takes, plus two live café takes): denoising before separation beat
   denoising the picked stream on every count. After separation, DFN treats separation artifacts as noise
   (-5 dB on voiced speech) and the café-only stream picked during a pause stays locked when the voice comes
   back, so word onsets after a pause went silent for ~1 s; before separation the streams carry no café,
   so pauses are quiet and the voice is picked again on its first word (SI-SDR up, not down).
2. :class:`Expander` on its OUTPUT (the picked voice): a gentle downward expander for what is left in
   pauses. 10 ms hops with one hop of lookahead (it opens during the hop before a word starts, so onsets
   are never clipped), a 150 ms hold and a slow release. Its threshold sits ``below_db`` under a slowly
   decaying speech-level reference, so it never chases the voice; at most ``depth_db`` is taken away.

Knobs (read when the tse stage is built, i.e. at server start):

* ``HEARING_LOUDEST_DENOISE``: on by default in loudest mode; ``0``/``off``/``false`` = off (as before).
* ``HEARING_LOUDEST_DENOISE_MODEL``: ``dfn_ll`` (default) or ``dfn``.
* ``HEARING_LOUDEST_GATE``: the expander, on by default; ``0`` = DeepFilterNet only.
* ``HEARING_LOUDEST_VAD``: the foreground-speech gate (:class:`ForegroundVad`), on by default; ``0`` = off.
  ``HEARING_LOUDEST_VAD_ON`` (Silero probability that opens it, 0.8), ``HEARING_LOUDEST_VAD_HOLD_MS`` (400),
  ``HEARING_LOUDEST_VAD_PREROLL_MS`` (250; Silero reports most onsets within 50 ms, some within 250 ms).

3. :class:`ForegroundVad` (Oct 11, live test: café babble with nobody talking played at about -30 dBFS). The
   expander only gates relative to the recent speech level, so with no speech it never closes, and the fixed
   +25 dB output gain lifts what DFN leaves of the babble. The VAD decides per 10 ms hop whether someone in the
   foreground is talking: Silero VAD on the RAW input (measured on the test clips: on the denoised signal Silero
   fires on café babble, median 0.74, because DFN turns the loudest babble into speech-like residue; on the raw
   input café-only is median 0.15 and the live café take 0.0 while foreground speech is 0.99) AND DFN's own
   local-SNR estimate >= 0 dB AND the denoised hop above an absolute floor (-60 dBFS). Open = Silero >= ``on``;
   stays open while Silero >= ``keep`` plus a ``hold_ms`` hangover; opens ``preroll_ms`` BEFORE the onset. The
   decision is made on the input side, so the tse stage, which plays the input ~1.5 s later, applies it exactly
   aligned (:attr:`LoudestDenoiser.vad_mask` / ``vad_retro``) with that look-ahead for free: no added delay.
   Closed = full mute (ramped 5 ms open / 30 ms close). If Silero cannot load, the local-SNR estimate alone
   drives the same state machine. Offline on the test clips: café-only open 5.7 % of hops, live café 0 %,
   foreground speech (within 25 dB of its peak) closed 2-2.5 %.

Latency: the DFN delay plus 10 ms for the expander's lookahead (23 ms with ``dfn_ll``), added to the tse
stage's reported ``latency_ms``. Compute: ~0.6 ms per 10 ms hop on one CPU thread on the M-series Mac.
If the model cannot load, the input passes through a plain delay line of the same length.
"""

from __future__ import annotations

import logging
import math
import os
import time

import numpy as np

from hearing.audio.io import HOP, PIPELINE_SR
from hearing.types import AudioChunk, StageContext

log = logging.getLogger(__name__)

ENV = "HEARING_LOUDEST_DENOISE"
MODEL_ENV = "HEARING_LOUDEST_DENOISE_MODEL"
GATE_ENV = "HEARING_LOUDEST_GATE"
VAD_ENV = "HEARING_LOUDEST_VAD"
VAD_ON_ENV = "HEARING_LOUDEST_VAD_ON"
VAD_HOLD_ENV = "HEARING_LOUDEST_VAD_HOLD_MS"
VAD_PREROLL_ENV = "HEARING_LOUDEST_VAD_PREROLL_MS"
DEFAULT_MODEL = "dfn_ll"
_OFF = ("0", "off", "false", "no")


def enabled() -> bool:
    """``$HEARING_LOUDEST_DENOISE``: on unless 0/off/false."""
    return os.environ.get(ENV, "").strip().lower() not in _OFF


def model() -> str:
    v = os.environ.get(MODEL_ENV, "").strip().lower()
    return v if v in ("dfn", "dfn_ll") else DEFAULT_MODEL


def gate_enabled() -> bool:
    return os.environ.get(GATE_ENV, "").strip().lower() not in _OFF


def vad_enabled() -> bool:
    """``$HEARING_LOUDEST_VAD``: the foreground-speech gate, on unless 0/off/false."""
    return os.environ.get(VAD_ENV, "").strip().lower() not in _OFF


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        v = float(os.environ.get(name, "") or default)
    except ValueError:
        return default
    return v if math.isfinite(v) and lo <= v <= hi else default


def vad_settings() -> dict:
    """The :class:`ForegroundVad` knobs from the environment (read when the tse stage is built)."""
    return {"on": _env_float(VAD_ON_ENV, 0.8, 0.05, 1.0), "hold_ms": _env_float(VAD_HOLD_ENV, 400.0, 0.0, 5000.0),
            "preroll_ms": _env_float(VAD_PREROLL_ENV, 250.0, 0.0, 1000.0)}


def _db(x: np.ndarray) -> float:
    return 10.0 * math.log10(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


class Expander:
    """Soft downward expander on 10 ms hops, one hop of lookahead (output delayed by ``hop`` samples).

    Per hop: ``ref`` follows the speech level (instant rise, ``ref_decay_db_s`` fall, never under
    ``ref_min_db``); the threshold is ``ref - below_db``. The gain for a hop looks at its own level and the
    next hop's: at or above the threshold 0 dB, under it ``(level - thr) * (ratio - 1)`` dB, at most
    ``-depth_db``. Opening is immediate (a linear ramp across the hop before the onset), closing waits
    ``hold_ms`` and then falls at ``release_db_s``. Gains are ramped linearly within each hop (no zipper).
    """

    def __init__(self, sr: int = PIPELINE_SR, hop: int = HOP, below_db: float = 24.0, ratio: float = 3.0,
                 depth_db: float = 20.0, hold_ms: float = 150.0, release_db_s: float = 60.0,
                 ref_decay_db_s: float = 3.0, ref_min_db: float = -60.0):
        self.sr, self.hop = int(sr), int(hop)
        hop_s = self.hop / self.sr
        self.below_db, self.ratio, self.depth_db = float(below_db), float(ratio), float(depth_db)
        self.hold_hops = max(0, round(hold_ms / 1000.0 / hop_s))
        self.release_db = release_db_s * hop_s
        self.ref_decay_db = ref_decay_db_s * hop_s
        self.ref_min_db = float(ref_min_db)
        self.latency_samples = self.hop
        self.reset()

    def reset(self) -> None:
        self._ref = self.ref_min_db
        self._g_db = 0.0
        self._hold = 0
        self._prev = np.zeros(self.hop, np.float32)
        self._prev_lev = -120.0
        self.gain_db = 0.0  # gain of the last emitted hop (telemetry)

    def _target(self, level: float) -> float:
        thr = self._ref - self.below_db
        if level >= thr:
            return 0.0
        return -min(self.depth_db, (thr - level) * (self.ratio - 1.0))

    def process(self, x: np.ndarray) -> np.ndarray:
        """``x``: whole hops. Returns the same number of samples, ``hop`` samples behind."""
        x = np.asarray(x, np.float32)
        out = np.empty_like(x)
        ramp = np.arange(1, self.hop + 1, dtype=np.float64) / self.hop
        for i in range(0, len(x), self.hop):
            h = x[i:i + self.hop]
            lev = _db(h)
            self._ref = max(lev, self._ref - self.ref_decay_db, self.ref_min_db)
            target = self._target(max(self._prev_lev, lev))  # gain for the previous hop, looking one hop ahead
            if target >= self._g_db:
                g = target
                self._hold = self.hold_hops
            elif self._hold > 0:
                self._hold -= 1
                g = self._g_db
            else:
                g = max(target, self._g_db - self.release_db)
            a, b = 10.0 ** (self._g_db / 20.0), 10.0 ** (g / 20.0)
            out[i:i + self.hop] = self._prev * (a + (b - a) * ramp)
            self._g_db = self.gain_db = g
            self._prev, self._prev_lev = h.copy(), lev
        return out


class ForegroundVad:
    """Per-hop foreground-speech decision: Silero VAD on the raw input, DFN local SNR and an absolute floor.

    :meth:`update` takes one chunk of whole hops (raw input, denoised output, DFN ``lsnr_hops``) and returns
    ``(mask, retro)``: ``mask`` is a float32 0/1 gate per sample of the chunk, ``retro`` the number of samples
    BEFORE the chunk to open as well (onset pre-roll that reaches into the previous chunk; the caller holds the
    not-yet-played history). State machine per 10 ms hop: open when Silero >= ``on`` and lsnr >= ``lsnr_on_db``
    and the denoised hop >= ``floor_db``; while open, any hop with Silero >= ``keep`` (and above the floor)
    restarts the ``hold_ms`` hangover; closed after the hangover runs out. Until :meth:`warm` has loaded Silero
    (or if it cannot), the local SNR alone stands in: lsnr >= 2 dB counts as ``on``, >= -5 dB as ``keep``.
    """

    WINDOW = 512  # Silero's fixed 16 kHz window (32 ms)

    def __init__(self, sr: int = PIPELINE_SR, hop: int = HOP, on: float = 0.8, keep: float = 0.5,
                 lsnr_on_db: float = 0.0, floor_db: float = -60.0, hold_ms: float = 400.0,
                 preroll_ms: float = 250.0):
        self.sr, self.hop = int(sr), int(hop)
        self.on, self.keep = float(on), min(float(keep), float(on))
        self.lsnr_on_db, self.floor_db = float(lsnr_on_db), float(floor_db)
        self.hold_hops = max(0, round(hold_ms / 1000.0 * self.sr / self.hop))
        self.preroll = max(0, round(preroll_ms / 1000.0 * self.sr))
        self.hold_ms, self.preroll_ms = float(hold_ms), float(preroll_ms)
        self._model = None
        self.ready = False
        self.error: str | None = None
        self.compute_ms = 0.0
        self.reset()

    @property
    def name(self) -> str:
        return "silero" if self.ready else "lsnr"

    def warm(self) -> bool:
        """Load Silero VAD (ONNX, CPU; ships with the ``silero-vad`` package, no download)."""
        if self.ready or self.error is not None:
            return self.ready
        try:
            import silero_vad

            self._model = silero_vad.load_silero_vad(onnx=True)
            self._model.reset_states()
        except Exception as e:  # noqa: BLE001  (package missing / onnxruntime trouble: the lsnr fallback)
            self.error = f"{type(e).__name__}: {e}"
            log.warning("foreground VAD: Silero unavailable, gating on DFN local SNR only: %s", self.error)
            return False
        self.ready = True
        return True

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset_states()
        self._raw = np.zeros(0, np.float32)
        self.p = 0.0  # latest Silero probability (telemetry)
        self.lsnr_db = math.nan
        self.open = False
        self._hold = 0  # hops of hangover left

    def _silero(self, raw: np.ndarray) -> list[tuple[int, float]]:
        """Feed raw samples; returns ``(sample offset in raw where the window ended, p)`` per finished window."""
        out: list[tuple[int, float]] = []
        if self._model is None:
            return out
        import torch

        buf = np.concatenate([self._raw, raw])
        pre = len(self._raw)
        i = 0
        while i + self.WINDOW <= len(buf):
            with torch.no_grad():
                p = float(self._model(torch.from_numpy(np.ascontiguousarray(buf[i:i + self.WINDOW])), self.sr).item())
            i += self.WINDOW
            out.append((i - pre, p))
        self._raw = buf[i:]
        return out

    def update(self, raw: np.ndarray, denoised: np.ndarray, lsnr_hops: np.ndarray | None) -> tuple[np.ndarray, int]:
        t0 = time.perf_counter()
        raw = np.asarray(raw, np.float32)
        n = len(raw)
        hops = n // self.hop
        mask = np.zeros(n, np.float32)
        retro = 0
        ends = self._silero(raw)  # the probability for hop h is the latest window that ended at or before it
        k = 0
        for h in range(hops):
            a, b = h * self.hop, (h + 1) * self.hop
            while k < len(ends) and ends[k][0] <= b:
                self.p = ends[k][1]
                k += 1
            ls = float(lsnr_hops[h]) if lsnr_hops is not None and h < len(lsnr_hops) else math.nan
            self.lsnr_db = ls
            lev = _db(denoised[a:b]) if len(denoised) >= b else -120.0
            above = lev >= self.floor_db
            if self.ready:
                p = self.p
            else:  # lsnr alone: >= 2 dB is a sure onset, >= -5 dB keeps the gate open
                p = 1.0 if ls >= 2.0 else (self.keep if ls >= -5.0 else 0.0)
            on = above and p >= self.on and (self.ready and ls >= self.lsnr_on_db or not self.ready)
            keep = above and p >= self.keep
            if on:
                if not self.open:  # onset: pre-roll into the samples before it
                    self.open = True
                    retro = max(retro, self.preroll - a)
                    mask[max(0, a - self.preroll):a] = 1.0
                self._hold = self.hold_hops
            elif self.open and keep:
                self._hold = self.hold_hops
            elif self.open:
                if self._hold > 0:
                    self._hold -= 1
                else:
                    self.open = False
            if self.open:
                mask[a:b] = 1.0
        self.compute_ms = 1000.0 * (time.perf_counter() - t0)
        return mask, max(0, retro)

    def health(self) -> dict:
        return {"model": self.name, "ready": self.ready, "error": self.error, "open": self.open,
                "p": round(self.p, 2), "lsnr_db": None if math.isnan(self.lsnr_db) else round(self.lsnr_db, 1),
                "on": self.on, "keep": self.keep, "hold_ms": self.hold_ms, "preroll_ms": self.preroll_ms,
                "compute_ms": round(self.compute_ms, 2)}


class _Reblocked:
    """Any chunk length in, same length out, constant delay: whole ``hop`` blocks go to ``_hops``."""

    hop = HOP
    latency_samples = 0

    def _clear(self) -> None:
        self._in = np.zeros(0, np.float32)
        self._out = np.zeros(0, np.float32)
        self._mq = np.zeros(0, np.float32)  # per-sample mask queued by _hops (LoudestDenoiser's VAD), with _out

    def _hops(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def process(self, x: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        x = np.asarray(x, np.float32)
        n = len(x)
        self._in = np.concatenate([self._in, x])
        whole = len(self._in) - len(self._in) % self.hop
        if whole:
            self._out = np.concatenate([self._out, self._hops(self._in[:whole])])
            self._in = self._in[whole:]
        if len(self._out) < n:  # only with chunks that are not whole 10 ms hops: the delay grows once
            pad = n - len(self._out)
            self._out = np.concatenate([np.zeros(pad, np.float32), self._out])
            self._mq = np.concatenate([np.zeros(pad, np.float32), self._mq])
        y, self._out = self._out[:n], self._out[n:]
        if len(self._mq) >= n:
            self.vad_mask, self._mq = self._mq[:n], self._mq[n:]
        self.compute_ms = 1000.0 * (time.perf_counter() - t0)
        return np.clip(y, -1.0, 1.0).astype(np.float32)


class LoudestDenoiser(_Reblocked):
    """DeepFilterNet3 at full strength for any chunk length, constant delay ``latency_samples``.

    ``warm()`` loads the model (call it off the audio thread); until then ``process`` is a delay line of
    the same length, so the output timing never moves. A failed load leaves the delay line in place.
    """

    def __init__(self, variant: str | None = None, sr: int = PIPELINE_SR, vad: ForegroundVad | None = None):
        from hearing.enhance.dfn import DFN

        self.variant = variant or model()
        self.dfn = DFN(self.variant, strength=1.0)
        self.sr = sr
        self.hop = self.dfn.hop
        # foreground-speech gate decided here, on the input side; the tse stage applies vad_mask / vad_retro
        # to the same samples when it plays them (~1.5 s later), so the pre-roll costs no delay
        self.vad = vad
        self.vad_mask: np.ndarray | None = None  # per sample of the last process() output, 1 = someone talks
        self.vad_retro = 0  # samples before the last process() chunk to open as well (onset pre-roll)
        self.latency_samples = self.dfn.latency_samples
        self.latency_ms = 1000.0 * self.latency_samples / sr
        self.name = self.variant
        self._ctx = StageContext.default()  # private: the dfn's telemetry stays out of the session's extras
        self.ready = False
        self.error: str | None = None
        self.compute_ms = 0.0  # last chunk's compute (telemetry)
        self.reset()

    def warm(self) -> bool:
        if self.vad is not None:
            self.vad.warm()
        if self.ready or self.error is not None:
            return self.ready
        from hearing.enhance.dfn import DFN

        try:  # a throwaway stage loads the shared backend (hearing.models caches it); the live one keeps its state
            DFN(self.variant, strength=1.0).process(AudioChunk(np.zeros(self.hop, np.float32), self.sr, 0, 0.0),
                                                    StageContext.default())
        except Exception as e:  # noqa: BLE001  (no weights / onnxruntime: keep the delay line)
            self.error = f"{type(e).__name__}: {e}"
            log.warning("loudest denoise (%s) unavailable, playing undenoised: %s", self.variant, self.error)
            return False
        self.ready = True
        return True

    def reset(self) -> None:
        self.dfn.reset()
        self._clear()
        self._line = np.zeros(self.latency_samples, np.float32)  # the delay line used until ready
        if self.vad is not None:
            self.vad.reset()
        self.vad_mask = None
        self.vad_retro = 0

    def process(self, x: np.ndarray) -> np.ndarray:
        self.vad_retro = 0
        return super().process(x)

    def _hops(self, x: np.ndarray) -> np.ndarray:
        if not self.ready:
            buf = np.concatenate([self._line, x])
            self._line = buf[len(x):]
            y, lsnr = buf[: len(x)], None
        else:
            y = self.dfn.process(AudioChunk(x, self.sr, 0, 0.0), self._ctx).samples
            lsnr = self.dfn.lsnr_hops
        if self.vad is not None:
            mask, retro = self.vad.update(x, y, lsnr)
            if retro > 0:  # before this block: whatever is still queued, then the caller's history
                q = min(retro, len(self._mq))
                if q:
                    self._mq[len(self._mq) - q:] = 1.0
                self.vad_retro += retro - q
            self._mq = np.concatenate([self._mq, mask])
        return y

    def health(self) -> dict:
        h = {"model": self.name, "ready": self.ready, "error": self.error,
             "latency_ms": round(self.latency_ms, 1), "compute_ms": round(self.compute_ms, 2),
             "lsnr_db": None if math.isnan(self.dfn.lsnr_db) else round(self.dfn.lsnr_db, 1)}
        if self.vad is not None:
            h["vad"] = self.vad.health()
        return h


class OutputGate(_Reblocked):
    """:class:`Expander` for any chunk length (delay: one 10 ms hop)."""

    def __init__(self, sr: int = PIPELINE_SR, **kwargs):
        self.expander = Expander(sr, **kwargs)
        self.hop = self.expander.hop
        self.latency_samples = self.expander.latency_samples
        self.latency_ms = 1000.0 * self.latency_samples / sr
        self.compute_ms = 0.0
        self.reset()

    def reset(self) -> None:
        self.expander.reset()
        self._clear()

    def _hops(self, x: np.ndarray) -> np.ndarray:
        return self.expander.process(x)

    @property
    def gain_db(self) -> float:
        return self.expander.gain_db
