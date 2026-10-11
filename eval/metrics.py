"""Evaluation metrics: SI-SDR(i), PESQ, STOI, WER, latency stats and RTF.

All audio metrics take 1-D float32 arrays at ``sr`` (16 kHz in the pipeline). Inputs of
different lengths are truncated to the shortest one; empty inputs raise ``ValueError``.
"""

from __future__ import annotations

import logging
import math
import numbers
import re
from dataclasses import dataclass
from math import gcd

import numpy as np

log = logging.getLogger(__name__)

PESQ_SR = 16000
_EPS = 1e-10  # caps SI-SDR at +/-100 dB for identical or silent signals


def _prep(*signals: np.ndarray) -> list[np.ndarray]:
    """Validate 1-D inputs, truncate to the shortest length and cast to float64."""
    arrs = [np.asarray(s, dtype=np.float64) for s in signals]
    for a in arrs:
        if a.ndim != 1:
            raise ValueError(f"expected a 1-D signal, got shape {a.shape}")
    n = min(a.shape[0] for a in arrs)
    if n == 0:
        raise ValueError("empty signal")
    return [a[:n] for a in arrs]


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly

    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g)


# --- SI-SDR -------------------------------------------------------------------------------


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant SDR in dB (zero-mean). Capped at about +/-100 dB, never inf/nan."""
    est, ref = _prep(est, ref)
    est = est - est.mean()
    ref = ref - ref.mean()
    # Projection form ||a*ref||^2 / ||est - a*ref||^2 equals rho^2 / (1 - rho^2), where rho
    # is the correlation of est and ref. This form stays exactly scale-invariant with eps.
    denom = float(np.linalg.norm(est) * np.linalg.norm(ref))
    rho = float(np.dot(est, ref)) / denom if denom > 0 else 0.0
    rho2 = min(max(rho * rho, 0.0), 1.0)
    return float(10.0 * np.log10((rho2 + _EPS) / (1.0 - rho2 + _EPS)))


def si_sdri(est: np.ndarray, mix: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR improvement in dB: si_sdr(est, ref) - si_sdr(mix, ref)."""
    est, mix, ref = _prep(est, mix, ref)
    return si_sdr(est, ref) - si_sdr(mix, ref)


# --- PESQ / STOI --------------------------------------------------------------------------


def pesq_wb(est: np.ndarray, ref: np.ndarray, sr: int = 16000) -> float:
    """Wide-band PESQ (MOS-LQO, about 1.0-4.64). Returns nan (with a warning) if pesq fails."""
    from pesq import PesqError, pesq

    est, ref = _prep(est, ref)
    est = _resample(est, sr, PESQ_SR)
    ref = _resample(ref, sr, PESQ_SR)
    if not np.any(est) or not np.any(ref):  # pesq would divide by zero, then raise
        log.warning("pesq_wb failed (silent input); returning nan")
        return math.nan
    try:
        return float(pesq(PESQ_SR, ref, est, "wb"))
    except (PesqError, ValueError) as e:  # NoUtterancesError, BufferTooShortError, ...
        log.warning("pesq_wb failed (%s: %s); returning nan", type(e).__name__, e)
        return math.nan


def stoi(est: np.ndarray, ref: np.ndarray, sr: int = 16000, extended: bool = False) -> float:
    """STOI (or extended STOI) in [0, 1] via pystoi (which resamples to 10 kHz itself)."""
    from pystoi import stoi as _stoi

    est, ref = _prep(est, ref)
    return float(_stoi(ref, est, sr, extended=extended))


# --- WER ----------------------------------------------------------------------------------

_APOSTROPHES = re.compile(r"['’]")
_PUNCT = re.compile(r"[^\w\s]|_")
_SPACES = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, drop apostrophes ("don't" -> "dont"), other punctuation -> space, collapse."""
    s = _APOSTROPHES.sub("", s.lower())
    s = _PUNCT.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def _clean(s: str, normalize: bool) -> str:
    return normalize_text(s) if normalize else _SPACES.sub(" ", s).strip()


def wer(hyp: str, ref: str, normalize: bool = True) -> float:
    """Word error rate of one hypothesis against one reference.

    Empty reference: 0.0 if the hypothesis is empty too, else nan with a warning (WER is
    undefined; ``summarize`` skips nan).
    """
    import jiwer

    hyp, ref = _clean(hyp, normalize), _clean(ref, normalize)
    if not ref:
        if not hyp:
            return 0.0
        log.warning("wer: empty reference with non-empty hypothesis; returning nan")
        return math.nan
    if not hyp:
        return 1.0
    return float(jiwer.wer(ref, hyp))


def wer_batch(hyps: list[str], refs: list[str], normalize: bool = True) -> float:
    """Corpus-level WER: total errors over total reference words."""
    import jiwer

    if len(hyps) != len(refs):
        raise ValueError(f"{len(hyps)} hypotheses for {len(refs)} references")
    hyps = [_clean(h, normalize) for h in hyps]
    refs = [_clean(r, normalize) for r in refs]
    n_ref = sum(len(r.split()) for r in refs)
    if n_ref == 0:
        if all(not h for h in hyps):
            return 0.0
        log.warning("wer_batch: all references empty; returning nan")
        return math.nan
    out = jiwer.process_words(refs, hyps)
    return float((out.substitutions + out.deletions + out.insertions) / n_ref)


# --- Latency / RTF ------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyStats:
    """Latency summary in milliseconds (linear-interpolated percentiles)."""

    p50: float
    p95: float
    max: float
    mean: float
    n: int

    @classmethod
    def from_ms(cls, samples_ms: list[float]) -> LatencyStats:
        """Summarise latency samples; nan samples are dropped. No samples -> all nan, n=0."""
        a = np.asarray(samples_ms, dtype=np.float64).ravel()
        a = a[~np.isnan(a)]
        if a.size == 0:
            return cls(math.nan, math.nan, math.nan, math.nan, 0)
        p50, p95 = np.percentile(a, [50, 95])
        return cls(float(p50), float(p95), float(a.max()), float(a.mean()), int(a.size))


def rtf(compute_s: float, audio_s: float) -> float:
    """Real-time factor: compute time / audio duration (< 1 is faster than real time)."""
    if audio_s <= 0:
        raise ValueError(f"audio_s must be > 0, got {audio_s}")
    return float(compute_s) / float(audio_s)


# --- Aggregation --------------------------------------------------------------------------


def _is_number(v: object) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool)


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Per numeric key: ``{"mean", "median", "n"}`` over rows, ignoring nan and non-numbers.

    A key whose values are all nan gets mean/median nan and n=0.
    """
    values: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.items():
            if _is_number(v):
                values.setdefault(k, []).append(float(v))
    out: dict[str, dict[str, float]] = {}
    for k, vs in values.items():
        a = np.asarray(vs, dtype=np.float64)
        a = a[~np.isnan(a)]
        if a.size == 0:
            out[k] = {"mean": math.nan, "median": math.nan, "n": 0}
        else:
            out[k] = {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size)}
    return out
