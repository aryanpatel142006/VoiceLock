import logging
import math

import numpy as np
import pytest

from eval.metrics import (
    LatencyStats,
    normalize_text,
    pesq_wb,
    rtf,
    si_sdr,
    si_sdri,
    stoi,
    summarize,
    wer,
    wer_batch,
)

SR = 16000


def speechlike(seconds: float = 2.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Voiced harmonics with a gliding pitch, ~4 Hz syllable envelope and short pauses."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    # Harmonics shaped by two fixed formant bumps (~500 Hz and ~1500 Hz).
    x = np.zeros_like(t)
    for k in range(1, 30):
        fk = k * 150
        gain = np.exp(-(((fk - 500) / 300) ** 2)) + 0.5 * np.exp(-(((fk - 1500) / 400) ** 2)) + 0.02
        x += gain * np.sin(k * phase + rng.uniform(0, 2 * np.pi))
    env = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 0.5
    env *= (np.sin(2 * np.pi * 0.5 * t) > -0.6)  # pauses between "phrases"
    x = x * env + 0.003 * rng.standard_normal(t.size)
    return (0.3 * x / np.max(np.abs(x))).astype(np.float32)


@pytest.fixture(scope="module")
def x() -> np.ndarray:
    return speechlike()


# --- SI-SDR --------------------------------------------------------------------------------


def test_si_sdr_identity_and_scale_invariance(x):
    s = si_sdr(x, x)
    assert s > 60
    assert math.isfinite(s)
    assert abs(si_sdr(0.3 * x, x) - s) < 1e-3


def test_si_sdr_white_noise_at_10db(x):
    rng = np.random.default_rng(1)
    ref = x.astype(np.float64)
    noise = rng.standard_normal(ref.size)
    noise *= np.sqrt(np.sum(ref**2) / np.sum(noise**2) / 10.0)  # exactly 10 dB SNR
    est = (ref + noise).astype(np.float32)
    assert abs(si_sdr(est, x) - 10.0) < 0.5


def test_si_sdri(x):
    rng = np.random.default_rng(2)
    mix = (x + 0.1 * rng.standard_normal(x.size)).astype(np.float32)
    assert si_sdri(x, mix, x) > 0
    assert si_sdri(mix, mix, x) == 0


def test_si_sdr_truncates_and_rejects_bad_input(x):
    assert si_sdr(x, x[:-100]) > 60
    with pytest.raises(ValueError):
        si_sdr(np.zeros(0, np.float32), x)
    with pytest.raises(ValueError):
        si_sdr(np.zeros((2, 10), np.float32), np.zeros((2, 10), np.float32))
    assert math.isfinite(si_sdr(np.zeros(100, np.float32), x[:100]))


# --- PESQ / STOI ---------------------------------------------------------------------------


def test_stoi_identity(x):
    assert stoi(x, x) >= 0.99


def test_pesq_identity(x):
    assert pesq_wb(x, x) >= 4.0


def test_pesq_resamples_from_48k():
    x48 = speechlike(sr=48000)
    assert pesq_wb(x48, x48, sr=48000) >= 4.0


def test_pesq_silence_returns_nan(caplog):
    silence = np.zeros(2 * SR, np.float32)
    with caplog.at_level(logging.WARNING, logger="eval.metrics"):
        assert math.isnan(pesq_wb(silence, silence))
    assert "pesq_wb failed" in caplog.text


def test_pesq_too_short_returns_nan(x):
    assert math.isnan(pesq_wb(x[: SR // 10], x[: SR // 10]))  # pesq needs >= 0.25 s


# --- WER -----------------------------------------------------------------------------------


def test_wer_basic():
    assert wer("the cat sat", "the cat sat") == 0
    assert abs(wer("the dog sat", "the cat sat") - 1 / 3) < 1e-9


def test_wer_normalization():
    assert normalize_text("  The cat,   sat! ") == "the cat sat"
    assert wer("The cat, sat!", "the cat sat") == 0
    assert wer("The cat, sat!", "the cat sat", normalize=False) > 0


def test_wer_empty_cases():
    assert wer("", "") == 0.0
    assert wer("", "the cat") == 1.0
    assert math.isnan(wer("hello", ""))


def test_wer_batch_is_corpus_level():
    # 1 error over 5 reference words, not the mean of per-utterance WERs (0.25).
    assert abs(wer_batch(["a x", "c d e"], ["a b", "c d e"]) - 0.2) < 1e-9
    with pytest.raises(ValueError):
        wer_batch(["a"], ["a", "b"])


# --- Latency / RTF / summarize -------------------------------------------------------------


def test_latency_stats():
    st = LatencyStats.from_ms([float(i) for i in range(1, 101)])
    assert abs(st.p50 - 50.5) <= 0.5
    assert abs(st.p95 - 95.05) <= 0.5
    assert st.max == 100
    assert st.mean == 50.5
    assert st.n == 100
    assert LatencyStats.from_ms([]).n == 0


def test_rtf():
    assert rtf(0.5, 10) == 0.05
    with pytest.raises(ValueError):
        rtf(1.0, 0)


def test_summarize_ignores_nan():
    rows = [
        {"id": "a", "si_sdri": 4.0, "pesq": math.nan, "ok": True},
        {"id": "b", "si_sdri": 8.0, "pesq": 3.0, "ok": False},
        {"id": "c", "si_sdri": math.nan, "pesq": 2.0, "wer": math.nan},
    ]
    s = summarize(rows)
    assert s["si_sdri"] == {"mean": 6.0, "median": 6.0, "n": 2}
    assert s["pesq"] == {"mean": 2.5, "median": 2.5, "n": 2}
    assert s["wer"]["n"] == 0 and math.isnan(s["wer"]["mean"])
    assert "id" not in s and "ok" not in s
