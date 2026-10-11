"""T-099: the Natural chain's room and processed summands are time-aligned at every slider.

In ``dfn_ll>gate>mixer`` the room is blended back inside DFN (``atten_lim_db``): the noisy
spectrum of the *same* STFT frame the mask is applied to, so the room path carries the same
algorithmic delay as the processed path and the sum cannot comb-filter. These tests pin that:
the output at slider 0 (room only) and at slider 1 (processed only) line up at lag 0, and a
partial slider is exactly the linear blend of the two (no phase-dependent notches).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from hearing.enhance.dfn import DFN, MockBackend
from hearing.fusion.gate import Gate
from hearing.fusion.mixer import Mixer, atten_lim_to_mix, strength_to_db
from hearing.testing import make_test_signal
from hearing.types import AudioChunk, StageContext


def natural(backend) -> list:
    return [DFN("dfn_ll", backend=backend), Gate(), Mixer()]


def run(stages, x: np.ndarray, s: float) -> np.ndarray:
    ctx = StageContext.default()
    ctx.suppression = s
    out = []
    for i in range(0, len(x), 160):
        chunk = AudioChunk(
            samples=x[i : i + 160],
            sample_rate=16000,
            seq=i // 160,
            capture_ts=i / 16000,
        )
        for stage in stages:
            chunk = stage.process(chunk, ctx)
        out.append(chunk.samples)
    return np.concatenate(out)


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int = 400) -> int:
    """Lag (samples) of ``b`` relative to ``a`` at the cross-correlation peak."""
    lags = range(-max_lag, max_lag + 1)
    scores = [
        np.dot(a[max_lag:-max_lag], np.roll(b, -lag)[max_lag:-max_lag]) for lag in lags
    ]
    return list(lags)[int(np.argmax(np.abs(scores)))]


def impulses(n: int = 16000) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    x[2000::3000] = 0.5
    return x


def test_room_and_processed_summands_are_time_aligned():
    x = impulses()
    room = run(natural(MockBackend("dfn_ll", gain=0.5)), x, 0.0)
    processed = run(natural(MockBackend("dfn_ll", gain=0.5)), x, 1.0)
    assert np.abs(room).max() > 0.1 and np.abs(processed).max() > 0.05
    assert abs(xcorr_lag(room, processed)) <= 1


@pytest.mark.parametrize("s", [0.25, 0.5, 0.75])
def test_partial_slider_is_the_phase_coherent_blend(s):
    """out(s) == m * out(0) + (1 - m) * out(1): no relative delay, hence no comb filter."""
    x = make_test_signal(1.0, kind="noise").astype(np.float32) * 0.3
    room = run(natural(MockBackend("dfn_ll", gain=0.5)), x, 0.0)
    processed = run(natural(MockBackend("dfn_ll", gain=0.5)), x, 1.0)
    out = run(natural(MockBackend("dfn_ll", gain=0.5)), x, s)
    m = atten_lim_to_mix(strength_to_db(s))
    np.testing.assert_allclose(out, m * room + (1 - m) * processed, atol=1e-4)
    # Spectral-flatness proxy: a comb filter carves periodic notches into the spectrum.
    spec = np.abs(np.fft.rfft(out[1600:])) + 1e-9
    ref = np.abs(np.fft.rfft(room[1600:])) + 1e-9
    ratio = spec / ref
    assert ratio.min() > 0.45 * np.median(ratio)


@pytest.mark.slow
def test_real_dfn_ll_room_and_processed_are_time_aligned():
    """Same lag check with the real DeepFilterNet3_ll weights on speech-like noise bursts."""
    from hearing import models

    try:
        backend = models.load("dfn_ll")
    except (RuntimeError, OSError, ValueError) as exc:  # weights not provisioned here
        pytest.skip(f"dfn_ll weights unavailable: {exc}")
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(32000) * 0.1).astype(np.float32)
    x[8000:12000] += 0.5 * np.sin(2 * np.pi * 440 * np.arange(4000) / 16000).astype(
        np.float32
    )
    room = run([DFN("dfn_ll", backend=backend), Gate(), Mixer()], x, 0.0)
    processed = run([DFN("dfn_ll", backend=backend), Gate(), Mixer()], x, 1.0)
    assert abs(xcorr_lag(room, processed)) <= 1


@pytest.mark.slow
def test_real_dfn_ll_partial_slider_never_below_the_room():
    """Regression for the mid-slider 'collapse': real DeepFilterNet3_ll on speech + babble at
    5 dB SNR, scored with DNSMOS P.835. OVRL at every partial slider must be >= the room
    (s = 0, the input) and must not fall as the slider rises (0.1 MOS tolerance)."""
    import soundfile as sf

    from eval.dnsmos import available, get_dnsmos
    from hearing import models

    if not available():
        pytest.skip("DNSMOS models absent (python -m eval.dnsmos --download)")
    try:
        backend = models.load("dfn_ll")
    except (RuntimeError, OSError, ValueError) as exc:  # weights not provisioned here
        pytest.skip(f"dfn_ll weights unavailable: {exc}")
    tiny = Path(__file__).resolve().parents[1] / "eval/fixtures/tiny"
    speech = np.concatenate(
        [
            sf.read(tiny / f, dtype="float32")[0]
            for f in ("speech_a.wav", "speech_b.wav")
        ]
    )
    noise = np.resize(
        sf.read(tiny / "noise_babble.wav", dtype="float32")[0], len(speech)
    )
    noise *= np.sqrt(np.mean(speech**2) / np.mean(noise**2)) * 10 ** (-5 / 20)
    x = (speech + noise).astype(np.float32)
    x = x[: len(x) // 160 * 160]
    scorer = get_dnsmos("dnsmos")
    ovrl = {
        s: scorer.score(
            run([DFN("dfn_ll", backend=backend), Gate(), Mixer()], x, s)
        ).ovrl
        for s in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    for s in (0.25, 0.5, 0.75, 1.0):
        assert ovrl[s] >= ovrl[0.0], ovrl
    vals = list(ovrl.values())
    assert all(b >= a - 0.1 for a, b in itertools.pairwise(vals)), ovrl
