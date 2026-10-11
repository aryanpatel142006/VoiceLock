"""Mixer stage, split helper and the slider mapping (hearing.fusion.mixer)."""

from __future__ import annotations

import numpy as np
import pytest

from hearing import registry
from hearing.enhance.gtcrn import GTCRN, MockBackend
from hearing.fusion.mixer import (
    ALIGNED_MIX_KEY,
    RESIDUAL_KEY,
    SLIDER_RAMP_MS,
    Mixer,
    Split,
    atten_lim_to_mix,
    room_atten_db,
    room_gain,
    strength_to_db,
)
from hearing.testing import assert_streaming_equivalent, make_test_signal, run_streaming
from hearing.types import AudioChunk, StageContext

SR = 16000


def _chunk(x: np.ndarray, seq: int = 3) -> AudioChunk:
    return AudioChunk(np.asarray(x, dtype=np.float32), SR, seq, 0.25)


def _ctx(s: float) -> StageContext:
    ctx = StageContext.default()
    ctx.suppression = s
    return ctx


def test_strength_to_db_endpoints():
    # s = 0 is an exact bypass (0 dB limit); the curve starts at 0 dB (no 6 dB jump at 1 %).
    assert strength_to_db(0.0) == 0.0
    assert strength_to_db(1e-9) == pytest.approx(0.0, abs=1e-6)
    assert strength_to_db(0.5) == pytest.approx(15.0)
    assert strength_to_db(0.99) == pytest.approx(room_atten_db(0.99))
    assert strength_to_db(1.0) is None


def test_slider_curve_is_perceptually_linear():
    """Equal slider steps are equal dB steps of the room (3 dB per 10 points) up to the knee."""
    sliders = np.linspace(0.0, 0.9, 10)
    db = np.array([room_atten_db(s) for s in sliders])
    np.testing.assert_allclose(np.diff(db), 3.0, atol=1e-9)
    assert room_atten_db(0.9) == pytest.approx(27.0)
    # Soft knee: keeps falling monotonically, then silence at 100.
    top = [room_atten_db(s) for s in np.linspace(0.9, 0.999, 12)]
    assert np.all(np.diff(top) > 0)
    assert room_atten_db(0.95) == pytest.approx(28.5 + 6.0206, abs=1e-3)
    assert room_atten_db(1.0) == float("inf")
    assert room_gain(1.0) == 0.0
    assert room_gain(0.0) == 1.0
    assert room_gain(0.5) == pytest.approx(10 ** (-15 / 20))
    # DFN's mix-back equals the mixer/gate gain everywhere.
    for s in np.linspace(0.01, 0.99, 25):
        assert atten_lim_to_mix(strength_to_db(s)) == pytest.approx(room_gain(s), rel=1e-6)


def test_registered_and_attributes():
    assert {"mixer", "split"} <= set(registry.names())
    m, s = registry.get("mixer"), registry.get("split", delay_ms=32)
    for stage in (m, s):
        assert stage.sample_rate == SR
        assert stage.latency_ms == 0.0
    assert s.name == "split(delay_ms=32)"


def test_pass_through_without_residual():
    x = make_test_signal(0.05)
    chunk = _chunk(x)
    ctx = _ctx(0.7)
    out = Mixer().process(chunk, ctx)
    assert out is chunk
    assert ctx.telemetry.extras["mixer"] == {"residual": False}


@pytest.mark.parametrize("s", [0.0, 0.3, 0.5, 1.0])
def test_residual_mix_exact(s):
    rng = np.random.default_rng(1)
    target = rng.uniform(-0.4, 0.4, 160).astype(np.float32)
    residual = rng.uniform(-0.4, 0.4, 160).astype(np.float32)
    ctx = _ctx(s)
    ctx.telemetry.extras[RESIDUAL_KEY] = residual
    out = Mixer().process(_chunk(target, seq=9), ctx)
    np.testing.assert_allclose(out.samples, target + np.float32(room_gain(s)) * residual, atol=1e-7)
    assert out.samples.dtype == np.float32
    assert (out.seq, out.sample_rate, out.capture_ts) == (9, SR, 0.25)
    # Consumed: the next chunk without a fresh residual passes through.
    assert RESIDUAL_KEY not in ctx.telemetry.extras
    assert ctx.telemetry.extras["mixer"] == {"residual": True}
    nxt = _chunk(target)
    assert Mixer().process(nxt, ctx) is nxt


def test_residual_wins_over_aligned_mix_and_length_mismatch_raises():
    target = np.full(160, 0.2, np.float32)
    ctx = _ctx(0.5)
    ctx.telemetry.extras[RESIDUAL_KEY] = np.full(160, 0.1, np.float32)
    ctx.telemetry.extras[ALIGNED_MIX_KEY] = np.full(160, 0.9, np.float32)
    out = Mixer().process(_chunk(target), ctx)
    np.testing.assert_allclose(out.samples, 0.2 + 0.1 * room_gain(0.5), atol=1e-7)
    assert ALIGNED_MIX_KEY not in ctx.telemetry.extras
    ctx.telemetry.extras[RESIDUAL_KEY] = np.zeros(80, np.float32)
    with pytest.raises(ValueError, match="80 samples"):
        Mixer().process(_chunk(target), ctx)


def test_split_delays_the_copy_and_passes_through():
    x = make_test_signal(0.2)
    split = Split(delay_ms=32)
    ctx = StageContext.default()
    copies = []
    for i in range(0, len(x), 160):
        chunk = _chunk(x[i : i + 160])
        assert split.process(chunk, ctx) is chunk
        copies.append(ctx.telemetry.extras[ALIGNED_MIX_KEY])
    aligned = np.concatenate(copies)
    np.testing.assert_array_equal(aligned[:512], 0.0)
    np.testing.assert_array_equal(aligned[512:], x[:-512])
    split.reset()
    split.process(_chunk(x[:160]), ctx)
    np.testing.assert_array_equal(ctx.telemetry.extras[ALIGNED_MIX_KEY], 0.0)
    with pytest.raises(ValueError):
        Split(delay_ms=-1)


class _HalfDelayed:
    """Stand-in extractor: the input delayed by ``d`` samples at half amplitude."""

    name = "half_delayed"
    sample_rate = SR

    def __init__(self, d: int):
        self.d = d
        self.latency_ms = 1000.0 * d / SR
        self.reset()

    def reset(self):
        self._fifo = np.zeros(self.d, np.float32)

    def process(self, chunk, ctx):
        buf = np.concatenate((self._fifo, chunk.samples))
        self._fifo = buf[len(chunk.samples) :]
        return _chunk(0.5 * buf[: len(chunk.samples)], chunk.seq)


@pytest.mark.parametrize("s", [0.0, 0.4, 1.0])
def test_split_extractor_mixer_gives_slider_blend(s):
    d = 512
    x = make_test_signal(0.3)
    pipe = registry.Pipeline("split>half>mixer", [Split(delay_ms=32), _HalfDelayed(d), Mixer()])
    assert pipe.latency_ms == pytest.approx(32.0)
    y = run_streaming(pipe, x, SR, ctx=_ctx(s))
    aligned = np.concatenate((np.zeros(d, np.float32), x[:-d]))
    # out = (1 - g) * target + g * aligned, target = 0.5 * aligned, g = room_gain(s).
    g = room_gain(s)
    np.testing.assert_allclose(y, (0.5 * (1 - g) + g) * aligned, atol=1e-6)


def test_streaming_equivalent_mixer():
    x = make_test_signal(1.0)
    assert_streaming_equivalent("mixer", x)
    assert_streaming_equivalent(lambda: registry.Pipeline("split>mixer", [Split(delay_ms=10), Mixer()]), x)


@pytest.mark.parametrize("hop", [160, 97])
def test_streaming_equivalent_split_gtcrn_mixer(hop):
    x = make_test_signal(1.0, kind="speechlike")

    def factory():
        return registry.Pipeline(
            "split(delay_ms=32)>gtcrn>mixer",
            [Split(delay_ms=32), GTCRN(backend=MockBackend()), Mixer()],
        )

    assert_streaming_equivalent(factory, x, hop=hop)
    assert factory().latency_ms == pytest.approx(32.0)


def test_spec_builds_split_gtcrn_mixer():
    pipe = registry.build_pipeline("split(delay_ms=32)>gtcrn>mixer")
    assert [type(s) for s in pipe.stages] == [Split, GTCRN, Mixer]


def _max_jump(y: np.ndarray) -> float:
    """Largest second difference: ~ (2 pi f / sr)^2 * A for a smooth tone, ~ the step at a click."""
    return float(np.max(np.abs(np.diff(y, n=2))))


CLICK = 10 ** (-40 / 20)  # -40 dBFS


def test_mixer_slider_jump_is_click_free():
    """Slider 0 -> 100 in one chunk while the residual is a tone: no discontinuity > -40 dBFS."""
    t = np.arange(SR) / SR
    residual = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    target = np.zeros_like(residual)
    mixer, outs = Mixer(), []
    for i in range(0, SR, 160):
        ctx = _ctx(0.0 if i < SR // 2 else 1.0)
        ctx.telemetry.extras[RESIDUAL_KEY] = residual[i : i + 160]
        outs.append(mixer.process(_chunk(target[i : i + 160]), ctx).samples)
    y = np.concatenate(outs)
    assert _max_jump(y) < CLICK
    # The ramp lasts SLIDER_RAMP_MS (>= 20 ms) and then the residual is gone.
    ramp = round(SLIDER_RAMP_MS * SR / 1000)
    assert SLIDER_RAMP_MS >= 20
    assert np.max(np.abs(y[SR // 2 + ramp :])) == 0.0
    assert np.max(np.abs(y[SR // 2 : SR // 2 + ramp // 2])) > 0.1
    # Without the ramp the same jump clicks (the test can see a click).
    snap, outs = Mixer(ramp_ms=0.0), []
    for i in range(0, SR, 160):
        ctx = _ctx(0.0 if i < SR // 2 else 1.0)
        ctx.telemetry.extras[RESIDUAL_KEY] = residual[i : i + 160]
        outs.append(snap.process(_chunk(target[i : i + 160]), ctx).samples)
    assert _max_jump(np.concatenate(outs)) > CLICK


@pytest.mark.parametrize("s", [0.5, 1.0])
def test_max_duck_db_bounds_the_residual(s):
    """T-095: the enrolled tier's mixer keeps the residual >= -20 dB, so a missed TSE window
    plays the target quieter instead of a hole; a slider gentler than the cap is unchanged."""
    target = np.zeros(160, np.float32)
    residual = np.full(160, 0.5, np.float32)
    ctx = _ctx(s)
    ctx.telemetry.extras[RESIDUAL_KEY] = residual
    out = Mixer(max_duck_db=20).process(_chunk(target), ctx)
    np.testing.assert_allclose(out.samples, 0.5 * max(room_gain(s), 0.1), rtol=1e-5)
    stage = registry.build_pipeline("mixer(max_duck_db=20)").stages[0]
    assert stage.max_duck_db == 20.0 and stage.name == "mixer(max_duck_db=20)"
    with pytest.raises(ValueError):
        Mixer(max_duck_db=-1)
