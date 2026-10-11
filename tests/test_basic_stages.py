"""Reference stages (passthrough, gain, delay) and the streaming-equivalence helper."""

from dataclasses import replace

import numpy as np
import pytest

import hearing.audio.basic  # noqa: F401  (registers the stages)
from hearing import registry
from hearing.testing import (
    assert_streaming_equivalent,
    make_test_signal,
    run_streaming,
    run_whole,
)
from hearing.types import AudioChunk, Stage, StageContext

SR = 16000


@pytest.fixture
def x():
    return make_test_signal(0.5, SR, kind="speechlike", seed=1)


@pytest.mark.parametrize("kind", ["speechlike", "tone", "noise"])
def test_make_test_signal_shape_range_and_determinism(kind):
    a = make_test_signal(0.3, SR, kind=kind, seed=3)
    b = make_test_signal(0.3, SR, kind=kind, seed=3)
    c = make_test_signal(0.3, SR, kind=kind, seed=4)
    assert a.dtype == np.float32 and a.shape == (int(0.3 * SR),)
    assert np.max(np.abs(a)) <= 1.0
    assert np.max(np.abs(a)) > 0.1
    np.testing.assert_array_equal(a, b)
    if kind != "tone":
        assert not np.array_equal(a, c)


def test_make_test_signal_speechlike_has_pauses():
    a = make_test_signal(2.0, SR, kind="speechlike")
    env = np.abs(a).reshape(-1, 160).max(axis=1)
    assert (env < 0.01).sum() > 10  # some silent 10 ms frames
    assert (env > 0.1).sum() > 50  # and plenty of active ones


def test_make_test_signal_rejects_unknown_kind():
    with pytest.raises(ValueError):
        make_test_signal(0.1, SR, kind="bogus")


def test_registered_names():
    for name in ("passthrough", "gain", "delay"):
        assert name in registry.names()
        assert isinstance(registry.get(name), Stage)


def test_passthrough_returns_identical_samples_and_metadata():
    stage = registry.get("passthrough")
    assert stage.latency_ms == 0.0 and stage.sample_rate == SR
    chunk = AudioChunk(np.full(160, 0.25, dtype=np.float32), SR, 7, 3.25)
    out = stage.process(chunk, StageContext.default())
    np.testing.assert_array_equal(out.samples, chunk.samples)
    assert (out.seq, out.capture_ts, out.sample_rate) == (7, 3.25, SR)


def test_gain_applies_db(x):
    stage = registry.get("gain", db=-6)
    out = run_whole(stage, x, SR)
    np.testing.assert_allclose(out, x * 10 ** (-6 / 20), atol=1e-6)
    assert out.dtype == np.float32
    assert registry.get("gain").latency_ms == 0.0
    np.testing.assert_array_equal(run_whole(registry.get("gain"), x, SR), x)


def test_delay_shifts_by_samples_and_reports_latency(x):
    stage = registry.get("delay", ms=30)
    assert stage.latency_ms == 30.0
    out = run_streaming(stage, x, SR, hop=160)
    assert len(out) == len(x)
    np.testing.assert_array_equal(out[:480], 0)
    np.testing.assert_allclose(out[480:], x[:-480], atol=1e-7)


def test_delay_reset_clears_state(x):
    stage = registry.get("delay", ms=10)
    run_streaming(stage, x, SR)
    stage.reset()
    out = run_streaming(stage, x, SR)
    np.testing.assert_array_equal(out[:160], 0)
    np.testing.assert_allclose(out[160:], x[:-160], atol=1e-7)


def test_delay_zero_is_passthrough(x):
    stage = registry.get("delay")
    assert stage.latency_ms == 0.0
    np.testing.assert_array_equal(run_streaming(stage, x, SR), x)


def test_delay_rejects_negative():
    with pytest.raises(ValueError):
        registry.get("delay", ms=-1)


def test_delay_handles_chunks_smaller_and_larger_than_delay(x):
    for hop in (37, 160, 1000, 4000):
        stage = registry.get("delay", ms=20)
        out = run_streaming(stage, x, SR, hop=hop)
        np.testing.assert_allclose(out[320:], x[:-320], atol=1e-7)
        np.testing.assert_array_equal(out[:320], 0)


def test_run_whole_and_run_streaming_shapes(x):
    stage = registry.get("passthrough")
    assert run_whole(stage, x, SR).shape == x.shape
    assert run_streaming(stage, x, SR, hop=160).shape == x.shape
    odd = x[:1000]
    assert run_streaming(registry.get("passthrough"), odd, SR, hop=160).shape == odd.shape


@pytest.mark.parametrize("spec", ["passthrough", "gain(db=-6)", "delay(ms=30)"])
def test_streaming_equivalent_passes_for_reference_stages(spec, x):
    assert_streaming_equivalent(lambda: registry.build_pipeline(spec), x, SR)


def test_streaming_equivalent_accepts_stage_name_string(x):
    assert_streaming_equivalent("gain(db=-6)>delay(ms=20)", x, SR)


class StatelessWrongDelay:
    """Deliberately broken: zero-pads every chunk instead of carrying state."""

    name = "wrong_delay"
    sample_rate = SR
    latency_ms = 30.0

    def process(self, chunk, ctx):
        n = 480
        s = chunk.samples
        out = np.concatenate([np.zeros(min(n, len(s)), np.float32), s[: max(len(s) - n, 0)]])
        return replace(chunk, samples=out.astype(np.float32))

    def reset(self):
        pass


def test_streaming_equivalent_fails_for_stateless_delay(x):
    with pytest.raises(AssertionError) as info:
        assert_streaming_equivalent(StatelessWrongDelay, x, SR)
    message = str(info.value)
    assert "max abs error" in message
    assert "first differing sample" in message
    assert "wrong_delay" in message


def test_streaming_equivalent_reports_length_mismatch(x):
    class Truncating:
        name = "trunc"
        sample_rate = SR
        latency_ms = 0.0

        def process(self, chunk, ctx):
            return replace(chunk, samples=chunk.samples[:100])

        def reset(self):
            pass

    with pytest.raises(AssertionError) as info:
        assert_streaming_equivalent(Truncating, x, SR)
    assert "length" in str(info.value)


def test_streaming_equivalent_honours_atol(x):
    class Jitter:
        name = "jitter"
        sample_rate = SR
        latency_ms = 0.0

        def __init__(self):
            self.calls = 0

        def process(self, chunk, ctx):
            self.calls += 1
            return replace(chunk, samples=chunk.samples + np.float32(1e-6 * (self.calls > 1)))

        def reset(self):
            self.calls = 0

    assert_streaming_equivalent(Jitter, x, SR, atol=1e-5)
    with pytest.raises(AssertionError):
        assert_streaming_equivalent(Jitter, x, SR, atol=1e-8)


def test_pipeline_reports_summed_latency_and_streams(x):
    pipeline = registry.build_pipeline("gain(db=-6)>delay(ms=20)")
    assert pipeline.latency_ms == 20.0
    out = run_streaming(pipeline, x, SR)
    np.testing.assert_allclose(out[320:], x[:-320] * 10 ** (-6 / 20), atol=1e-6)
