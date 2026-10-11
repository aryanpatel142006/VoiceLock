"""Offline contract checks plus opt-in tests against the published ONNX weights."""

import numpy as np
import pytest

from hearing import registry
from hearing.enhance.gtcrn import GTCRN, MockBackend
from hearing.testing import assert_streaming_equivalent, make_test_signal, run_whole
from hearing.types import AudioChunk, StageContext


def factory():
    return GTCRN(backend=MockBackend())


@pytest.fixture(scope="module")
def gtcrn_weights():
    """Provision the pinned, checksum-verified ONNX (~0.5 MB) once per models dir.

    Fresh worktrees have an empty ``models/``; this downloads only when missing,
    so slow runs work without a manual step and stay offline afterwards.
    """
    from hearing.enhance.gtcrn import download_model

    return download_model()


@pytest.mark.parametrize("hop", [1, 160, 256, 731])
def test_streaming_equivalent(hop):
    assert_streaming_equivalent(factory, make_test_signal(0.217, kind="noise"), hop=hop)


def test_identity_stft_delay_and_metadata():
    stage = factory()
    x = make_test_signal(0.4, kind="noise")
    chunk = AudioChunk(x, 16000, 42, 3.25)
    out = stage.process(chunk, StageContext.default())
    assert (out.seq, out.capture_ts, out.sample_rate) == (42, 3.25, 16000)
    assert out.samples.dtype == np.float32
    assert stage.latency_ms == 32
    np.testing.assert_allclose(out.samples[:512], 0, atol=1e-6)
    np.testing.assert_allclose(out.samples[512:], x[:-512], atol=1e-6)
    stage.reset()
    np.testing.assert_array_equal(
        stage.process(chunk, StageContext.default()).samples, out.samples
    )


def test_bypass_and_empty():
    stage = factory()
    ctx = StageContext.default()
    ctx.suppression = 0
    x = make_test_signal(0.1, kind="noise")
    np.testing.assert_array_equal(run_whole(stage, x, 16000, ctx), x)
    assert len(run_whole(stage, np.zeros(0, np.float32), 16000)) == 0


def test_registration_and_independent_state():
    a, b = registry.get("gtcrn", backend=MockBackend()), factory()
    x = make_test_signal(0.1, kind="noise")
    run_whole(a, x, 16000)
    np.testing.assert_array_equal(
        run_whole(b, x, 16000), run_whole(factory(), x, 16000)
    )


@pytest.mark.parametrize(
    "samples,sr",
    [(np.zeros(5), 48000), (np.zeros((2, 4)), 16000), (np.array([np.nan]), 16000)],
)
def test_invalid_audio(samples, sr):
    with pytest.raises(ValueError):
        run_whole(factory(), samples, sr)


@pytest.mark.slow
def test_real_streaming_equivalent(gtcrn_weights):
    assert_streaming_equivalent("gtcrn", make_test_signal(0.53, kind="noise"))


def test_strength_uses_aligned_dry():
    class HalfBackend(MockBackend):
        def infer(self, spectrum, state):
            return spectrum * 0.5, state

    x = make_test_signal(0.2, kind="noise")
    ctx = StageContext.default()
    ctx.suppression = 0.4
    y = run_whole(GTCRN(backend=HalfBackend()), x, 16000, ctx)
    np.testing.assert_allclose(y[512:], 0.8 * x[:-512], atol=1e-6)


def test_recurrent_state_is_carried_and_reset():
    class StatefulBackend(MockBackend):
        def infer(self, spectrum, state):
            count = state[0].flat[0]
            return spectrum / (1 + count), [s + 1 for s in state]

    def make():
        return GTCRN(backend=StatefulBackend())

    x = make_test_signal(0.3, kind="noise")
    assert_streaming_equivalent(make, x)
    stage = make()
    first = run_whole(stage, x, 16000)
    assert stage._state[0].flat[0] == len(x) // 256
    assert not np.allclose(first, run_whole(stage, x, 16000))
    stage.reset()
    np.testing.assert_array_equal(first, run_whole(stage, x, 16000))


def test_missing_weights_and_corruption_fail_visibly(monkeypatch, tmp_path):
    from hearing.enhance.gtcrn import OnnxBackend, model_path

    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="provision"):
        OnnxBackend()
    path = model_path()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not weights")
    with pytest.raises(ValueError, match="checksum"):
        OnnxBackend()


def test_default_backend_is_lazy(monkeypatch):
    from hearing.enhance import gtcrn

    calls = []

    def load(name):
        calls.append(name)
        return MockBackend()

    monkeypatch.setattr(gtcrn.models, "load", load)
    stage = GTCRN()
    assert not calls
    run_whole(stage, make_test_signal(0.1, kind="noise"), 16000)
    run_whole(stage, make_test_signal(0.1, kind="noise"), 16000)
    assert calls == ["gtcrn"]


@pytest.mark.slow
def test_real_benchmark(gtcrn_weights, capsys):
    """Reproducible synthetic smoke benchmark, not a speech-corpus quality claim."""
    import json

    from hearing.enhance.gtcrn import benchmark

    result = benchmark()
    assert result["rtf"] < 0.5
    assert result["si_sdri"] > 3
    with capsys.disabled():
        print("\nGTCRN benchmark: " + json.dumps(result, sort_keys=True))
