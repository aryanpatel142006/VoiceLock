"""DeepFilterNet3 stage: offline contract checks (mock backend) plus slow real-weight tests."""

import math

import numpy as np
import pytest

from hearing import registry
from hearing.enhance.dfn import (
    CONFIGS,
    DFN,
    DfState,
    Downsampler,
    MockBackend,
    Upsampler,
    design_lowpass,
    erb_widths,
    norm_alpha,
    vorbis_window,
)
from hearing.fusion.mixer import atten_lim_to_mix, strength_to_db
from hearing.testing import assert_streaming_equivalent, make_test_signal, run_whole
from hearing.types import AudioChunk, StageContext


def factory(variant="dfn", **kwargs):
    return lambda: DFN(variant, backend=MockBackend(variant, **kwargs))


RESAMPLER_DELAY = (len(design_lowpass()) - 1) // 3  # 48 samples at 16 kHz (3 ms)


def roundtrip(x):
    """The stage's own 16 -> 48 -> 16 kHz path without any model, delay-compensated."""
    taps = design_lowpass()
    return Downsampler(3, taps).process(Upsampler(3, taps).process(x))[RESAMPLER_DELAY:]


def expected_identity(stage, x, gain=1.0):
    """What an identity model must produce: the resampler round trip (with its own 3 ms
    delay and pre-ringing) shifted by the STFT hop and the model lookahead."""
    taps = design_lowpass()
    full = Downsampler(3, taps).process(Upsampler(3, taps).process(x))
    shift = stage.latency_samples - RESAMPLER_DELAY
    out = np.zeros(len(x), dtype=np.float32)
    out[shift:] = gain * full[: len(x) - shift]
    return out


@pytest.fixture(scope="module")
def dfn_weights():
    """Provision the pinned archive (7.9 MB) and the streaming graphs once per models dir."""
    from hearing.enhance.dfn_graphs import provision

    return provision("dfn")


# --- slider mapping -----------------------------------------------------------------------


def test_strength_to_db_mapping():
    assert strength_to_db(0.0) == 0.0
    assert strength_to_db(1.0) is None
    assert strength_to_db(0.5) == pytest.approx(15.0)
    assert strength_to_db(0.01) == pytest.approx(0.3)
    assert atten_lim_to_mix(None) == 0.0
    assert atten_lim_to_mix(0.0) == 1.0
    assert atten_lim_to_mix(20.0) == pytest.approx(0.1)
    for bad in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError):
            strength_to_db(bad)


# --- DSP ports ----------------------------------------------------------------------------


def test_erb_widths_match_upstream():
    widths = erb_widths(48000, 960, 32, 2)
    assert widths.sum() == 481
    assert widths[:13].tolist() == [2] * 13
    assert widths[-1] == 67


def test_norm_alpha_and_window():
    assert norm_alpha(48000, 480, 1.0) == pytest.approx(0.99)
    w = vorbis_window(960).astype(np.float64)
    np.testing.assert_allclose(w[:480] ** 2 + w[480:] ** 2, 1.0, atol=1e-6)


def test_stft_roundtrip_is_one_hop_delay():
    state = DfState(CONFIGS["dfn"])
    x = make_test_signal(0.5, sr=48000, kind="noise")
    out = np.concatenate(
        [state.synthesis(state.analysis(x[i : i + 480])) for i in range(0, len(x), 480)]
    )
    np.testing.assert_allclose(out[480:], x[:-480], atol=1e-6)


def test_resampler_delay_and_quality():
    taps = design_lowpass()
    assert len(taps) == 145 and (len(taps) - 1) % 3 == 0
    t = np.arange(16000) / 16000
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    y = roundtrip(tone)  # delay-compensated: len(tone) - RESAMPLER_DELAY samples
    np.testing.assert_allclose(y[500:], tone[500 : len(y)], atol=2e-3)
    # Chunked == whole for both directions.
    up = Upsampler(3, taps)
    chunked = np.concatenate(
        [up.process(tone[i : i + 160]) for i in range(0, 16000, 160)]
    )
    np.testing.assert_array_equal(chunked, Upsampler(3, taps).process(tone))


# --- stage contract (mock backend) --------------------------------------------------------


@pytest.mark.parametrize("variant", ["dfn", "dfn_ll"])
@pytest.mark.parametrize("hop", [160, 320, 1600])
def test_streaming_equivalent_mock(variant, hop):
    assert_streaming_equivalent(
        factory(variant), make_test_signal(0.41, kind="noise"), hop=hop, atol=1e-6
    )


@pytest.mark.parametrize("variant,latency_ms", [("dfn", 33.0), ("dfn_ll", 13.0)])
def test_identity_delay_latency_and_metadata(variant, latency_ms):
    stage = factory(variant)()
    assert stage.latency_ms == latency_ms
    x = make_test_signal(0.6, kind="speechlike") * 0.5
    ctx = StageContext.default()
    ctx.suppression = 1.0
    out = stage.process(AudioChunk(x, 16000, 42, 3.25), ctx)
    assert (out.seq, out.capture_ts, out.sample_rate) == (42, 3.25, 16000)
    assert out.samples.dtype == np.float32 and len(out.samples) == len(x)
    delay = stage.latency_samples
    expected = roundtrip(x)
    np.testing.assert_allclose(out.samples[:delay], 0, atol=1e-5)
    np.testing.assert_allclose(
        out.samples[delay:], expected[: len(x) - delay], atol=1e-5
    )
    assert ctx.telemetry.extras[variant]["mock"] is True
    assert ctx.telemetry.extras[variant]["atten_lim_db"] is None
    stage.reset()
    np.testing.assert_array_equal(
        stage.process(AudioChunk(x, 16000, 0, 0.0), ctx).samples, out.samples
    )


def test_bypass_ignores_the_model_and_keeps_latency():
    """suppression 0 -> 0 dB limit: the (delayed) noisy signal passes, the model is kept warm."""
    stage = factory("dfn", gain=0.0)()  # a model that would zero everything
    ctx = StageContext.default()
    ctx.suppression = 0.0
    x = make_test_signal(0.4, kind="noise") * 0.5
    out = run_whole(stage, x, 16000, ctx)
    np.testing.assert_allclose(out, expected_identity(stage, x), atol=1e-5)
    assert ctx.telemetry.extras["dfn"]["atten_lim_db"] == 0.0
    assert stage._backend.calls == len(x) // 160


def test_attenuation_limit_mixes_noisy_and_enhanced():
    x = make_test_signal(0.4, kind="noise") * 0.5
    ctx = StageContext.default()
    ctx.suppression = 0.5  # 15 dB limit -> mix 10^(-15/20) of the noisy signal
    out = run_whole(factory("dfn", gain=0.0)(), x, 16000, ctx)
    delay = DFN("dfn", backend=MockBackend()).latency_samples
    mix = atten_lim_to_mix(15.0)
    np.testing.assert_allclose(
        out[delay:], mix * roundtrip(x)[: len(x) - delay], atol=1e-5
    )
    ctx.suppression = 1.0
    out = run_whole(factory("dfn", gain=0.0)(), x, 16000, ctx)
    np.testing.assert_allclose(out[delay + 200 :], 0, atol=1e-5)


def test_fixed_strength_kwarg_overrides_context():
    ctx = StageContext.default()
    ctx.suppression = 1.0
    x = make_test_signal(0.2, kind="noise") * 0.5
    silent = run_whole(
        DFN("dfn", backend=MockBackend(gain=0.0), strength=0.0), x, 16000, ctx
    )
    assert np.abs(silent).max() > 0.1


def test_registration_pipeline_grammar_and_independent_state():
    stage = registry.get("dfn_ll", backend=MockBackend("dfn_ll"))
    assert stage.name == "dfn_ll" and stage.sample_rate == 16000
    import hearing.audio.basic  # noqa: F401  (registers passthrough)

    pipeline = registry.build_pipeline("passthrough>dfn(strength=0.5)")
    assert pipeline.latency_ms == 33.0
    a, b = factory()(), factory()()
    x = make_test_signal(0.2, kind="noise")
    run_whole(a, x, 16000)
    np.testing.assert_array_equal(
        run_whole(b, x, 16000), run_whole(factory()(), x, 16000)
    )


@pytest.mark.parametrize(
    "samples,sr",
    [
        (np.zeros(160), 48000),
        (np.zeros((2, 160)), 16000),
        (np.full(160, np.nan), 16000),
    ],
)
def test_invalid_audio(samples, sr):
    with pytest.raises(ValueError):
        run_whole(factory()(), samples.astype(np.float32), sr)


def test_partial_hop_chunks_are_rejected_and_empty_passes():
    stage = factory()()
    with pytest.raises(ValueError, match="whole 10 ms hops"):
        run_whole(stage, np.zeros(100, np.float32), 16000)
    assert len(run_whole(stage, np.zeros(0, np.float32), 16000)) == 0


def test_default_backend_is_lazy(monkeypatch):
    from hearing.enhance import dfn

    calls = []

    def load(name):
        calls.append(name)
        return MockBackend(name)

    monkeypatch.setattr(dfn.models, "load", load)
    stage = DFN("dfn_ll")
    assert not calls
    run_whole(stage, make_test_signal(0.1, kind="noise"), 16000)
    run_whole(stage, make_test_signal(0.1, kind="noise"), 16000)
    assert calls == ["dfn_ll"]


def test_missing_weights_fail_visibly(monkeypatch, tmp_path):
    from hearing.enhance.dfn import OnnxBackend

    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="provision"):
        OnnxBackend("dfn")


# --- real weights -------------------------------------------------------------------------


@pytest.mark.slow
def test_streaming_graphs_match_whole_sequence_graphs(dfn_weights):
    """Per-frame state passing reproduces the upstream whole-sequence graphs bit-exactly."""
    import onnxruntime as ort

    from hearing.enhance import dfn_graphs as g

    manifest = g.load_manifest("dfn")
    original = g.original_graphs("dfn")
    paths = g.streaming_paths("dfn")
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    rng = np.random.default_rng(0)
    frames = 12
    feat_erb = rng.normal(size=(1, 1, frames, 32)).astype(np.float32)
    feat_spec = rng.normal(size=(1, 2, frames, 96)).astype(np.float32)
    ref_enc = ort.InferenceSession(
        original["enc"], opts, providers=["CPUExecutionProvider"]
    )
    ref_outputs = ref_enc.run(None, {"feat_erb": feat_erb, "feat_spec": feat_spec})
    enc = ort.InferenceSession(
        str(paths["enc"]), opts, providers=["CPUExecutionProvider"]
    )
    state = {
        name: np.zeros(shape, np.float32)
        for name, _, shape in manifest["states"]["enc"]
    }
    outs = [o.name for o in ref_enc.get_outputs()]
    streamed = []
    for t in range(frames):
        feed = {
            "feat_erb": feat_erb[:, :, t : t + 1],
            "feat_spec": feat_spec[:, :, t : t + 1],
        }
        result = enc.run(
            outs + [out for _, out, _ in manifest["states"]["enc"]], feed | state
        )
        streamed.append(result[: len(outs)])
        state = dict(zip(state, result[len(outs) :], strict=True))
    for i, name in enumerate(outs):
        axis = 1 if name in ("emb", "lsnr") else 2
        joined = np.concatenate([s[i] for s in streamed], axis=axis)
        np.testing.assert_array_equal(joined, ref_outputs[i], err_msg=name)


@pytest.mark.slow
def test_real_streaming_equivalent(dfn_weights):
    ctx = StageContext.default()
    ctx.suppression = 0.7
    x = make_test_signal(0.8, kind="speechlike") * 0.5
    x = x + 0.05 * make_test_signal(0.8, kind="noise", seed=3)
    assert_streaming_equivalent("dfn", x.astype(np.float32), atol=1e-4, ctx=ctx)
    assert_streaming_equivalent(
        "dfn", x.astype(np.float32), hop=320, atol=1e-4, ctx=ctx
    )


@pytest.mark.slow
def test_real_benchmark(dfn_weights, capsys):
    """Synthetic smoke benchmark: speech-like bursts in white noise, not a corpus claim."""
    import json

    from hearing.enhance.dfn import benchmark

    result = benchmark("dfn", seconds=2.0)
    rows = result.pop("rows")
    assert result["rtf"] < 0.5
    assert result["si_sdri_by_snr"]["0.0"] > 0
    assert all(r["si_sdri"] > 0 for r in rows if r["snr_db"] == 0.0)
    with capsys.disabled():
        print("\nDFN3 benchmark: " + json.dumps(result, sort_keys=True))


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["dfn", "dfn_ll"])
def test_real_tiny_noise_bank(dfn_weights, variant, capsys):
    """Committed LibriSpeech tiny talkers + white/babble noise at 0 dB SNR, slider 0.5."""
    from hearing.enhance.dfn import benchmark, tiny_noise_bank
    from hearing.enhance.dfn_graphs import provision

    provision(variant)
    clean, noises = tiny_noise_bank()
    result = benchmark(variant, clean=clean, noises=noises, snrs_db=(0.0,), strength=0.5)
    rows = result.pop("rows")
    assert result["rtf"] < 0.5
    assert result["si_sdri"] > 0
    assert all(r["si_sdri"] > 3 for r in rows if r["noise"] == "white")
    with capsys.disabled():
        print(f"\n{variant} tiny noise bank @ 0 dB: si_sdri {result['si_sdri']:.2f}")


def test_slider_jump_during_tone_is_click_free():
    """Slider 0 -> 100 mid-tone: DFN ramps its mix-back over >= 20 ms, no step > -40 dBFS (T-093)."""
    sr = 16000
    t = np.arange(sr) / sr
    x = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    def run(stage):
        outs = []
        for i in range(0, sr, 160):
            ctx = StageContext.default()
            ctx.suppression = 0.0 if i < sr // 2 else 1.0
            outs.append(stage.process(AudioChunk(x[i : i + 160], sr, i // 160, 0.0), ctx).samples)
        return np.concatenate(outs)

    y = run(DFN("dfn_ll", backend=MockBackend("dfn_ll", gain=0.0)))
    jump = float(np.max(np.abs(np.diff(y[800:], n=2))))  # skip the resampler start-up
    assert jump < 10 ** (-40 / 20)
    assert np.max(np.abs(y[-2000:])) < 1e-3  # the tone is gone at 100 (mock removes everything)
