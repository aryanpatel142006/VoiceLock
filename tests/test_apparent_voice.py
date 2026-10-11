import itertools

import numpy as np
import pytest

from hearing.synth import apparent_voice as av
from hearing.synth import voice_match
from hearing.types import AudioChunk

HOP = 160


class StubModel:
    def __init__(self, age=35.0, probs=(0.9, 0.08, 0.02)):
        self.age, self.probs, self.calls = age, np.array(probs), 0

    def predict(self, samples):
        self.calls += 1
        return self.age, self.probs


def chunk(i=0):
    return AudioChunk(np.full(HOP, 0.1, np.float32), 16000, i, i * 0.01)


def feed(r, seconds, speaking=True, start=0):
    out = []
    for i in range(int(seconds * 100)):
        e = r.update(chunk(start + i), speaking)
        if e is not None:
            out.append(start + i)
    return out


def rolling(**kw):
    return av.RollingVoiceEstimator(av.VoiceEstimator(StubModel()), **kw)


def test_short_input_none():
    est = av.VoiceEstimator(StubModel())
    assert est.estimate(np.zeros(15999, np.float32)) is None
    assert est.estimate(np.zeros(16000, np.float32)) is not None


def test_outputs_in_range_and_source():
    for age in (5, 25, 45, 80):
        for probs in (
            (0.9, 0.05, 0.05),
            (0.05, 0.9, 0.05),
            (0.1, 0.1, 0.8),
            (0.34, 0.33, 0.33),
        ):
            e = av.VoiceEstimator(StubModel(age, probs)).estimate(
                np.zeros(48000, np.float32)
            )
            assert e.source == "voice"
            assert 0 <= e.p_female <= 1 and 0 <= e.confidence <= 1
            assert abs(sum(e.p_age) - 1) < 1e-6 and min(e.p_age) >= 0


def test_mapping_semantics():
    f = av.VoiceEstimator(StubModel(25, (0.95, 0.03, 0.02))).estimate(
        np.zeros(48000, np.float32)
    )
    m = av.VoiceEstimator(StubModel(70, (0.03, 0.95, 0.02))).estimate(
        np.zeros(48000, np.float32)
    )
    c = av.VoiceEstimator(StubModel(8, (0.1, 0.1, 0.8))).estimate(
        np.zeros(48000, np.float32)
    )
    assert f.p_female > 0.9 and m.p_female < 0.1
    assert np.argmax(m.p_age) == 2 and np.argmax(f.p_age) == 0
    assert c.confidence < f.confidence / 2
    short = av.VoiceEstimator(StubModel()).estimate(np.zeros(16000, np.float32))
    assert (
        short.confidence
        < av.VoiceEstimator(StubModel())
        .estimate(np.zeros(48000, np.float32))
        .confidence
    )


def test_isinstance_protocol():
    assert isinstance(av.VoiceEstimator(), voice_match.VoiceEstimator)


def test_import_downloads_nothing():
    from hearing import models

    assert "apparent_voice" not in models._cache


def test_weights_unavailable_with_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    assert av.weights_available() is False
    assert voice_match.get_voice_estimator() is None


def test_no_estimate_before_window_then_cadence():
    r = rolling()
    assert feed(r, 2.99) == []
    emitted = feed(r, 6.0, start=299)
    assert emitted[0] == 299  # first estimate as the window fills (3.00 s)
    assert [b - a for a, b in itertools.pairwise(emitted)] == [200] * (len(emitted) - 1)
    assert len(emitted) == 3


def test_silence_not_counted_and_reset():
    r = rolling()
    feed(r, 2.0)
    assert feed(r, 10.0, speaking=False, start=200) == []
    r.reset()
    assert feed(r, 2.99) == []
    assert feed(r, 0.02, start=299) != []


def test_scattered_speech_rejected_by_voiced_ratio():
    r = rolling(min_voiced_ratio=0.5)
    out = []
    for i in range(1000):  # 10% voiced
        out += [i] if r.update(chunk(i), i % 10 == 0) is not None else []
    assert out == []


def test_stub_called_once_per_estimate():
    stub = StubModel()
    r = av.RollingVoiceEstimator(av.VoiceEstimator(stub))
    feed(r, 5.0)
    assert stub.calls == 2


@pytest.mark.slow
@pytest.mark.network
def test_real_model_inference():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from hearing import models

    try:
        est = av.VoiceEstimator()
        t = np.arange(48000) / 16000
        x = sum(np.sin(2 * np.pi * f * h * t) / h for f in (120,) for h in range(1, 20))
        x = (0.3 * x / np.abs(x).max()).astype(np.float32)
        e = est.estimate(x)
    except OSError as exc:
        pytest.skip(f"weights unavailable offline: {exc}")
    finally:
        models.clear()
    assert e is not None and e.source == "voice" and 0 <= e.p_female <= 1
