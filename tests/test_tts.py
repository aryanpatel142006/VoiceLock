import base64
import json
import os
import queue

import numpy as np
import pytest

from hearing.audio.io import PIPELINE_SR, float_to_pcm16
from hearing.synth import tts
from hearing.synth.tts import (
    ElevenLabsTTS,
    FallbackTTS,
    KokoroTTS,
    MockTTS,
    TtsEngine,
    TtsError,
    bucket_names,
    get_tts,
    phrase_latency,
    resolve_voice,
)

# --- voices.yaml -----------------------------------------------------------


def test_voices_yaml_has_neutral_and_contract_buckets():
    names = bucket_names()
    assert names[0] == "neutral"
    assert {"female_adult", "male_adult"} <= set(names)


def test_every_bucket_maps_every_engine():
    for b in bucket_names():
        assert resolve_voice(b, "elevenlabs")
        assert resolve_voice(b, "local")
        assert float(resolve_voice(b, "mock")) > 50


def test_resolve_voice_defaults_and_passthrough():
    assert resolve_voice(None, "local") == resolve_voice("neutral", "local")
    # A raw engine voice id (not a bucket) passes through unchanged.
    assert resolve_voice("af_sky", "local") == "af_sky"
    with pytest.raises(ValueError):
        resolve_voice("neutral", "nope")


# --- MockTTS ---------------------------------------------------------------


def test_mock_is_a_tts_engine():
    m = MockTTS()
    assert isinstance(m, TtsEngine)
    assert m.name == "mock" and m.sample_rate == PIPELINE_SR


def test_mock_one_tone_per_word_deterministic():
    m = MockTTS()
    chunks = list(m.stream("see you at noon", "neutral"))
    assert len(chunks) == 4
    assert all(c.dtype == np.float32 and c.ndim == 1 for c in chunks)
    assert all(np.abs(c).max() <= 1.0 for c in chunks)
    np.testing.assert_array_equal(m.synthesize("see you at noon", "neutral"), np.concatenate(chunks))
    np.testing.assert_array_equal(m.synthesize("see you", None), MockTTS().synthesize("see you", None))


def test_mock_longer_words_give_longer_tones():
    m = MockTTS()
    short, long_ = m.stream("a extraordinarily", None)
    assert len(long_) > len(short)


def test_mock_pitch_follows_voice_bucket():
    def pitch(voice):
        x = MockTTS().synthesize("hello", voice)
        spec = np.abs(np.fft.rfft(x))
        return np.fft.rfftfreq(len(x), 1 / PIPELINE_SR)[spec.argmax()]

    assert pitch("female_adult") > pitch("neutral") > pitch("male_adult")


def test_mock_empty_text_is_silent():
    assert list(MockTTS().stream("   ", None)) == []
    assert MockTTS().synthesize("", None).size == 0


# --- get_tts ---------------------------------------------------------------


def test_get_tts_mock_and_unknown():
    assert isinstance(get_tts("mock"), MockTTS)
    with pytest.raises(ValueError):
        get_tts("nope")


def test_get_tts_auto_offline_is_mock(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("HEARING_TTS", raising=False)
    monkeypatch.setattr(tts, "local_available", lambda: False)
    assert isinstance(get_tts(), MockTTS)


def test_get_tts_auto_with_key_is_elevenlabs_with_fallback(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.delenv("HEARING_TTS", raising=False)
    monkeypatch.setattr(tts, "local_available", lambda: False)
    engine = get_tts()
    assert isinstance(engine, FallbackTTS)
    assert isinstance(engine.primary, ElevenLabsTTS)
    assert isinstance(engine.fallback, MockTTS)


def test_get_tts_auto_local_when_installed(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("HEARING_TTS", raising=False)
    monkeypatch.setattr(tts, "local_available", lambda: True)
    engine = get_tts()
    assert isinstance(engine, FallbackTTS) and isinstance(engine.primary, KokoroTTS)


def test_get_tts_env_override(monkeypatch):
    monkeypatch.setenv("HEARING_TTS", "mock")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert isinstance(get_tts(), MockTTS)


def test_get_tts_elevenlabs_without_key_raises(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(TtsError):
        get_tts("elevenlabs")


# --- FallbackTTS -----------------------------------------------------------


class _Broken:
    name = "broken"
    sample_rate = PIPELINE_SR

    def __init__(self, after=0):
        self.after = after

    def stream(self, text, voice):
        for _ in range(self.after):
            yield np.ones(10, np.float32) * 0.1
        raise TtsError("down")

    def synthesize(self, text, voice):
        return np.concatenate(list(self.stream(text, voice)))

    def close(self):
        pass


def test_fallback_before_audio_uses_fallback_for_the_phrase():
    f = FallbackTTS(_Broken(), MockTTS())
    out = f.synthesize("hi there", "neutral")
    np.testing.assert_array_equal(out, MockTTS().synthesize("hi there", "neutral"))
    assert f.degraded and "down" in f.last_error and f.name == "mock"


def test_fallback_mid_phrase_truncates_then_degrades():
    f = FallbackTTS(_Broken(after=2), MockTTS())
    out = f.synthesize("hi there", "neutral")
    assert len(out) == 20  # no duplicated audio for the words already spoken
    assert f.degraded
    np.testing.assert_array_equal(f.synthesize("ok", None), MockTTS().synthesize("ok", None))


# --- ElevenLabsTTS protocol (offline, fake socket) -------------------------


class FakeSocket:
    def __init__(self, replies):
        self.sent: list[dict] = []
        self.inbox: queue.Queue = queue.Queue()
        self.replies = replies
        self.closed = False

    def send(self, msg):
        self.sent.append(json.loads(msg))
        if self.sent[-1].get("text") == "":
            for r in self.replies:
                self.inbox.put(json.dumps(r))

    def recv(self, timeout=None):
        try:
            return self.inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None

    def close(self):
        self.closed = True


def _audio_msg(x):
    return {"audio": base64.b64encode(float_to_pcm16(x)).decode(), "isFinal": None}


def test_elevenlabs_streams_pcm_from_flash_websocket():
    a = np.linspace(-0.5, 0.5, 320, dtype=np.float32)
    b = np.full(160, 0.25, np.float32)
    sock = FakeSocket([_audio_msg(a), _audio_msg(b), {"audio": None, "isFinal": True}])
    calls = []

    def connect(url, headers):
        calls.append((url, headers))
        return sock

    e = ElevenLabsTTS(api_key="k", connect=connect)
    chunks = list(e.stream("see you at noon", "male_adult"))
    url, headers = calls[0]
    voice_id = resolve_voice("male_adult", "elevenlabs")
    assert url.startswith(f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?")
    assert "model_id=eleven_flash_v2_5" in url and "output_format=pcm_16000" in url
    assert headers == {"xi-api-key": "k"}
    assert sock.sent[0]["text"] == " "
    assert sock.sent[1]["text"] == "see you at noon " and sock.sent[1]["flush"] is True
    assert sock.sent[-1] == {"text": ""}
    assert [len(c) for c in chunks] == [320, 160]
    np.testing.assert_allclose(np.concatenate(chunks), np.concatenate([a, b]), atol=1e-4)
    assert sock.closed


def test_elevenlabs_server_error_raises_tts_error():
    sock = FakeSocket([{"message": "invalid api key", "error": "auth_error", "code": 1008}])
    e = ElevenLabsTTS(api_key="k", connect=lambda url, headers: sock)
    with pytest.raises(TtsError, match="invalid api key"):
        list(e.stream("hi", None))


def test_elevenlabs_connect_failure_raises_tts_error():
    def connect(url, headers):
        raise OSError("no route")

    with pytest.raises(TtsError):
        ElevenLabsTTS(api_key="k", connect=connect).synthesize("hi", None)


def test_elevenlabs_timeout_raises_tts_error():
    sock = FakeSocket([])  # never answers
    e = ElevenLabsTTS(api_key="k", connect=lambda url, headers: sock, recv_timeout_s=0.05)
    with pytest.raises(TtsError, match="timed out"):
        list(e.stream("hi", None))


# --- latency ---------------------------------------------------------------


def test_phrase_latency_reports_first_audio_and_rtf():
    r = phrase_latency(MockTTS(), "see you at noon by the fountain", "neutral")
    assert r["engine"] == "mock"
    assert 0 <= r["first_audio_ms"] <= r["total_ms"] < 50
    assert r["audio_s"] > 1.0 and r["rtf"] < 0.05


@pytest.mark.slow
@pytest.mark.skipif(not tts.local_available(), reason="kokoro-onnx or its weights missing")
def test_kokoro_speaks_at_16k():
    k = KokoroTTS()
    x = k.synthesize("see you at noon", "female_adult")
    assert x.dtype == np.float32 and 0.5 < len(x) / PIPELINE_SR < 4.0
    assert 0.01 < np.abs(x).max() <= 1.0


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("ELEVENLABS_API_KEY"), reason="needs ELEVENLABS_API_KEY")
def test_elevenlabs_live_phrase_latency():
    r = phrase_latency(ElevenLabsTTS(), "see you at noon by the fountain", "neutral")
    print(r)
    assert r["audio_s"] > 0.5 and r["first_audio_ms"] < 3000


def test_split_sentences_cuts_at_sentence_ends_and_long_commas():
    from hearing.synth.tts import split_sentences

    assert split_sentences("They're new. I can't do it.") == ["They're new.", "I can't do it."]
    assert split_sentences("one two three four five six seven eight nine, ten", first_words=0) == [
        "one two three four five six seven eight nine,", "ten"]
    assert split_sentences("short, one.") == ["short, one."]
    assert split_sentences("  ") == []


def test_kokoro_stream_yields_one_chunk_per_sentence(monkeypatch):
    calls = []

    class Fake:
        def create(self, text, voice, speed, lang):
            calls.append(text)
            return np.zeros(2400, np.float32), 24000

    monkeypatch.setattr(tts.models, "load", lambda name: Fake())
    chunks = list(KokoroTTS().stream("They're new. I can't do it. Really now.", None))
    assert len(chunks) == 3 and calls == ["They're new.", "I can't do it.", "Really now."]


# --- T-094: voice table, first-chunk split, ElevenLabs mock, captions -------------------


def test_local_voices_distinct_per_bucket_and_speed():
    ids = [resolve_voice(b, "local") for b in bucket_names()]
    assert len(set(ids)) == len(ids), ids  # no two buckets share a Kokoro voice
    assert tts.local_speed("female_senior") < 1.0 < tts.local_speed("female_young")
    assert tts.local_speed("female_adult") == tts.local_speed(None) or tts.local_speed(None) == 1.0
    assert tts.local_speed("some_raw_id") == 1.0
    eleven = [resolve_voice(b, "elevenlabs") for b in bucket_names()]
    assert len(set(eleven)) == len(eleven)


def test_first_chunk_is_short_and_text_is_kept():
    text = "Meet me by the fountain at noon, and bring your umbrella."
    parts = tts.split_sentences(text, first_words=3)
    assert len(parts[0].split()) == 3
    assert " ".join(parts) == text
    assert tts.split_sentences("Hi there.", first_words=3) == ["Hi there."]
    assert tts.split_sentences(text, first_words=0)[0].startswith("Meet me by the fountain")


def test_mock_elevenlabs_streams_through_the_real_adapter():
    e = tts.mock_elevenlabs()
    r = tts.phrase_latency(e, "see you at noon by the fountain", "male_adult")
    assert 80 <= r["first_audio_ms"] < 400  # Flash v2.5-like: ~100 ms to the first chunk
    assert r["audio_s"] > 1.0
    assert tts.get_tts("elevenlabs-mock").name == "elevenlabs"


def test_synth_test_renders_every_bucket(tmp_path):
    rows = tts.synth_test(MockTTS(), out_dir=tmp_path)
    assert [r["bucket"] for r in rows] == bucket_names()
    assert all((tmp_path / f"mock_{r['bucket']}.wav").exists() for r in rows)


def test_cli_synth_test_mock(capsys):
    from hearing.cli import main

    assert main(["synth-test", "--tts", "elevenlabs-mock"]) == 0
    assert "male_adult" in capsys.readouterr().out
