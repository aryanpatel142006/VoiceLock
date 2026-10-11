import base64
import json
import os
import queue
import time

import numpy as np
import pytest

from hearing.audio.io import HOP, PIPELINE_SR, pcm16_to_float
from hearing.audio.stream import chunk_stream
from hearing.synth import asr, tts
from hearing.synth.asr import (
    AsrError,
    ElevenLabsASR,
    FallbackASR,
    MockASR,
    StreamingASR,
    Transcript,
    get_asr,
)
from hearing.testing import make_test_signal
from hearing.types import AudioChunk


def _feed_all(engine, x, sr=PIPELINE_SR, hop=HOP):
    events = []
    for chunk in chunk_stream(x, sr, hop_ms=1000.0 * hop / sr):
        events += engine.feed(chunk)
    events += engine.flush()
    return events


def _voiced(seconds):
    return np.full(round(seconds * PIPELINE_SR), 0.1, np.float32)


def _finals(events):
    return [e for e in events if e.final]


# --- Transcript ------------------------------------------------------------


def test_transcript_to_caption_matches_wire_contract():
    t = Transcript("see you at noon", True, 12.31, 13.02)
    assert t.to_caption() == {
        "type": "caption",
        "text": "see you at noon",
        "final": True,
        "t0": 12.31,
        "t1": 13.02,
    }


# --- MockASR ---------------------------------------------------------------


def test_mock_is_a_streaming_asr():
    m = MockASR()
    assert isinstance(m, StreamingASR)
    assert m.name == "mock" and m.sample_rate == PIPELINE_SR


def test_mock_returns_the_transcript_over_voiced_audio():
    m = MockASR("see you at noon by the fountain", words_per_second=4.0)
    finals = _finals(_feed_all(m, _voiced(3.0)))
    assert " ".join(f.text for f in finals) == "see you at noon by the fountain"


def test_mock_is_deterministic_and_chunking_invariant():
    x = make_test_signal(4.0)
    a = _feed_all(MockASR(), x, hop=HOP)
    b = _feed_all(MockASR(), x, hop=4 * HOP)
    assert [e.text for e in _finals(a)] == [e.text for e in _finals(b)]
    assert a == _feed_all(MockASR(), x, hop=HOP)


def test_mock_is_silent_on_silence():
    assert _feed_all(MockASR(), np.zeros(PIPELINE_SR * 2, np.float32)) == []


def test_mock_emits_partials_that_grow_then_a_final():
    m = MockASR("one two three four", words_per_second=4.0, phrase_words=4)
    events = _feed_all(m, _voiced(1.5))
    partials = [e.text for e in events if not e.final]
    assert partials == ["one", "one two", "one two three"]
    assert _finals(events)[0].text == "one two three four"


def test_mock_splits_phrases_on_pause():
    m = MockASR("a b c d e f g h", words_per_second=4.0, phrase_words=99, pause_s=0.3)
    x = np.concatenate([_voiced(0.5), np.zeros(int(0.5 * PIPELINE_SR), np.float32), _voiced(0.5)])
    finals = _finals(_feed_all(m, x))
    assert [f.text for f in finals] == ["a b", "c d"]


def test_mock_timestamps_follow_capture_clock():
    m = MockASR("hello there", words_per_second=4.0)
    events = []
    for i, c in enumerate(chunk_stream(_voiced(1.0), PIPELINE_SR, hop_ms=10.0)):
        c.capture_ts = 100.0 + i * 0.01
        events += m.feed(c)
    events += m.flush()
    final = _finals(events)[0]
    assert final.t0 == pytest.approx(100.0, abs=0.02)
    assert final.t1 == pytest.approx(100.5, abs=0.02)


def test_mock_reset_restarts_transcript():
    m = MockASR("alpha beta", words_per_second=4.0)
    first = _feed_all(m, _voiced(1.0))
    m.reset()
    assert _feed_all(m, _voiced(1.0)) == first


def test_mock_from_fixture_reads_manifest_transcript(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"transcript": {"target": "go left here"}}))
    m = MockASR.from_fixture(tmp_path)
    finals = _finals(_feed_all(m, _voiced(2.0)))
    assert " ".join(f.text for f in finals) == "go left here"


# --- get_asr ---------------------------------------------------------------


def test_get_asr_mock_and_unknown():
    assert isinstance(get_asr("mock"), MockASR)
    with pytest.raises(ValueError):
        get_asr("nope")


def test_mock_transcript_kwarg_and_env_override(monkeypatch):
    long = "one two three four five six seven eight nine ten eleven twelve"
    monkeypatch.delenv("HEARING_MOCK_TRANSCRIPT", raising=False)
    assert get_asr("mock").words == asr.DEFAULT_TRANSCRIPT.split()
    m = get_asr("mock", transcript=long)
    assert " ".join(f.text for f in _finals(_feed_all(m, _voiced(6.0)))) == long
    monkeypatch.setenv("HEARING_MOCK_TRANSCRIPT", "alpha beta gamma")
    assert get_asr("mock").words == ["alpha", "beta", "gamma"]
    assert get_asr("mock", transcript=long).words == long.split()  # kwarg wins over env


def test_get_asr_auto_offline_is_mock(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("HEARING_ASR", raising=False)
    monkeypatch.setattr(asr, "local_available", lambda: False)
    assert isinstance(get_asr(), MockASR)


def test_get_asr_auto_with_key_is_elevenlabs_with_fallback(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.delenv("HEARING_ASR", raising=False)
    monkeypatch.setattr(asr, "local_available", lambda: False)
    engine = get_asr()
    assert isinstance(engine, FallbackASR)
    assert isinstance(engine.primary, ElevenLabsASR)
    assert isinstance(engine.fallback, MockASR)


def test_get_asr_env_override(monkeypatch):
    monkeypatch.setenv("HEARING_ASR", "mock")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    assert isinstance(get_asr(), MockASR)


def test_get_asr_elevenlabs_without_key_raises(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(AsrError):
        get_asr("elevenlabs")


# --- FallbackASR -----------------------------------------------------------


class _Broken:
    name = "broken"
    sample_rate = PIPELINE_SR

    def feed(self, chunk):
        raise AsrError("down")

    def flush(self):
        raise AsrError("down")

    def reset(self):
        pass

    def close(self):
        pass


def test_fallback_switches_on_error_and_reports_it():
    f = FallbackASR(_Broken(), MockASR("hi there", words_per_second=4.0))
    events = _feed_all(f, _voiced(1.0))
    assert f.degraded and "down" in f.last_error
    assert f.name == "mock"
    assert _finals(events)[0].text == "hi there"


# --- ElevenLabsASR protocol (offline, fake socket) -------------------------


class FakeSocket:
    """In-memory stand-in for a websockets sync connection."""

    def __init__(self):
        self.sent: list[dict] = []
        self.inbox: queue.Queue = queue.Queue()
        self.closed = False

    def send(self, msg):
        self.sent.append(json.loads(msg))
        m = self.sent[-1]
        if m.get("commit"):
            self.inbox.put(json.dumps({"message_type": "committed_transcript", "text": "hello world"}))

    def recv(self, timeout=None):
        try:
            item = self.inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if item is None:
            raise ConnectionError("closed")
        return item

    def close(self):
        self.closed = True
        self.inbox.put(None)


def _wait_for(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return False


def test_elevenlabs_sends_base64_pcm16_chunks_and_parses_transcripts():
    sock = FakeSocket()
    urls = []

    def connect(url, headers):
        urls.append((url, headers))
        return sock

    e = ElevenLabsASR(api_key="k", connect=connect, send_ms=50)
    x = _voiced(0.2)
    events = []
    for c in chunk_stream(x, PIPELINE_SR, hop_ms=10.0):
        events += e.feed(c)
    url, headers = urls[0]
    assert url.startswith("wss://api.elevenlabs.io/v1/speech-to-text/realtime?")
    assert "model_id=scribe_v2_realtime" in url and "audio_format=pcm_16000" in url
    assert headers == {"xi-api-key": "k"}
    audio = [m for m in sock.sent if m["message_type"] == "input_audio_chunk"]
    assert len(audio) == 4  # 200 ms in 50 ms batches
    pcm = np.concatenate([pcm16_to_float(base64.b64decode(m["audio_base_64"])) for m in audio])
    np.testing.assert_allclose(pcm, x, atol=1e-4)
    assert all(m["sample_rate"] == PIPELINE_SR for m in audio)

    sock.inbox.put(json.dumps({"message_type": "partial_transcript", "text": "hel"}))
    assert _wait_for(lambda: not e._events.empty())
    events += e.feed(AudioChunk(np.zeros(HOP, np.float32), PIPELINE_SR, 99, 0.0))
    assert [(ev.text, ev.final) for ev in events] == [("hel", False)]

    finals = e.flush()
    assert [(ev.text, ev.final) for ev in finals] == [("hello world", True)]
    e.close()
    assert sock.closed


def test_elevenlabs_server_error_raises_asr_error():
    sock = FakeSocket()
    e = ElevenLabsASR(api_key="k", connect=lambda url, headers: sock)
    e.feed(AudioChunk(np.zeros(HOP, np.float32), PIPELINE_SR, 0, 0.0))
    sock.inbox.put(json.dumps({"message_type": "auth_error", "error": "bad key"}))
    assert _wait_for(lambda: e._error is not None)
    with pytest.raises(AsrError, match="bad key"):
        e.feed(AudioChunk(np.zeros(HOP, np.float32), PIPELINE_SR, 1, 0.0))


def test_elevenlabs_reconnects_after_close():
    socks = []

    def connect(url, headers):
        socks.append(FakeSocket())
        return socks[-1]

    e = ElevenLabsASR(api_key="k", connect=connect, send_ms=10)
    e.feed(AudioChunk(np.zeros(HOP, np.float32), PIPELINE_SR, 0, 0.0))
    e.close()
    e.feed(AudioChunk(np.zeros(HOP, np.float32), PIPELINE_SR, 1, 0.01))
    assert len(socks) == 2 and e._thread.is_alive()
    assert [ev.text for ev in e.flush()] == ["hello world"]
    e.close()


def test_elevenlabs_connect_failure_raises_asr_error():
    def connect(url, headers):
        raise OSError("no route")

    e = ElevenLabsASR(api_key="k", connect=connect)
    with pytest.raises(AsrError):
        e.feed(AudioChunk(np.zeros(HOP, np.float32), PIPELINE_SR, 0, 0.0))


# --- latency (mock always; real engines when available) --------------------


def test_mock_phrase_latency_is_tiny():
    m = MockASR(words_per_second=4.0)
    x = _voiced(2.0)
    t = time.perf_counter()
    _feed_all(m, x)
    per_chunk_ms = 1000 * (time.perf_counter() - t) / (len(x) / HOP)
    assert per_chunk_ms < 1.0


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("ELEVENLABS_API_KEY"), reason="needs ELEVENLABS_API_KEY")
def test_elevenlabs_live_roundtrip():
    from hearing.synth.tts import ElevenLabsTTS

    speech = ElevenLabsTTS().synthesize("see you at noon by the fountain", "neutral")
    e = ElevenLabsASR()
    try:
        events = _feed_all(e, np.concatenate([speech, np.zeros(PIPELINE_SR, np.float32)]))
    finally:
        e.close()
    text = " ".join(ev.text for ev in _finals(events)).lower()
    assert "noon" in text


@pytest.mark.slow
@pytest.mark.skipif(not asr.local_available(), reason="faster-whisper or model missing")
@pytest.mark.skipif(not tts.local_available(), reason="Kokoro (used to make the speech) missing")
def test_local_whisper_transcribes_on_pause():
    from hearing.synth.tts import get_tts

    tts = get_tts("local")
    speech = tts.synthesize("see you at noon by the fountain", "neutral")
    w = asr.WhisperASR()
    x = np.concatenate([speech, np.zeros(PIPELINE_SR, np.float32)])
    finals = _finals(_feed_all(w, x))
    assert "noon" in " ".join(f.text for f in finals).lower()


def test_whisper_cuts_continuous_speech_at_max_phrase_and_at_energy_dips(monkeypatch):
    """No pause between overlapping talkers: phrases still end every ~3 s (T-074)."""
    calls = []
    monkeypatch.setattr(asr.WhisperASR, "_transcribe",
                        lambda self: calls.append(len(self._frames)) or setattr(self, "_frames", [])
                        or setattr(self, "_voiced", False) or setattr(self, "_rms", []) or [])
    w = asr.WhisperASR(model="base.en")
    t = np.arange(PIPELINE_SR * 12) / PIPELINE_SR
    loud = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)  # continuous: no pause, no dip
    _feed_all(w, loud)
    assert calls[:4] == [300] * 4  # hard cuts at 3 s (the last call is the final flush)

    calls.clear()
    w.reset()
    dipped = loud[: PIPELINE_SR * 4].copy()
    dipped[int(2.2 * PIPELINE_SR) : int(2.2 * PIPELINE_SR) + 400] = 0.001  # a 25 ms word gap
    _feed_all(w, dipped)
    assert calls and 215 <= calls[0] <= 230  # cut in the gap, before the 3 s limit


# ------------------------------------------------------------------ T-136: caption hallucination guards

def test_caption_vocab_refuses_sentences(monkeypatch):
    from hearing.synth import asr

    assert asr.caption_vocab("Aryan, Joseph, HackRU, see you at noon by the fountain") == ["Aryan", "Joseph", "HackRU"]
    monkeypatch.setenv(asr.VOCAB_ENV, "Aryan")
    assert asr.caption_vocab() == ["Aryan"]


def test_vocab_only_and_stock_phrases():
    from hearing.synth.asr import vocab_only

    assert vocab_only("Aryan.", ["Aryan", "Joseph"]) and vocab_only("Joseph, Aryan!", ["Aryan", "Joseph"])
    assert vocab_only("Thank you.", []) and vocab_only("Thanks for watching!", ["Aryan"])
    assert not vocab_only("Aryan, see you at noon", ["Aryan"]) and not vocab_only("hello there", [])


def test_collapse_repeats():
    from hearing.synth.asr import collapse_repeats

    assert collapse_repeats("go home. go home. go home.") == "go home."
    assert collapse_repeats("I said no no no no") == "I said no"
    assert collapse_repeats("no no, I am fine") == "no no, I am fine"  # two is speech, not a loop
    # T-139: the team preset's loop was one hyphen-chained token (100+ caption words from one phrase)
    assert collapse_repeats("Set Green with J-F-S-C-A-C-A-C-A-C-A-C-A-C-A-") == "Set Green with J-F-S-C-A-"
    assert collapse_repeats("E-6, Plains") == "E-6, Plains"
    # live run 2 (attempt 2): Whisper's masked-word loop has no letters but is still a loop
    assert collapse_repeats("Been blue with the ******* **** **** **** **** ****") == "Been blue with the ******* ****"


def test_deduper_drops_a_repeated_final_within_3_s():
    from hearing.synth.asr import CaptionDeduper

    d = CaptionDeduper()
    assert d.keep("See you at noon.", 1.0)
    assert not d.keep("see you at noon", 2.5) and d.dropped == 1
    assert d.keep("See you at noon.", 9.0) and d.keep("By the fountain.", 9.5)


def test_whisper_drops_vocab_only_finals_near_the_floor(monkeypatch):
    from types import SimpleNamespace

    from hearing import models
    from hearing.synth import asr

    said = {"text": " Aryan."}
    monkeypatch.setattr(models, "load", lambda key: SimpleNamespace(
        transcribe=lambda x, **kw: ([SimpleNamespace(text=said["text"], no_speech_prob=0.1, avg_logprob=-0.3)], None)))
    eng = asr.WhisperASR.__new__(asr.WhisperASR)
    eng._key, eng.vocab, eng.dropped = "fake", ["Aryan"], 0
    eng._floor = asr.deque([0.02] * 100, maxlen=asr.FLOOR_FRAMES)  # babble floor ~ -34 dBFS

    def run(amp):
        eng._frames, eng._voiced, eng._rms, eng._t0, eng._t1 = [np.full(PIPELINE_SR, amp, np.float32)], True, [], 0.0, 1.0
        return [t.text for t in eng._transcribe()]

    assert run(0.03) == [] and eng.dropped == 1  # the prompted name out of babble at the floor
    assert run(0.2) == ["Aryan."]  # a loud, real "Aryan!"
    said["text"] = " Aryan, over here."
    assert run(0.03) == ["Aryan, over here."]  # real words besides the vocabulary


def test_strip_prompt_echo_drops_the_prompt_read_back_but_keeps_a_name():
    from hearing.synth.asr import strip_prompt_echo

    vocab = ["Selective Hearing", "HackRU", "Rutgers", "Focus HQ", "Aryan", "Joseph"]
    echo = "Selective Hearing, HackRU, Rutgers, Focus HQ, Aryan, Joseph Selective Hearing,"  # T-139 run 1
    assert strip_prompt_echo(f"Set blue {echo} now", vocab) == "Set blue now"
    assert strip_prompt_echo("Hi Aryan, this is Selective Hearing.", vocab) == "Hi Aryan, this is Selective Hearing."
    assert strip_prompt_echo("Set blue now", []) == "Set blue now"
