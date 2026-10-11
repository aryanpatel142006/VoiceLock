"""SynthStage: ASR -> phrase queue -> TTS with captions, voice switching, backpressure, latency."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from conftest import scaled

from hearing import registry
from hearing.audio.io import HOP, PIPELINE_SR
from hearing.synth import pipeline as synth_pipeline
from hearing.synth import tts as tts_mod
from hearing.synth.asr import MockASR
from hearing.synth.pipeline import SynthStage, bench_signal, run_bench
from hearing.synth.tts import MockTTS, resolve_voice
from hearing.types import AudioChunk, Stage, StageContext

TEN_WORDS = "one two three four five six seven eight nine ten"


def voiced(seconds: float, hz: float = 220.0, amp: float = 0.2) -> np.ndarray:
    t = np.arange(round(seconds * PIPELINE_SR)) / PIPELINE_SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * PIPELINE_SR), np.float32)


def feed(
    stage: SynthStage,
    x: np.ndarray,
    ctx: StageContext | None = None,
    settle: bool = True,
    settle_s: float = 2.0,
):
    """Feed ``x`` in 10 ms hops. Returns (outputs per chunk, captions, chunk index of the first final).

    After a chunk that produced a final, wait for the worker to finish the phrase so the
    output does not depend on thread timing.
    """
    ctx = ctx or StageContext.default()
    outs, caps, first_final = [], [], None
    for i, s in enumerate(range(0, len(x), HOP)):
        chunk = AudioChunk(x[s : s + HOP], PIPELINE_SR, i, s / PIPELINE_SR)
        out = stage.process(chunk, ctx)
        assert len(out.samples) == len(chunk.samples)
        assert out.seq == chunk.seq and out.capture_ts == chunk.capture_ts
        outs.append(np.asarray(out.samples, np.float32))
        new = stage.drain_captions()
        caps += new
        if any(c["final"] for c in new):
            if first_final is None:
                first_final = i
            if settle:
                assert stage.wait_idle(settle_s), "TTS worker did not finish the phrase"
    return outs, caps, first_final


def peak_hz(x: np.ndarray) -> float:
    spec = np.abs(np.fft.rfft(x))
    return float(np.fft.rfftfreq(len(x), 1 / PIPELINE_SR)[spec.argmax()])


@pytest.fixture
def stage():
    s = SynthStage(asr="mock", tts="mock")
    yield s
    s.close()


# --- (1) registry ---------------------------------------------------------------


def test_registered_and_buildable_from_spec():
    p = registry.build_pipeline("synth(asr=mock,tts=mock)")
    try:
        assert isinstance(p, Stage)
        s = p.stages[0]
        assert isinstance(s, Stage) and isinstance(s, SynthStage)
        assert s.name == "synth" and s.sample_rate == 16000
        assert s.latency_ms == pytest.approx(400.0)  # MockASR pause_s
        assert isinstance(s.asr, MockASR) and isinstance(s.tts, MockTTS)
    finally:
        p.stages[0].close()


def test_factory_kwargs_and_engine_instances():
    s = SynthStage(asr=MockASR(TEN_WORDS), tts=MockTTS(), speak=False, voice="male_adult", max_queue=5)
    try:
        assert s.current_voice == "male_adult" and s.max_queue == 5 and s.speak is False
        assert s.asr.words == TEN_WORDS.split()
    finally:
        s.close()
    with pytest.raises(ValueError):
        SynthStage(asr="mock", tts="mock", max_queue=0)


# --- (2) zero room audio, output only after the first final ----------------------


def test_output_is_tts_only_after_first_final(stage):
    x = np.concatenate([voiced(2.0), silence(3.0)])
    outs, caps, first_final = feed(stage, x)
    assert first_final is not None
    before = np.concatenate(outs[: first_final + 1])
    after = np.concatenate(outs[first_final + 1 :])
    assert float(np.sum(before**2)) == 0.0
    assert float(np.sum(after**2)) > 0.0
    # The spoken phrase is exactly the mock tone for the final text.
    final_text = next(c["text"] for c in caps if c["final"])
    expected = MockTTS().synthesize(final_text, None)
    out = np.concatenate(outs)
    start = stage.phrases()[0].out_start
    assert start == (first_final + 1) * HOP  # audio starts in the chunk after the final
    np.testing.assert_allclose(out[start : start + len(expected)], expected, atol=1e-6)
    assert np.all(out[start + len(expected) :] == 0)


# --- (3) text mode: zeros + captions ----------------------------------------------


def test_speak_false_gives_zeros_and_captions():
    s = SynthStage(asr="mock", tts="mock", speak=False)
    try:
        x = np.concatenate([voiced(2.0), silence(1.0)])
        outs, caps, first_final = feed(s, x)
        assert first_final is not None
        assert not np.any(np.concatenate(outs))
        finals = [c for c in caps if c["final"]]
        assert len(finals) == 1
        for c in caps:
            assert {"text", "final", "t0", "t1"} <= set(c) and c["type"] == "caption"
        i_final = caps.index(finals[0])
        partials = [c for c in caps[:i_final] if not c["final"]]
        assert partials, "expected at least one partial caption before the final"
        assert finals[0]["t1"] >= partials[-1]["t1"] and finals[0]["text"].startswith(partials[0]["text"])
        assert s.latency_stats()["n"] == 0
        assert s.drain_captions() == []
    finally:
        s.close()


def test_partials_are_throttled_to_250ms():
    asr = MockASR(" ".join(["w"] * 60), words_per_second=20, phrase_words=100)  # one 2 s phrase
    s = SynthStage(asr=asr, tts="mock", speak=False)
    try:
        _, caps, _ = feed(s, np.concatenate([voiced(2.0), silence(1.0)]))
        assert sum(c["final"] for c in caps) == 1
        partials = [c for c in caps if not c["final"]]
        assert 1 <= len(partials) <= 9  # 2 s / 250 ms (+1): not one per 50 ms word
        gaps = np.diff([c["t1"] for c in partials])
        assert np.all(gaps >= 0.25 - 1e-6)
    finally:
        s.close()


# --- (4) voice switches at the phrase boundary --------------------------------------


def test_set_voice_applies_to_next_phrase():
    s = SynthStage(asr=MockASR(TEN_WORDS), tts="mock")
    try:
        assert s.current_voice is None
        x1 = np.concatenate([voiced(2.0), silence(1.0)])
        x2 = np.concatenate([voiced(2.0), silence(3.0)])
        outs1, _, ff1 = feed(s, x1)
        assert ff1 is not None and s.latency_stats()["n"] == 1
        s.set_voice("female_young")
        assert s.current_voice == "female_young"
        ctx = StageContext.default()
        outs2 = []
        for i, k in enumerate(range(0, len(x2), HOP)):
            seq = len(outs1) + i
            out = s.process(AudioChunk(x2[k : k + HOP], PIPELINE_SR, seq, seq * HOP / PIPELINE_SR), ctx)
            outs2.append(np.asarray(out.samples, np.float32))
            if any(c["final"] for c in s.drain_captions()):
                assert s.wait_idle(scaled(2.0))
        out = np.concatenate(outs1 + outs2)
        recs = s.phrases()
        assert len(recs) == 2 and recs[0].voice is None and recs[1].voice == "female_young"
        seg1 = out[recs[0].out_start : recs[1].out_start]
        seg2 = out[recs[1].out_start :]
        assert peak_hz(seg1) == pytest.approx(float(resolve_voice(None, "mock")), abs=5)
        assert peak_hz(seg2) == pytest.approx(float(resolve_voice("female_young", "mock")), abs=5)
        assert peak_hz(seg1) != peak_hz(seg2)
    finally:
        s.close()


# --- (5) latency stats ---------------------------------------------------------------


def test_latency_stats_count_finals_and_pause(stage):
    x = np.concatenate([voiced(2.0), silence(3.0)])
    _, caps, _ = feed(stage, x)
    n_finals = sum(c["final"] for c in caps)
    st = stage.latency_stats()
    assert st["n"] == n_finals == 1
    assert isinstance(st["p50"], float) and np.isfinite(st["p50"]) and np.isfinite(st["p95"])
    # Stream clock: the final fires 400 ms after the last word's frame and the audio starts in the
    # next chunk; the mock's t1 already covers that frame, so latency == pause == 400 ms.
    assert st["p50"] == pytest.approx(400.0, abs=1e-6)
    assert st["pause_p50"] == pytest.approx(400.0, abs=1e-6)
    assert 0.0 <= st["tts_p50"] < 2000.0
    rec = stage.phrases()[0]
    assert rec.latency_ms == st["p50"] and rec.out_start >= 0 and rec.first_audio_ts > rec.t1


def test_latency_stats_empty():
    s = SynthStage(asr="mock", tts="mock")
    try:
        st = s.latency_stats()
        assert st["n"] == 0 and np.isnan(st["p50"]) and st["dropped"] == 0
    finally:
        s.close()


# --- (6) reset reproduces the output -------------------------------------------------


def test_reset_reproduces_output(stage):
    x = np.concatenate([voiced(2.0), silence(2.5)])
    outs1, caps1, _ = feed(stage, x)
    stage.reset()
    assert stage.drain_captions() == [] and stage.latency_stats()["n"] == 0
    outs2, caps2, _ = feed(stage, x)
    np.testing.assert_allclose(np.concatenate(outs1), np.concatenate(outs2), atol=1e-6)
    assert caps1 == caps2
    assert np.any(np.concatenate(outs1))


def test_reset_drops_in_flight_phrase():
    gate = threading.Event()

    class SlowTTS(MockTTS):
        def stream(self, text, voice):
            gate.wait(5.0)
            yield from super().stream(text, voice)

    s = SynthStage(asr="mock", tts=SlowTTS())
    try:
        feed(s, np.concatenate([voiced(2.0), silence(0.5)]), settle=False)
        assert s.queued + int(s._busy) >= 1
        s.reset()
        gate.set()
        assert s.wait_idle(scaled(2.0))
        outs, _, _ = feed(s, silence(1.0))
        assert not np.any(np.concatenate(outs))
        assert s.latency_stats()["n"] == 0
    finally:
        s.close()


# --- (7) close joins the worker -----------------------------------------------------


def test_close_joins_worker_within_1s():
    s = SynthStage(asr="mock", tts="mock")
    thread = s._thread
    assert thread is not None and thread.daemon and thread.is_alive()
    t = time.perf_counter()
    s.close()
    assert time.perf_counter() - t < 1.0
    assert not thread.is_alive()
    s.close()  # idempotent


# --- backpressure ---------------------------------------------------------------------


def test_queue_drops_oldest_and_counts_in_telemetry():
    gate = threading.Event()

    class BlockedTTS(MockTTS):
        def stream(self, text, voice):
            gate.wait(5.0)
            yield from super().stream(text, voice)

    s = SynthStage(asr=MockASR(" ".join(f"w{i}" for i in range(40)), phrase_words=2), tts=BlockedTTS(), max_queue=2)
    try:
        ctx = StageContext.default()
        # 2 words per 0.8 s of voiced audio -> one final per 0.8 s; 4 s gives 5 phrases.
        _, caps, _ = feed(s, np.concatenate([voiced(4.0), silence(0.5)]), ctx, settle=False)
        finals = [c["text"] for c in caps if c["final"]]
        assert len(finals) >= 4
        assert s.dropped >= 1
        assert ctx.telemetry.extras["synth_dropped"] == s.dropped
        assert isinstance(ctx.telemetry.extras["synth_dropped"], int)
        assert all(not isinstance(v, str) for v in ctx.telemetry.extras.values())
        assert s.queued <= 2
    finally:
        gate.set()
        s.close()


def test_tts_failure_is_counted_not_raised():
    class BrokenTTS(MockTTS):
        def stream(self, text, voice):
            raise RuntimeError("boom")
            yield  # pragma: no cover

    s = SynthStage(asr="mock", tts=BrokenTTS())
    try:
        outs, _, ff = feed(s, np.concatenate([voiced(2.0), silence(1.0)]))
        assert ff is not None and not np.any(np.concatenate(outs))
        assert s.latency_stats()["tts_errors"] == 1
    finally:
        s.close()


# --- bench -------------------------------------------------------------------------------


def test_bench_signal_and_mock_bench_run():
    x = bench_signal(["a", "b"], speech_s=0.5, gap_s=0.2)
    assert len(x) == round((0.2 + 2 * 0.7) * PIPELINE_SR)
    st = run_bench("mock", "mock", phrases=2, speed=0.0)
    assert st["phrases"] == 2 and st["engine"] == "mock" and st["n"] + st["dropped"] <= 2
    assert np.isfinite(st["est_p50"]) or st["n"] == 0


def test_cli_prints_table(capsys):
    synth_pipeline.main(["--bench", "--tts", "mock", "--phrases", "2", "--speed", "0"])
    out = capsys.readouterr().out
    assert "engine" in out and "mock" in out and "dropped" in out


# --- Kokoro (weights required) ------------------------------------------------------------


@pytest.mark.slow
def test_kokoro_phrase_is_spoken():
    if not tts_mod.local_available():
        pytest.skip("Kokoro weights not installed")
    s = SynthStage(asr="mock", tts="local")
    try:
        # The worker loads Kokoro (cold: several seconds, more under a parallel verify) before
        # the first phrase, so give the settle wait room beyond the mock's 2 s.
        outs, _, ff = feed(s, np.concatenate([voiced(2.0), silence(4.0)]), settle_s=30.0)
        assert ff is not None and np.any(np.concatenate(outs))
        st = s.latency_stats()
        assert st["n"] == 1 and np.isfinite(st["tts_p50"])
    finally:
        s.close()


# --- T-094: caption cleanup ---------------------------------------------------------


@pytest.mark.parametrize("raw,clean", [
    (". . . .", ""), ("...", ""), (" - - ", ""), ("", ""),
    ("hello  there .", "Hello there."), (". . . I think so", "I think so"),
    ("well , okay ,, fine", "Well, okay, fine"),
    ("See you at noon.", "See you at noon."),
    ("U.S.A is big", "U.S.A is big"), ("e.g. this", "E.g. this"), ("pi is 3.5", "Pi is 3.5"),
    ("go to example.com", "Go to example.com"), ("yes,no", "Yes, no"),
])
def test_clean_caption(raw, clean):
    from hearing.synth.asr import clean_caption

    assert clean_caption(raw) == clean


def test_junk_phrase_makes_no_caption_and_no_speech():
    from hearing.synth.asr import Transcript

    s = SynthStage(asr=MockASR(TEN_WORDS), tts=MockTTS(), speak=True)
    try:
        s._on_transcripts([Transcript(". . . .", True, 0.0, 1.0)], 1.0)
        assert s.drain_captions() == [] and s.queued == 0
        s._on_transcripts([Transcript("hello there .", True, 0.0, 1.0)], 1.0)
        assert [c["text"] for c in s.drain_captions()] == ["Hello there."]
    finally:
        s.close()


class RecordingChunkedTTS:
    """Chunked like Kokoro (3 pieces per phrase); records the voice of every call and what it emitted."""

    name = "recording"
    sample_rate = PIPELINE_SR

    def __init__(self):
        self.calls: list[str | None] = []
        self.emitted: list[int] = []

    def stream(self, text, voice):
        self.calls.append(voice)
        n = 0
        for _ in range(3):
            piece = voiced(0.2, hz=300.0 if voice is None else 600.0)
            n += len(piece)
            yield piece
        self.emitted.append(n)

    def synthesize(self, text, voice):
        return np.concatenate(list(self.stream(text, voice)))

    def close(self):
        pass


def test_voice_switch_between_phrases_keeps_every_sample():
    rec_tts = RecordingChunkedTTS()
    s = SynthStage(asr=MockASR(TEN_WORDS), tts=rec_tts)
    try:
        x = np.concatenate([voiced(2.0), silence(1.0)])
        outs1, _, _ = feed(s, x)
        s.set_voice("male_adult")
        outs2, _, _ = feed(s, x)
        out = np.concatenate(outs1 + outs2)
        assert rec_tts.calls[-2:] == [None, "male_adult"]
        recs = s.phrases()
        seg1 = out[recs[0].out_start : recs[1].out_start]
        seg2 = out[recs[1].out_start :]
        assert peak_hz(seg1) == pytest.approx(300.0, abs=5) and peak_hz(seg2) == pytest.approx(600.0, abs=5)
        # nothing dropped or cut: all samples the TTS emitted for both phrases reach the output
        voiced_samples = int(np.count_nonzero(np.abs(out) > 1e-4))
        assert voiced_samples >= 0.95 * sum(rec_tts.emitted)
    finally:
        s.close()
