"""Synthetic / Text mode stage: streaming ASR -> phrase queue -> TTS, with captions.

``SynthStage`` (registered as ``synth``) takes the cleaned target stream in 10 ms chunks,
feeds it to a :class:`~hearing.synth.asr.StreamingASR` on the caller thread and turns
every final transcript into a phrase. One daemon worker thread pops phrases and streams
:class:`~hearing.synth.tts.TtsEngine` audio into a ring buffer; ``process`` returns
exactly the input length drained from that buffer, zeros when it is empty. Room audio
never passes through. With ``speak=False`` (Text mode) no worker runs and the output is
all zeros; captions are still produced.

Phrase latency is measured per phrase as ``first_audio_ts - t1`` in the stream clock
(``chunk.capture_ts``): the time from the speaker's last word to the first synthesized
sample leaving the stage. It includes the ASR endpoint pause (reported as the stage's
``latency_ms``) and TTS compute.

``python -m hearing.synth.pipeline --bench`` prints a phrase-latency table.
"""

from __future__ import annotations

import argparse
import collections
import logging
import math
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from hearing import registry
from hearing.audio.io import PIPELINE_SR
from hearing.audio.stream import RingBuffer
from hearing.synth.asr import (
    CaptionDeduper,
    StreamingASR,
    Transcript,
    clean_caption,
    get_asr,
)
from hearing.synth.tts import TtsEngine, get_tts
from hearing.types import AudioChunk, StageContext

log = logging.getLogger(__name__)

PARTIAL_INTERVAL_S = 0.25  # at most one partial caption per phrase per 250 ms of stream time


@dataclass
class PhraseRecord:
    """One spoken phrase: when it ended, when its audio started, and how long that took."""

    text: str
    voice: str | None
    t0: float
    t1: float  # stream clock: end of the last word (ASR final t1)
    final_ts: float  # stream clock: end of the chunk that produced the final
    final_wall: float  # perf_counter when the final was queued
    offset: int = 0  # index of its first sample in the pushed-sample count
    first_audio_ts: float = math.nan  # stream clock of the first drained sample
    out_start: int = -1  # index of the first sample in the stage's output stream
    latency_ms: float = math.nan  # first_audio_ts - t1 (stream clock)
    tts_ms: float = math.nan  # wall clock from the final to the first drained sample

    @property
    def pause_ms(self) -> float:
        return 1000.0 * (self.final_ts - self.t1)


def _engine_pause_s(asr: StreamingASR) -> float:
    """The ASR's endpoint pause in seconds (0 when the engine does not expose one)."""
    engine = getattr(asr, "active", asr)  # FallbackASR: the engine in use
    pause = getattr(engine, "pause_s", None)
    if pause is None and getattr(engine, "pause_frames", None) is not None:
        pause = engine.pause_frames / 100.0
    if pause is None:
        pause = getattr(engine, "vad_silence_s", None)
    return float(pause or 0.0)


def _percentiles(values: list[float]) -> tuple[float, float]:
    try:
        from eval.metrics import LatencyStats

        s = LatencyStats.from_ms(values)
        return s.p50, s.p95
    except ImportError:  # eval extra not installed
        if not values:
            return math.nan, math.nan
        p50, p95 = np.percentile(np.asarray(values, np.float64), [50, 95])
        return float(p50), float(p95)


@registry.register("synth")
class SynthStage:
    """ASR -> phrase queue -> TTS. See the module docstring.

    Factory kwargs (all usable from a pipeline spec): ``asr`` / ``tts`` engine names for
    :func:`get_asr` / :func:`get_tts` (Python callers may pass engine instances),
    ``speak`` (``False`` for Text mode), ``voice`` (initial bucket), ``max_queue``
    phrases, ``buffer_s`` of output buffering.
    """

    name = "synth"
    sample_rate = PIPELINE_SR

    def __init__(
        self,
        asr: str | StreamingASR | None = None,
        tts: str | TtsEngine | None = None,
        speak: bool = True,
        voice: str | None = None,
        max_queue: int = 3,
        buffer_s: float = 30.0,
    ):
        if max_queue < 1:
            raise ValueError(f"max_queue must be >= 1, got {max_queue}")
        self.asr: StreamingASR = asr if not isinstance(asr, str | type(None)) else get_asr(asr)
        self.tts: TtsEngine = tts if not isinstance(tts, str | type(None)) else get_tts(tts)
        self.speak = bool(speak)
        self.max_queue = int(max_queue)
        self.latency_ms = 1000.0 * _engine_pause_s(self.asr)

        self._cond = threading.Condition()
        self._voice = voice
        self._queue: collections.deque[PhraseRecord] = collections.deque()
        self._pending: collections.deque[PhraseRecord] = collections.deque()  # pushed, not yet drained
        self._records: list[PhraseRecord] = []
        self._ring = RingBuffer(max(PIPELINE_SR // 100, round(buffer_s * PIPELINE_SR)))
        self._pushed = 0  # TTS samples pushed into the ring (lifetime, reset by reset())
        self._drained = 0  # TTS samples pulled out of the ring
        self._out_total = 0  # output samples returned (TTS + zeros)
        self._dropped = 0
        self._tts_errors = 0
        self._busy = False  # worker is between popping a phrase and finishing its audio
        self._gen = 0  # bumped by reset(): the worker abandons a phrase from an older generation
        self._stop = False
        self._captions: list[dict] = []
        self._partial_last: float | None = None
        self._dedupe = CaptionDeduper()  # T-136: the same final again within 3 s is dropped

        self._thread: threading.Thread | None = None
        if self.speak:
            self._thread = threading.Thread(target=self._worker, name="synth-tts", daemon=True)
            self._thread.start()

    # --- voice ---------------------------------------------------------------

    @property
    def current_voice(self) -> str | None:
        return self._voice

    def set_voice(self, bucket: str | None) -> None:
        """Voice for phrases that start after this call; the one being spoken keeps its voice."""
        with self._cond:
            self._voice = bucket

    # --- counters / stats ----------------------------------------------------

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def queued(self) -> int:
        with self._cond:
            return len(self._queue)

    def drain_captions(self) -> list[dict]:
        """Return and clear the caption dicts (partials throttled, every final) since the last call."""
        with self._cond:
            out, self._captions = self._captions, []
        return out

    def phrases(self) -> list[PhraseRecord]:
        """Phrases whose audio has started playing, in output order (copies)."""
        with self._cond:
            return [replace(r) for r in self._records]

    def latency_stats(self) -> dict:
        """``p50`` / ``p95`` / ``n`` of phrase latency (stream clock, ms) plus TTS wall-clock stats."""
        recs = self.phrases()
        lat = [r.latency_ms for r in recs]
        tts = [r.tts_ms for r in recs]
        pause = [r.pause_ms for r in recs]
        p50, p95 = _percentiles(lat)
        t50, t95 = _percentiles(tts)
        pause50, _ = _percentiles(pause)
        return {
            "n": len(recs),
            "p50": p50,
            "p95": p95,
            "tts_p50": t50,
            "tts_p95": t95,
            "pause_p50": pause50,
            "dropped": self._dropped,
            "tts_errors": self._tts_errors,
        }

    def wait_idle(self, timeout: float = 1.0) -> bool:
        """Block until every queued phrase has been synthesized into the buffer (or ``timeout``)."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._queue or self._busy:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._cond.wait(left)
        return True

    # --- Stage ---------------------------------------------------------------

    def process(self, chunk: AudioChunk, ctx: StageContext) -> AudioChunk:
        n = len(chunk.samples)
        sr = chunk.sample_rate or PIPELINE_SR
        chunk_end = float(chunk.capture_ts) + n / sr
        # Drain first, then feed: a phrase's audio can start no earlier than the chunk after
        # its final, so chunked output is deterministic once the worker has caught up.
        out = self._drain(n, float(chunk.capture_ts), sr)
        events = self.asr.feed(chunk)
        if events:
            self._on_transcripts(events, chunk_end)
        extras = ctx.telemetry.extras
        extras["synth_dropped"] = self._dropped
        extras["synth_queued"] = len(self._queue)
        return replace(chunk, samples=out)

    def _on_transcripts(self, events: list[Transcript], chunk_end: float) -> None:
        now = time.perf_counter()
        with self._cond:
            for t in events:
                text = clean_caption(t.text)
                if not text:
                    if t.final:
                        self._partial_last = None
                    continue  # ". . . ." and friends: no caption, no phrase
                t = replace(t, text=text)
                if t.final and not self._dedupe.keep(text, t.t1):  # T-136: Whisper re-cut the same words
                    self._partial_last = None
                    continue
                if t.final:
                    self._captions.append(t.to_caption())
                    self._partial_last = None
                    if self.speak and t.text.strip():
                        if len(self._queue) >= self.max_queue:
                            self._queue.popleft()
                            self._dropped += 1
                        self._queue.append(PhraseRecord(t.text, None, t.t0, t.t1, chunk_end, now))
                elif self._partial_last is None or t.t1 - self._partial_last >= PARTIAL_INTERVAL_S:
                    self._captions.append(t.to_caption())
                    self._partial_last = t.t1
            self._cond.notify_all()

    def _drain(self, n: int, capture_ts: float, sr: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        with self._cond:
            avail = min(n, self._ring.available)
            if avail:
                out[:avail] = self._ring.pull(avail)
                start = self._drained
                self._drained += avail
                now = time.perf_counter()
                while self._pending and self._pending[0].offset < start + avail:
                    rec = self._pending.popleft()
                    i = max(0, rec.offset - start)
                    rec.first_audio_ts = capture_ts + i / sr
                    rec.out_start = self._out_total + i
                    rec.latency_ms = 1000.0 * (rec.first_audio_ts - rec.t1)
                    rec.tts_ms = 1000.0 * (now - rec.final_wall)
                    self._records.append(rec)
                self._cond.notify_all()  # buffer space freed for the worker
            self._out_total += n
        return out

    def reset(self) -> None:
        """Restart: ASR state, queue, buffers, captions and stats; an in-flight phrase is dropped."""
        self.asr.reset()
        with self._cond:
            self._gen += 1
            self._queue.clear()
            self._pending.clear()
            self._records.clear()
            self._captions.clear()
            self._ring.clear()
            self._pushed = self._drained = self._out_total = 0
            self._dropped = self._tts_errors = 0
            self._partial_last = None
            self._dedupe = CaptionDeduper()
            self._cond.notify_all()

    def close(self) -> None:
        """Stop the worker (joins within 1 s) and close both engines."""
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self.asr.close()
        self.tts.close()

    # --- worker --------------------------------------------------------------

    def _warm_tts(self) -> None:
        """Run the local TTS once before the first phrase (Kokoro's first call costs ~0.8 s extra)."""
        if self.tts.name != "kokoro":
            return  # mocks need none; hosted engines must not be called without a phrase
        try:
            for _ in self.tts.stream("Ready.", None):
                pass
        except Exception as e:  # noqa: BLE001 - warm-up is best effort
            log.info("synth: TTS warm-up failed: %s", e)

    def _worker(self) -> None:
        self._warm_tts()
        while True:
            with self._cond:
                while not self._queue and not self._stop:
                    self._cond.wait(0.1)
                if self._stop:
                    return
                rec = self._queue.popleft()
                rec.voice = self._voice
                gen = self._gen
                self._busy = True
            try:
                self._speak(rec, gen)
            except Exception as e:  # noqa: BLE001 - one bad phrase must not kill the session
                log.warning("synth: TTS %s failed on a phrase: %s", self.tts.name, e)
                with self._cond:
                    self._tts_errors += 1
            finally:
                with self._cond:
                    self._busy = False
                    self._cond.notify_all()

    def _speak(self, rec: PhraseRecord, gen: int) -> None:
        first = True
        for piece in self.tts.stream(rec.text, rec.voice):
            x = np.asarray(piece, np.float32).ravel()
            while len(x):
                with self._cond:
                    if self._gen != gen or self._stop:
                        return  # reset()/close() while speaking: drop the rest of this phrase
                    free = self._ring.capacity - self._ring.available
                    if free == 0:
                        self._cond.wait(0.05)
                        continue
                    part, x = x[:free], x[free:]
                    if first:
                        rec.offset = self._pushed
                        self._pending.append(rec)
                        first = False
                    self._ring.push(part)
                    self._pushed += len(part)
                    self._cond.notify_all()


# --- bench -------------------------------------------------------------------


def bench_signal(texts: list[str], speech_s: float = 2.0, gap_s: float = 1.0) -> np.ndarray:
    """``len(texts)`` voiced segments (a soft 220 Hz tone, well above the energy VAD) with pauses."""
    n_on, n_off = round(speech_s * PIPELINE_SR), round(gap_s * PIPELINE_SR)
    t = np.arange(n_on) / PIPELINE_SR
    tone = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    parts = [np.zeros(n_off, np.float32)]
    for _ in texts:
        parts += [tone, np.zeros(n_off, np.float32)]
    return np.concatenate(parts)


def run_bench(asr: str = "mock", tts: str = "mock", phrases: int = 10, speed: float = 4.0,
              words: int = 5) -> dict:
    """Feed ``phrases`` tone/pause segments through ``synth`` at ``speed`` x realtime (0 = unpaced)."""
    from hearing import device
    from hearing.synth.asr import MockASR

    texts = [f"phrase {i + 1} " + " ".join(f"word{j}" for j in range(words - 2)) for i in range(phrases)]
    if asr == "mock":
        asr_engine: StreamingASR | str = MockASR(" ".join(texts), words_per_second=words / 2.0)
    else:
        asr_engine = asr
    stage = SynthStage(asr=asr_engine, tts=tts, voice="neutral")
    stage.tts.synthesize("warm up", "neutral")  # load weights / resolve DNS outside the timing
    x = bench_signal(texts)
    hop = PIPELINE_SR // 100
    ctx = StageContext.default()
    t_start = time.perf_counter()
    for i, seq in enumerate(range(0, len(x), hop)):
        if speed > 0:
            time.sleep(max(0.0, t_start + i * 0.01 / speed - time.perf_counter()))
        stage.process(AudioChunk(x[seq : seq + hop], PIPELINE_SR, i, seq / PIPELINE_SR), ctx)
    # Let the last phrase come out: feed silence until its audio has started (bounded).
    stage.wait_idle(10.0)
    for j in range(300):
        i += 1
        stage.process(AudioChunk(np.zeros(hop, np.float32), PIPELINE_SR, i, (len(x) + j * hop) / PIPELINE_SR), ctx)
        if stage.latency_stats()["n"] >= phrases - stage.dropped:
            break
    stats = stage.latency_stats()
    stats.update(engine=stage.tts.name, asr=stage.asr.name, device=device.pick(), phrases=phrases,
                 speed=speed, est_p50=stats["pause_p50"] + stats["tts_p50"],
                 est_p95=stats["pause_p50"] + stats["tts_p95"])
    stage.close()
    return stats


def _print_table(rows: list[dict]) -> None:
    cols = [("engine", 10), ("asr", 7), ("device", 6), ("phrases", 7), ("n", 3), ("speed", 5),
            ("p50", 8), ("p95", 8), ("tts_p50", 8), ("tts_p95", 8), ("est_p50", 8), ("est_p95", 8),
            ("dropped", 7)]
    print("  ".join(f"{c:>{w}}" for c, w in cols))
    for r in rows:
        cells = []
        for c, w in cols:
            v = r.get(c, "")
            cells.append(f"{v:>{w}.1f}" if isinstance(v, float) else f"{v!s:>{w}}")
        print("  ".join(cells))
    print("ms; p50/p95 = phrase latency in the stream clock (speaker's last word -> first TTS sample "
          "out, includes the ASR pause); tts = wall clock final -> first sample; est = pause + tts "
          "(what a realtime session sees, independent of bench speed)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bench", action="store_true", help="run the phrase-latency bench")
    p.add_argument("--tts", default="mock", choices=["mock", "local", "elevenlabs", "elevenlabs-mock", "auto", "all"])
    p.add_argument("--asr", default="mock", choices=["mock", "local", "elevenlabs", "auto"])
    p.add_argument("--phrases", type=int, default=10)
    p.add_argument("--speed", type=float, default=4.0, help="x realtime; 0 = as fast as possible")
    args = p.parse_args(argv)
    if not args.bench:
        p.print_help()
        return
    from hearing.synth import tts as tts_mod

    engines = [args.tts]
    if args.tts == "all":
        engines = ["mock"] + (["local"] if tts_mod.local_available() else [])
    rows = []
    for name in engines:
        try:
            rows.append(run_bench(args.asr, name, args.phrases, args.speed))
        except Exception as e:  # noqa: BLE001 - report the failure in the table instead of dying
            print(f"{name}: failed: {e}")
    _print_table(rows)


if __name__ == "__main__":
    main()
