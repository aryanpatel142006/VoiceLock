"""Streaming ASR adapters: ElevenLabs Scribe v2 Realtime, local faster-whisper, offline mock.

Every engine takes pipeline ``AudioChunk``s (16 kHz mono float32, any chunk size) and
returns ``Transcript`` events: growing partials and one final per phrase. Timestamps
follow the capture clock of the chunks so captions line up with the audio.

``get_asr()`` picks an engine from ``HEARING_ASR`` (``mock`` / ``local`` /
``elevenlabs`` / ``auto``); ``auto`` prefers ElevenLabs when ``ELEVENLABS_API_KEY``
is set, then local faster-whisper if installed and cached, then the mock. Networked
and local engines are wrapped in ``FallbackASR`` so a dead socket degrades instead of
killing the session.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import logging
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlencode

import numpy as np

from hearing import device, models
from hearing.audio.io import PIPELINE_SR, float_to_pcm16, resample
from hearing.types import AudioChunk

log = logging.getLogger(__name__)

DEFAULT_TRANSCRIPT = "see you at noon by the fountain"
VOICED_RMS = 0.01  # ~ -40 dBFS: 10 ms frames above this count as speech for the energy VAD
ELEVENLABS_STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
DEFAULT_WHISPER_MODEL = "base.en"  # override with HEARING_WHISPER_MODEL (eval uses small.en)


class AsrError(RuntimeError):
    """An ASR engine failed (connection, auth, quota, server error, missing key)."""


@dataclass
class Transcript:
    text: str
    final: bool
    t0: float  # capture-clock seconds of the phrase start
    t1: float  # capture-clock seconds of the last audio covered

    def to_caption(self) -> dict:
        """The ``caption`` wire event (docs/CONTRACTS.md#events)."""
        return {"type": "caption", "text": self.text, "final": self.final, "t0": self.t0, "t1": self.t1}


def clean_caption(text: str) -> str:
    """Whisper output tidied for captions and TTS: ``""`` when nothing speakable is left
    (``". . . ."``, ``"..."``, ``"- -"``), else collapsed whitespace and repeated punctuation,
    no space before punctuation, first letter capitalised."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not re.search(r"\w", text):
        return ""
    text = re.sub(r"^[\s.,;:!?\-–—…]+", "", text)  # leading junk
    text = re.sub(r"(?:\s*\.){2,}", ".", text)  # ". . ." / "..." -> "."
    text = re.sub(r"([,;:!?])(?:\s*\1)+", r"\1", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"(?<=[A-Za-z]),(?=[A-Za-z])", ", ", text)
    return text[:1].upper() + text[1:]


# T-136 caption hallucination guards. Whisper on babble or a quiet room invents the words it was
# primed with or its stock phrases; a final made only of those while the phrase sits within
# FLOOR_RATIO of the room's energy floor is dropped. The vocabulary is names/domain words only.
VOCAB_ENV = "HEARING_ASR_VOCAB"  # comma-separated names/domain words, given to Whisper as hotwords
VOCAB_MAX_WORDS = 3  # an entry longer than this is a sentence: refused (scripted lines get "heard")
FLOOR_RATIO = 2.0  # phrase mean RMS below this x the floor (+6 dB) is "near the floor"
FLOOR_FRAMES = 500  # the floor is the 10th percentile of the last 5 s of 10 ms frame RMS
STOCK_PHRASES = frozenset({"thank you", "thanks for watching", "thank you for watching", "thank you so much",
                           "you", "bye", "okay", "so", "um", "uh", "hmm", "oh"})
NO_SPEECH_DROP = 0.6  # a Whisper segment with no_speech_prob >= this and avg_logprob < LOGPROB_DROP is dropped
LOGPROB_DROP = -1.0
REPEAT_S = 3.0  # the same final again within this is a duplicate


def caption_vocab(raw: str | None = None) -> list[str]:
    """``$HEARING_ASR_VOCAB`` (or ``raw``) split on commas; sentence-like entries are refused."""
    raw = os.environ.get(VOCAB_ENV, "") if raw is None else raw
    words = [w.strip() for w in raw.split(",") if w.strip()]
    return [w for w in words if len(w.split()) <= VOCAB_MAX_WORDS]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def vocab_only(text: str, vocab: list[str]) -> bool:
    """True when every word of ``text`` is a vocabulary word, or the text is a stock Whisper phrase."""
    words = _words(text)
    if not words or " ".join(words) in STOCK_PHRASES:
        return True
    allowed = {w for v in vocab for w in _words(v)}
    return bool(allowed) and set(words) <= allowed


_HYPHEN_LOOP = re.compile(r"((?:[A-Za-z0-9']+-){1,6}?)\1{2,}")  # a hyphen-joined n-gram 3+ times in a row


def collapse_repeats(text: str, min_run: int = 3) -> str:
    """``"go home. go home. go home."`` -> ``"go home."``: an n-gram (n <= 6) repeated ``min_run``+ times
    in a row is kept once (Whisper's repetition loop). A loop chained inside one token by hyphens
    (``"J-F-C-A-C-A-C-A-..."``, T-139: 100+ caption words from one phrase) is collapsed the same way."""
    while (shorter := _HYPHEN_LOOP.sub(r"\1", text)) != text:
        text = shorter
    toks = text.split()
    # a token with no letters keeps its own text as key: Whisper's masked-word loop "**** **** ****" (T-139)
    key = [re.sub(r"[^a-z0-9']", "", t.lower()) or t for t in toks]
    out, i = [], 0
    while i < len(toks):
        for n in range(1, 7):
            run = 1
            while key[i + run * n : i + (run + 1) * n] == key[i : i + n] and i + (run + 1) * n <= len(toks):
                run += 1
            if run >= min_run and any(key[i : i + n]):
                out.extend(toks[i + (run - 1) * n : i + run * n])
                i += run * n
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


ECHO_RUN = 4  # this many vocabulary words in a row is Whisper reading its prompt back, not speech


def strip_prompt_echo(text: str, vocab: list[str], run: int = ECHO_RUN) -> str:
    """Drops every run of ``run``+ consecutive vocabulary words (T-139: on overlapped talkers Whisper
    printed the whole caption prompt twice, 20 caption words in one final). A name said alone stays."""
    allowed = {w for v in vocab for w in _words(v)}
    if not allowed:
        return text
    toks = text.split()
    hit = [bool(_words(t)) and set(_words(t)) <= allowed for t in toks]
    out, i = [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and hit[j]:
            j += 1
        if j - i >= run:
            i = j
            continue
        out.extend(toks[i : max(j, i + 1)])
        i = max(j, i + 1)
    return " ".join(out)


class CaptionDeduper:
    """Drops a final whose text repeats the previous final within ``REPEAT_S`` (engine-independent)."""

    def __init__(self, repeat_s: float = REPEAT_S):
        self.repeat_s = repeat_s
        self._last: tuple[str, float] | None = None
        self.dropped = 0

    def keep(self, text: str, t1: float) -> bool:
        key = " ".join(_words(text))
        if self._last is not None and key == self._last[0] and t1 - self._last[1] <= self.repeat_s:
            self._last = (key, t1)
            self.dropped += 1
            return False
        self._last = (key, t1)
        return True


@runtime_checkable
class StreamingASR(Protocol):
    name: str
    sample_rate: int

    def feed(self, chunk: AudioChunk) -> list[Transcript]: ...
    def flush(self) -> list[Transcript]: ...
    def reset(self) -> None: ...
    def close(self) -> None: ...


class _Framer:
    """Re-frames arbitrary chunks into 10 ms frames with capture timestamps.

    Framing internally (not per chunk) makes the energy VAD, and so the engines built on
    it, give the same events whatever chunk size the caller uses.
    """

    def __init__(self, sr: int = PIPELINE_SR):
        self.sr = sr
        self.n = sr // 100
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, np.float32)
        self._ts = 0.0

    def push(self, chunk: AudioChunk) -> list[tuple[np.ndarray, float, bool]]:
        x = np.asarray(chunk.samples, np.float32)
        if chunk.sample_rate != self.sr:
            x = resample(x, chunk.sample_rate, self.sr)
        if not len(self._buf):
            self._ts = float(chunk.capture_ts)
        self._buf = np.concatenate([self._buf, x])
        frames = []
        k = len(self._buf) // self.n
        for i in range(k):
            f = self._buf[i * self.n : (i + 1) * self.n]
            rms = float(np.sqrt(np.mean(f.astype(np.float64) ** 2)))
            frames.append((f, self._ts + i * self.n / self.sr, rms > VOICED_RMS))
        self._buf = self._buf[k * self.n :]
        self._ts += k * self.n / self.sr
        return frames


def _manifest_transcript(manifest: dict, which: str = "target") -> str:
    t = manifest.get("transcript", manifest.get("transcripts"))
    if isinstance(t, dict):
        t = t.get(which, t.get("target"))
    if not isinstance(t, str):
        raise KeyError(f"manifest has no transcript for {which!r}")
    return t


class MockASR:
    """Deterministic offline ASR: reveals the given transcript word by word while voiced.

    One word per ``1 / words_per_second`` seconds of voiced audio (energy VAD); a phrase
    ends on ``pause_s`` of silence or after ``phrase_words`` words. Silence yields nothing,
    and once the transcript is used up the mock stays quiet. ``reset()`` starts over.
    The transcript is the argument, else ``HEARING_MOCK_TRANSCRIPT``, else ``DEFAULT_TRANSCRIPT``.
    """

    name = "mock"
    sample_rate = PIPELINE_SR

    def __init__(
        self,
        transcript: str | None = None,
        words_per_second: float = 2.5,
        phrase_words: int = 8,
        pause_s: float = 0.4,
    ):
        self.words = (transcript or os.environ.get("HEARING_MOCK_TRANSCRIPT") or DEFAULT_TRANSCRIPT).split()
        self.frames_per_word = max(1, round(100 / words_per_second))
        self.phrase_words = phrase_words
        self.pause_frames = max(1, round(pause_s * 100))
        self._framer = _Framer()
        self.reset()

    @classmethod
    def from_fixture(cls, fixture_dir: str | os.PathLike, **kwargs) -> MockASR:
        """Use the target transcript of a fixture's ``manifest.json``."""
        manifest = json.loads((Path(fixture_dir) / "manifest.json").read_text())
        return cls(_manifest_transcript(manifest), **kwargs)

    def reset(self) -> None:
        self._framer.reset()
        self._next = 0  # index of the next word to reveal
        self._phrase: list[str] = []
        self._voiced = 0  # voiced frames since the last word
        self._silent = 0
        self._t0 = self._t1 = 0.0

    def _end_phrase(self) -> list[Transcript]:
        if not self._phrase:
            return []
        out = [Transcript(" ".join(self._phrase), True, self._t0, self._t1)]
        self._phrase = []
        return out

    def feed(self, chunk: AudioChunk) -> list[Transcript]:
        out: list[Transcript] = []
        for _, ts, voiced in self._framer.push(chunk):
            if not voiced:
                self._silent += 1
                if self._silent == self.pause_frames:
                    out += self._end_phrase()
                    self._voiced = 0
                continue
            self._silent = 0
            if self._next >= len(self.words):
                continue
            if not self._phrase and self._voiced == 0:
                self._t0 = ts
            self._voiced += 1
            if self._voiced < self.frames_per_word:
                continue
            self._voiced = 0
            self._phrase.append(self.words[self._next])
            self._next += 1
            self._t1 = ts + 0.01
            if len(self._phrase) >= self.phrase_words or self._next >= len(self.words):
                out += self._end_phrase()
            else:
                out.append(Transcript(" ".join(self._phrase), False, self._t0, self._t1))
        return out

    def flush(self) -> list[Transcript]:
        return self._end_phrase()

    def close(self) -> None:
        pass


# --- local: faster-whisper on VAD-chunked phrases ---------------------------


def whisper_model_name() -> str:
    return os.environ.get("HEARING_WHISPER_MODEL", DEFAULT_WHISPER_MODEL)


def local_available() -> bool:
    """faster-whisper installed and the configured model already in the HF cache."""
    if importlib.util.find_spec("faster_whisper") is None:
        return False
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return root.exists() and any(root.glob(f"models--*faster-whisper-{whisper_model_name()}"))


def _whisper_loader(model: str) -> str:
    """Register (once) and return the shared loader key; eval/transcribe.py uses the same key."""
    key = f"faster-whisper-{model}"

    def load():
        from faster_whisper import WhisperModel

        # CTranslate2 has no MPS backend: CUDA float16 or CPU int8.
        if device.pick() == "cuda":
            return WhisperModel(model, device="cuda", compute_type="float16")
        return WhisperModel(model, device="cpu", compute_type="int8")

    try:
        models.register_loader(key)(load)
    except ValueError:
        pass  # already registered
    return key


# Overlapping talkers leave no pause for the energy VAD, so a phrase is also cut inside speech:
# at the first energy dip once it is ``MIN_CUT_S`` long, or hard at ``max_phrase_s`` (T-074:
# one 8 s phrase made Synthetic speak 3.3 s late; 3 s phrases cut that to ~1.5 s).
DEFAULT_MAX_PHRASE_S = 3.0  # override with HEARING_ASR_MAX_PHRASE_S
MIN_CUT_S = 1.8  # no dip cut before the phrase is this long
DIP_RATIO = 0.4  # a dip is a 30 ms mean RMS below this fraction of the phrase's mean RMS
DIP_FRAMES = 3


def whisper_caption_kwargs() -> dict:
    """``HEARING_ASR_TEMPERATURE`` / ``HEARING_ASR_PROMPT`` (the team demo preset, T-130) as
    faster-whisper kwargs; unset: none (faster-whisper's defaults). The prompt is vocabulary only."""
    kw: dict = {}
    if os.environ.get("HEARING_ASR_TEMPERATURE"):
        kw["temperature"] = float(os.environ["HEARING_ASR_TEMPERATURE"])
    if os.environ.get("HEARING_ASR_PROMPT"):
        kw["initial_prompt"] = os.environ["HEARING_ASR_PROMPT"]
    return kw


class WhisperASR:
    """Local fallback: energy-VAD phrases transcribed whole by faster-whisper.

    Finals only (no partials); a phrase is cut on ``pause_s`` of silence, at an energy dip
    after ``MIN_CUT_S`` or at ``max_phrase_s`` (continuous speech, e.g. two overlapping
    talkers). Phrase latency = pause_s + whisper compute (see report numbers).
    """

    name = "whisper"
    sample_rate = PIPELINE_SR

    def __init__(self, model: str | None = None, pause_s: float = 0.4,
                 max_phrase_s: float | None = None, preroll_s: float = 0.2, vocab: list[str] | None = None):
        self.model_name = model or whisper_model_name()
        self.vocab = caption_vocab() if vocab is None else caption_vocab(",".join(vocab))
        self._floor: deque[float] = deque(maxlen=FLOOR_FRAMES)  # frame RMS for the room floor (not reset)
        self.dropped = 0  # finals the hallucination guards dropped
        self._key = _whisper_loader(self.model_name)
        if max_phrase_s is None:
            max_phrase_s = float(os.environ.get("HEARING_ASR_MAX_PHRASE_S") or DEFAULT_MAX_PHRASE_S)
        self.pause_frames = max(1, round(pause_s * 100))
        self.max_frames = round(max_phrase_s * 100)
        self.min_cut_frames = min(round(MIN_CUT_S * 100), self.max_frames)
        self.preroll = round(preroll_s * 100)
        self._framer = _Framer()
        self.reset()

    def reset(self) -> None:
        self._framer.reset()
        self._frames: list[np.ndarray] = []
        self._recent: list[tuple[np.ndarray, float]] = []  # pre-roll before speech onset
        self._voiced = False
        self._silent = 0
        self._t0 = self._t1 = 0.0
        self._rms: list[float] = []  # per-frame RMS of the current phrase

    def _dip(self) -> bool:
        """The last ``DIP_FRAMES`` frames are quiet against the phrase so far (a word gap)."""
        r = self._rms
        return len(r) >= self.min_cut_frames and float(np.mean(r[-DIP_FRAMES:])) < DIP_RATIO * float(np.mean(r))

    def _transcribe(self) -> list[Transcript]:
        frames, self._frames, self._voiced, self._rms = self._frames, [], False, []
        if not frames:
            return []
        x = np.concatenate(frames)
        model = models.load(self._key)
        kw = {"hotwords": ", ".join(self.vocab)} if self.vocab else {}
        segments, _ = model.transcribe(x, language="en", beam_size=1, vad_filter=False,
                                       condition_on_previous_text=False,
                                       # no fallback re-decodes: 1.1-1.6 s each on babble (T-120); the team
                                       # preset's HEARING_ASR_TEMPERATURE / _PROMPT override (T-130)
                                       **{"temperature": 0.0, **kw, **whisper_caption_kwargs()})
        kept = [s for s in segments if not (getattr(s, "no_speech_prob", 0.0) >= NO_SPEECH_DROP
                                            and getattr(s, "avg_logprob", 0.0) < LOGPROB_DROP)]
        text = strip_prompt_echo(collapse_repeats(" ".join(s.text.strip() for s in kept).strip()), self.vocab)
        if text and self.hallucinated(text, float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))):
            self.dropped += 1
            return []
        return [Transcript(text, True, self._t0, self._t1)] if text else []

    def floor_rms(self) -> float:
        return float(np.percentile(self._floor, 10)) if self._floor else 0.0

    def hallucinated(self, text: str, rms: float) -> bool:
        """A final made only of vocabulary words / a stock phrase while the phrase is near the floor."""
        return rms < FLOOR_RATIO * max(self.floor_rms(), 1e-6) and vocab_only(text, self.vocab)

    def feed(self, chunk: AudioChunk) -> list[Transcript]:
        out: list[Transcript] = []
        for f, ts, voiced in self._framer.push(chunk):
            self._floor.append(float(np.sqrt(np.mean(np.square(f, dtype=np.float64)))))
            if not self._voiced:
                self._recent = (self._recent + [(f, ts)])[-self.preroll - 1 :]
                if voiced:
                    self._voiced, self._silent = True, 0
                    self._t0 = self._recent[0][1]
                    self._frames = [g for g, _ in self._recent]
                    self._rms = [float(np.sqrt(np.mean(g.astype(np.float64) ** 2))) for g in self._frames]
                    self._recent = []
                    self._t1 = ts + 0.01
                continue
            self._frames.append(f)
            self._rms.append(float(np.sqrt(np.mean(f.astype(np.float64) ** 2))))
            if voiced:
                self._silent = 0
                self._t1 = ts + 0.01
            else:
                self._silent += 1
            if (self._silent >= self.pause_frames or len(self._frames) >= self.max_frames
                    or self._dip()):
                out += self._transcribe()
        return out

    def flush(self) -> list[Transcript]:
        return self._transcribe()

    def close(self) -> None:
        pass


# --- ElevenLabs Scribe v2 Realtime ------------------------------------------

Connect = Callable[[str, dict], object]

_ERROR_TYPES = {
    "error", "auth_error", "quota_exceeded", "commit_throttled", "unaccepted_terms", "rate_limited",
    "queue_overflow", "resource_exhausted", "session_time_limit_exceeded", "input_error",
    "invalid_request", "chunk_size_exceeded", "transcriber_error",
}


def _ws_connect(url: str, headers: dict):
    from websockets.sync.client import connect

    # certifi CAs: the python.org build has none (same fix as TTS)
    from hearing.synth.tts import tls_context

    tls = {"ssl": tls_context()} if url.startswith("wss://") else {}
    return connect(url, additional_headers=headers, open_timeout=5, max_size=None, **tls)


class ElevenLabsASR:
    """Scribe v2 Realtime over a WebSocket, server-side VAD commits phrases.

    Audio is sent as base64 PCM16 in ``send_ms`` batches; a background thread turns
    ``partial_transcript`` / ``committed_transcript`` messages into ``Transcript`` events
    that ``feed`` / ``flush`` drain. Any server error or socket failure raises
    ``AsrError`` from the next call (``FallbackASR`` catches it). Timestamps are
    approximated from the amount of audio sent (the server's own clock is not used).
    """

    name = "elevenlabs"
    sample_rate = PIPELINE_SR

    def __init__(
        self,
        api_key: str | None = None,
        connect: Connect | None = None,
        send_ms: int = 50,
        model_id: str = "scribe_v2_realtime",
        language_code: str | None = "en",
        vad_silence_s: float = 0.5,
        flush_timeout_s: float = 3.0,
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise AsrError("ELEVENLABS_API_KEY is not set")
        self._connect_fn = connect or _ws_connect
        self.send_samples = PIPELINE_SR * send_ms // 1000
        params = {
            "model_id": model_id,
            "audio_format": f"pcm_{PIPELINE_SR}",
            "commit_strategy": "vad",
            "vad_silence_threshold_secs": vad_silence_s,
        }
        if language_code:
            params["language_code"] = language_code
        self.url = f"{ELEVENLABS_STT_URL}?{urlencode(params)}"
        self.flush_timeout_s = flush_timeout_s
        self._sock = None
        self._thread: threading.Thread | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._events: queue.Queue[Transcript] = queue.Queue()
        self._commits = threading.Semaphore(0)
        self._error: str | None = None
        self._closing = False
        self._pending = np.zeros(0, np.float32)
        self._origin: float | None = None  # capture_ts of the first sample sent
        self._sent = 0  # samples sent
        self._phrase_t0 = 0.0

    def _clock(self) -> float:
        return (self._origin or 0.0) + (self._sent + len(self._pending)) / PIPELINE_SR

    def _ensure_open(self) -> None:
        if self._sock is not None:
            return
        self._closing = False
        try:
            self._sock = self._connect_fn(self.url, {"xi-api-key": self.api_key})
        except Exception as e:  # any connect failure degrades the engine
            raise AsrError(f"ElevenLabs STT connect failed: {e}") from e
        self._thread = threading.Thread(target=self._reader, name="elevenlabs-asr", daemon=True)
        self._thread.start()

    def _reader(self) -> None:
        sock = self._sock
        while not self._closing:
            try:
                raw = sock.recv(timeout=0.1)
            except TimeoutError:
                continue
            except Exception as e:  # noqa: BLE001 - closed socket ends the reader
                if not self._closing:
                    self._error = f"connection lost: {e}"
                    self._commits.release()
                return
            try:
                msg = json.loads(raw)
            except (TypeError, ValueError):
                continue
            kind = msg.get("message_type", "")
            text = (msg.get("text") or "").strip()
            if kind == "partial_transcript" and text:
                self._events.put(Transcript(text, False, self._phrase_t0, self._clock()))
            elif kind in ("committed_transcript", "committed_transcript_with_timestamps"):
                t1 = self._clock()
                if text:
                    self._events.put(Transcript(text, True, self._phrase_t0, t1))
                self._phrase_t0 = t1
                self._commits.release()
            elif kind in _ERROR_TYPES:
                self._error = f"{kind}: {msg.get('error') or msg.get('message') or msg}"
                self._commits.release()

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            raise AsrError(f"ElevenLabs STT {self._error}")

    def _send(self, x: np.ndarray, commit: bool = False) -> None:
        msg = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(float_to_pcm16(x)).decode("ascii"),
            "commit": commit,
            "sample_rate": PIPELINE_SR,
        }
        try:
            self._sock.send(json.dumps(msg))
        except Exception as e:
            self._error = f"send failed: {e}"
            raise AsrError(f"ElevenLabs STT {self._error}") from e
        self._sent += len(x)

    def _drain(self) -> list[Transcript]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def feed(self, chunk: AudioChunk) -> list[Transcript]:
        self._raise_if_failed()
        self._ensure_open()
        x = np.asarray(chunk.samples, np.float32)
        if chunk.sample_rate != PIPELINE_SR:
            x = resample(x, chunk.sample_rate, PIPELINE_SR)
        if self._origin is None:
            self._origin = self._phrase_t0 = float(chunk.capture_ts)
        self._pending = np.concatenate([self._pending, x])
        while len(self._pending) >= self.send_samples:
            batch, self._pending = self._pending[: self.send_samples], self._pending[self.send_samples :]
            self._send(batch)
        return self._drain()

    def flush(self) -> list[Transcript]:
        """Send the remaining audio with ``commit`` and wait for the committed text."""
        self._raise_if_failed()
        if self._sock is None:
            return self._drain()
        while self._commits.acquire(blocking=False):  # forget commits already delivered
            pass
        batch, self._pending = self._pending, np.zeros(0, np.float32)
        self._send(batch, commit=True)
        if not self._commits.acquire(timeout=self.flush_timeout_s):
            log.warning("ElevenLabs STT: no commit within %.1f s", self.flush_timeout_s)
        self._raise_if_failed()
        return self._drain()

    def close(self) -> None:
        self._closing = True
        if self._sock is not None:
            with contextlib.suppress(Exception):
                self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sock = self._thread = None

    def reset(self) -> None:
        self.close()
        self._reset_state()


# --- fallback and selection -------------------------------------------------


class FallbackASR:
    """Use ``primary`` until it raises, then ``fallback`` for the rest of the session.

    ``degraded`` / ``last_error`` feed the ``error`` wire event (``degraded_to``).
    """

    sample_rate = PIPELINE_SR

    def __init__(self, primary: StreamingASR, fallback: StreamingASR):
        self.primary, self.fallback = primary, fallback
        self.degraded = False
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return (self.fallback if self.degraded else self.primary).name

    @property
    def active(self) -> StreamingASR:
        return self.fallback if self.degraded else self.primary

    def _degrade(self, e: Exception) -> None:
        log.warning("ASR %s failed (%s); degrading to %s", self.primary.name, e, self.fallback.name)
        self.degraded, self.last_error = True, str(e)
        with contextlib.suppress(Exception):
            self.primary.close()

    def feed(self, chunk: AudioChunk) -> list[Transcript]:
        if not self.degraded:
            try:
                return self.primary.feed(chunk)
            except Exception as e:  # noqa: BLE001 - any engine failure must not kill the session
                self._degrade(e)
        return self.fallback.feed(chunk)

    def flush(self) -> list[Transcript]:
        if not self.degraded:
            try:
                return self.primary.flush()
            except Exception as e:  # noqa: BLE001
                self._degrade(e)
        return self.fallback.flush()

    def reset(self) -> None:
        self.active.reset()

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


def get_asr(name: str | None = None, *, transcript: str | None = None) -> StreamingASR:
    """``mock`` | ``local`` | ``elevenlabs`` | ``auto`` (default: ``HEARING_ASR`` or auto).

    ``transcript`` sets the mock's text (also when ``auto`` falls back to the mock);
    otherwise ``HEARING_MOCK_TRANSCRIPT`` applies, then ``DEFAULT_TRANSCRIPT``.
    """
    name = (name or os.environ.get("HEARING_ASR") or "auto").lower()
    if name == "mock":
        return MockASR(transcript)
    if name in ("local", "whisper"):
        return WhisperASR()
    if name == "elevenlabs":
        return ElevenLabsASR()
    if name != "auto":
        raise ValueError(f"Unknown ASR {name!r}; use auto, mock, local or elevenlabs")
    backup: StreamingASR = WhisperASR() if local_available() else MockASR(transcript)
    if os.environ.get("ELEVENLABS_API_KEY"):
        log.info("asr: elevenlabs (fallback %s)", backup.name)
        return FallbackASR(ElevenLabsASR(), backup)
    if isinstance(backup, WhisperASR):
        log.info("asr: whisper %s (fallback mock)", backup.model_name)
        return FallbackASR(backup, MockASR(transcript))
    log.info("asr: mock (no ELEVENLABS_API_KEY, faster-whisper %s not cached)", whisper_model_name())
    return backup


def endpoint_latency(engine: StreamingASR, speech: np.ndarray, tail_s: float = 1.5) -> dict:
    """Feed ``speech`` + silence in 10 ms chunks (as fast as possible) and time the first final.

    ``endpoint_ms`` = audio time from the last voiced frame to the chunk that produced the
    first final (pause detection) plus the wall-clock compute spent in that call.
    """
    x = np.concatenate([speech.astype(np.float32), np.zeros(round(tail_s * PIPELINE_SR), np.float32)])
    n = PIPELINE_SR // 100
    rms = np.sqrt(np.mean(speech[: len(speech) // n * n].astype(np.float64).reshape(-1, n) ** 2, axis=1))
    voiced = np.flatnonzero(rms > VOICED_RMS)
    speech_end = (int(voiced[-1]) + 1) * n if len(voiced) else len(speech)
    compute = []
    for i in range(0, len(x), n):
        t = time.perf_counter()
        events = engine.feed(AudioChunk(x[i : i + n], PIPELINE_SR, i // n, i / PIPELINE_SR))
        dt = time.perf_counter() - t
        compute.append(dt)
        if any(e.final for e in events):
            audio_ms = max(0.0, (i + n - speech_end) / PIPELINE_SR * 1000)
            return {"engine": engine.name, "endpoint_ms": audio_ms + dt * 1000,
                    "pause_ms": audio_ms, "compute_ms": dt * 1000,
                    "chunk_p50_ms": float(np.median(compute) * 1000),
                    "text": next(e.text for e in events if e.final)}
    t = time.perf_counter()
    events = engine.flush()
    return {"engine": engine.name, "endpoint_ms": tail_s * 1000 + (time.perf_counter() - t) * 1000,
            "pause_ms": tail_s * 1000, "compute_ms": (time.perf_counter() - t) * 1000,
            "chunk_p50_ms": float(np.median(compute) * 1000),
            "text": " ".join(e.text for e in events if e.final)}


def _bench() -> None:
    from hearing.synth import tts
    from hearing.testing import make_test_signal

    text = DEFAULT_TRANSCRIPT
    if tts.local_available():
        speech, source = tts.KokoroTTS().synthesize(text, "neutral"), "kokoro"
    else:
        speech, source = make_test_signal(2.5), "speechlike"
    engines: list[StreamingASR] = [MockASR()]
    if local_available():
        engines.append(WhisperASR())
    if os.environ.get("ELEVENLABS_API_KEY"):
        engines.append(ElevenLabsASR())
    print(f"device={device.pick()} speech={source} {len(speech) / PIPELINE_SR:.2f}s")
    for e in engines:
        try:
            endpoint_latency(e, speech)  # load weights / connect
            e.reset()
            r = endpoint_latency(e, speech)
        except AsrError as err:
            print(f"{e.name:<11} failed: {err}")
            continue
        finally:
            e.close()
        print(f"{r['engine']:<11} endpoint_ms={r['endpoint_ms']:7.1f} (pause {r['pause_ms']:.0f} + "
              f"compute {r['compute_ms']:.1f}) chunk_p50_ms={r['chunk_p50_ms']:.3f} text={r['text']!r}")


if __name__ == "__main__":
    _bench()
