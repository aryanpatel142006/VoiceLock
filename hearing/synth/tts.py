"""TTS adapters: ElevenLabs Flash v2.5 WebSocket, local Kokoro-82M (ONNX), offline mock tone.

Every engine turns one phrase into 16 kHz mono float32: ``stream(text, voice)`` yields
chunks as they are produced (so playback can start on the first one) and
``synthesize`` returns the whole phrase. ``voice`` is a bucket from
``config/voices.yaml`` (``neutral``, ``female_adult``, ...), ``None`` for the default,
or a raw engine voice id. Buckets reflect apparent gender/age: they pick a voice and are
never logged here.

``get_tts()`` picks an engine from ``HEARING_TTS`` (``mock`` / ``local`` /
``elevenlabs`` / ``auto``); ``auto`` prefers ElevenLabs when ``ELEVENLABS_API_KEY`` is
set, then Kokoro if installed with weights, then the mock, wrapping real engines in
``FallbackTTS``.

``python -m hearing.synth.tts`` prints phrase latency for every available engine.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import importlib.util
import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlencode

import numpy as np
import yaml

from hearing import models
from hearing.audio.io import PIPELINE_SR, pcm16_to_float, resample

log = logging.getLogger(__name__)

VOICES_PATH = Path(__file__).resolve().parents[2] / "config" / "voices.yaml"
ELEVENLABS_TTS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input"
ENGINES = ("elevenlabs", "local", "mock")


class TtsError(RuntimeError):
    """A TTS engine failed (connection, auth, quota, timeout, missing key or weights)."""


@runtime_checkable
class TtsEngine(Protocol):
    name: str
    sample_rate: int  # output rate

    def stream(self, text: str, voice: str | None) -> Iterator[np.ndarray]: ...
    def synthesize(self, text: str, voice: str | None) -> np.ndarray: ...
    def close(self) -> None: ...


def _concat(chunks) -> np.ndarray:
    chunks = list(chunks)
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


# --- voices.yaml -------------------------------------------------------------


@functools.cache
def _voices(path: str) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if "neutral" not in data.get("buckets", {}):
        raise ValueError(f"{path}: voices.yaml must define a 'neutral' bucket")
    return data


def load_voices() -> dict:
    """The parsed voices file (``HEARING_VOICES`` overrides ``config/voices.yaml``)."""
    return _voices(os.environ.get("HEARING_VOICES", str(VOICES_PATH)))


def bucket_names() -> list[str]:
    """Bucket names in file order (``neutral`` first): the user's override choices."""
    return list(load_voices()["buckets"])


def resolve_voice(voice: str | None, engine: str) -> str:
    """Engine voice id for a bucket; ``None`` means the default; other strings pass through."""
    if engine not in ENGINES:
        raise ValueError(f"Unknown TTS engine {engine!r}; use one of {ENGINES}")
    data = load_voices()
    bucket = data["buckets"].get(voice or data.get("default", "neutral"))
    if bucket is None:
        return voice
    entry = bucket["mock_hz" if engine == "mock" else engine]
    return str(entry["id"] if isinstance(entry, dict) else entry)


def local_speed(voice: str | None) -> float:
    """Kokoro speaking-rate multiplier of a bucket (``local_speed`` in voices.yaml, default 1.0)."""
    data = load_voices()
    bucket = data["buckets"].get(voice or data.get("default", "neutral")) or {}
    return float(bucket.get("local_speed", 1.0))


# --- mock -------------------------------------------------------------------


class MockTTS:
    """Deterministic offline TTS: one Hann-ramped tone per word, pitch from the bucket.

    Each word lasts 0.12 s + 0.03 s per character (≤ 0.5 s), followed by 0.06 s of
    silence. Unknown voices use the neutral pitch.
    """

    name = "mock"
    sample_rate = PIPELINE_SR

    def __init__(self, amplitude: float = 0.3, gap_s: float = 0.06):
        self.amplitude = amplitude
        self.gap = np.zeros(round(gap_s * PIPELINE_SR), np.float32)

    def _hz(self, voice: str | None) -> float:
        try:
            return float(resolve_voice(voice, "mock"))
        except ValueError:
            return float(resolve_voice(None, "mock"))

    def stream(self, text: str, voice: str | None) -> Iterator[np.ndarray]:
        hz = self._hz(voice)
        for word in text.split():
            n = round(min(0.5, 0.12 + 0.03 * len(word)) * PIPELINE_SR)
            t = np.arange(n) / PIPELINE_SR
            ramp = min(n // 2, PIPELINE_SR // 200)  # 5 ms fades, no clicks
            env = np.ones(n)
            env[:ramp] = env[n - ramp :][::-1] = 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
            tone = (self.amplitude * env * np.sin(2 * np.pi * hz * t)).astype(np.float32)
            yield np.concatenate([tone, self.gap])

    def synthesize(self, text: str, voice: str | None) -> np.ndarray:
        return _concat(self.stream(text, voice))

    def close(self) -> None:
        pass


# --- local: Kokoro-82M via kokoro-onnx ----------------------------------------

KOKORO_MODELS = ("kokoro-v1.0.onnx", "kokoro-v1.0.int8.onnx", "kokoro-v1.0.fp16.onnx")
KOKORO_VOICES = "voices-v1.0.bin"
KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"


def kokoro_dir() -> Path:
    """``HEARING_KOKORO_DIR`` or ``<models cache_dir>/kokoro``."""
    env = os.environ.get("HEARING_KOKORO_DIR")
    return Path(env).expanduser() if env else models.cache_dir() / "kokoro"


def _kokoro_files() -> tuple[Path, Path] | None:
    d = kokoro_dir()
    voices = d / KOKORO_VOICES
    for m in KOKORO_MODELS:
        if (d / m).exists() and voices.exists():
            return d / m, voices
    return None


def local_available() -> bool:
    """kokoro-onnx installed and its model + voices files in :func:`kokoro_dir`."""
    return importlib.util.find_spec("kokoro_onnx") is not None and _kokoro_files() is not None


def _load_kokoro():
    files = _kokoro_files()
    if files is None:
        raise TtsError(
            f"Kokoro weights missing in {kokoro_dir()}: download {KOKORO_MODELS[0]} and "
            f"{KOKORO_VOICES} from {KOKORO_RELEASE}"
        )
    from kokoro_onnx import Kokoro

    return Kokoro(str(files[0]), str(files[1]))


try:
    models.register_loader("kokoro-82m")(_load_kokoro)
except ValueError:  # module re-imported
    pass


# Cut the first piece to N words (0 = off). Measured T-094 on the M4 Pro (load ~12): fp32 CPU takes 193 ms for
# 1 word, 242 ms for 3, 355 ms for 7 (int8 and CoreML: 560-1500 ms), so a 3-word head plays well under 400 ms.
FIRST_CHUNK_WORDS = 3


def split_sentences(text: str, max_words: int = 8, first_words: int | None = None) -> list[str]:
    """``text`` as sentences, long ones cut at commas, so the first can play while the rest renders.
    The first piece is cut to ``first_words`` words (0 = off) so the first word is heard sooner."""
    first_words = FIRST_CHUNK_WORDS if first_words is None else first_words
    out = _split_sentences(text, max_words)
    if first_words and out:
        words = out[0].split()
        if len(words) > first_words + 1:
            out[0:1] = [" ".join(words[:first_words]), " ".join(words[first_words:])]
    return out


def _split_sentences(text: str, max_words: int) -> list[str]:
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        words, parts = sentence.split(), re.split(r"(?<=,)\s+", sentence)
        if len(words) > max_words and len(parts) > 1:
            out.extend(p for p in parts if p.strip())
        elif sentence.strip():
            out.append(sentence)
    return out


class KokoroTTS:
    """Kokoro-82M on onnxruntime (CPU), 24 kHz resampled to 16 kHz, one chunk per sentence
    (first audio after the first sentence, not after the whole phrase; T-074)."""

    name = "kokoro"
    sample_rate = PIPELINE_SR

    def __init__(self, speed: float = 1.0, lang: str = "en-us"):
        self.speed, self.lang = speed, lang

    def stream(self, text: str, voice: str | None) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        for sentence in split_sentences(text):
            try:
                kokoro = models.load("kokoro-82m")
                x, sr = kokoro.create(sentence, voice=resolve_voice(voice, "local"),
                                      speed=self.speed * local_speed(voice), lang=self.lang)
            except TtsError:
                raise
            except Exception as e:  # any local failure degrades to the fallback
                raise TtsError(f"Kokoro failed: {e}") from e
            yield np.clip(resample(np.asarray(x, np.float32), int(sr), PIPELINE_SR), -1.0, 1.0)

    def synthesize(self, text: str, voice: str | None) -> np.ndarray:
        return _concat(self.stream(text, voice))

    def close(self) -> None:
        pass


# --- ElevenLabs Flash WebSocket -------------------------------------------------

Connect = Callable[[str, dict], object]


def _ws_connect(url: str, headers: dict):
    from websockets.sync.client import connect

    return connect(url, additional_headers=headers, open_timeout=5, max_size=None)


class ElevenLabsTTS:
    """ElevenLabs ``stream-input`` WebSocket, one connection per phrase, raw PCM 16 kHz out.

    Protocol: open with ``{"text": " "}`` + voice settings, send the phrase with
    ``flush``, close with ``{"text": ""}``, then read ``audio`` chunks until ``isFinal``.
    """

    name = "elevenlabs"
    sample_rate = PIPELINE_SR

    def __init__(
        self,
        api_key: str | None = None,
        connect: Connect | None = None,
        model_id: str = "eleven_flash_v2_5",
        recv_timeout_s: float = 5.0,
        voice_settings: dict | None = None,
    ):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise TtsError("ELEVENLABS_API_KEY is not set")
        self._connect = connect or _ws_connect
        self.model_id = model_id
        self.recv_timeout_s = recv_timeout_s
        self.voice_settings = voice_settings or {"stability": 0.5, "similarity_boost": 0.8, "speed": 1.0}

    def url(self, voice: str | None) -> str:
        params = {"model_id": self.model_id, "output_format": f"pcm_{PIPELINE_SR}"}
        voice_id = resolve_voice(voice, "elevenlabs")
        return f"{ELEVENLABS_TTS_URL.format(voice_id=voice_id)}?{urlencode(params)}"

    def stream(self, text: str, voice: str | None) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        try:
            sock = self._connect(self.url(voice), {"xi-api-key": self.api_key})
        except Exception as e:
            raise TtsError(f"ElevenLabs TTS connect failed: {e}") from e
        try:
            sock.send(json.dumps({"text": " ", "voice_settings": self.voice_settings}))
            sock.send(json.dumps({"text": text.strip() + " ", "flush": True}))
            sock.send(json.dumps({"text": ""}))
            while True:
                try:
                    msg = json.loads(sock.recv(timeout=self.recv_timeout_s))
                except TimeoutError:
                    raise TtsError(f"ElevenLabs TTS timed out after {self.recv_timeout_s} s") from None
                if msg.get("error"):
                    raise TtsError(f"ElevenLabs TTS {msg.get('error')}: {msg.get('message')}")
                if msg.get("audio"):
                    yield pcm16_to_float(base64.b64decode(msg["audio"]))
                if msg.get("isFinal"):
                    return
        except TtsError:
            raise
        except Exception as e:  # closed socket etc.
            raise TtsError(f"ElevenLabs TTS failed: {e}") from e
        finally:
            with contextlib.suppress(Exception):
                sock.close()

    def synthesize(self, text: str, voice: str | None) -> np.ndarray:
        return _concat(self.stream(text, voice))

    def close(self) -> None:
        pass


# --- ElevenLabs mock: Flash v2.5 timing, offline ------------------------------------

FLASH_FIRST_AUDIO_S = 0.10  # ~75 ms model + network to the first PCM chunk (Flash v2.5 docs)
FLASH_CHUNK_S = 0.25  # audio per message after the first


class MockElevenLabsSocket:
    """A fake ``stream-input`` socket: speaks the real protocol (init, text+flush, close ->
    ``audio`` messages, ``isFinal``) with Flash v2.5-like timing, tones from :class:`MockTTS`.
    Plug it into :class:`ElevenLabsTTS` through ``connect=`` to exercise the real adapter offline."""

    def __init__(self, url: str, first_audio_s: float = FLASH_FIRST_AUDIO_S, chunk_s: float = FLASH_CHUNK_S):
        self.url, self.first_audio_s, self.chunk_s = url, first_audio_s, chunk_s
        self.sent: list[dict] = []
        self._out: list[dict] = []
        self._t0 = 0.0
        self.closed = False
        self._sent_audio = False

    def send(self, data: str) -> None:
        msg = json.loads(data)
        self.sent.append(msg)
        if msg.get("flush") and msg.get("text", "").strip():
            self._t0 = time.perf_counter()
            tone = _concat(MockTTS().stream(msg["text"], None))
            step = round(self.chunk_s * PIPELINE_SR)
            for i in range(0, len(tone), step):
                pcm = (np.clip(tone[i : i + step], -1, 1) * 32767).astype("<i2").tobytes()
                self._out.append({"audio": base64.b64encode(pcm).decode()})
        elif msg.get("text") == "":
            self._out.append({"isFinal": True})

    def recv(self, timeout: float | None = None) -> str:
        if not self._out:
            raise TimeoutError
        msg = self._out.pop(0)
        if msg.get("audio"):
            # first chunk at t0 + first_audio_s, the rest generated faster than real time
            due = self._t0 + self.first_audio_s if not self._sent_audio else time.perf_counter() + self.chunk_s * 0.4
            self._sent_audio = True
            time.sleep(max(0.0, due - time.perf_counter()))
        return json.dumps(msg)

    def close(self) -> None:
        self.closed = True


def mock_elevenlabs_connect(url: str, headers: dict) -> MockElevenLabsSocket:
    return MockElevenLabsSocket(url)


def mock_elevenlabs() -> ElevenLabsTTS:
    """:class:`ElevenLabsTTS` over :class:`MockElevenLabsSocket` (``HEARING_TTS=elevenlabs-mock``)."""
    return ElevenLabsTTS(api_key="mock", connect=mock_elevenlabs_connect)


# --- fallback and selection -----------------------------------------------------


class FallbackTTS:
    """Use ``primary`` until it fails, then ``fallback`` for the rest of the session.

    A failure before any audio re-speaks the phrase with the fallback; a failure after
    audio was yielded truncates that phrase (never repeats words) and degrades.
    """

    sample_rate = PIPELINE_SR

    def __init__(self, primary: TtsEngine, fallback: TtsEngine):
        self.primary, self.fallback = primary, fallback
        self.degraded = False
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return (self.fallback if self.degraded else self.primary).name

    def stream(self, text: str, voice: str | None) -> Iterator[np.ndarray]:
        if not self.degraded:
            started = False
            try:
                for chunk in self.primary.stream(text, voice):
                    started = True
                    yield chunk
                return
            except Exception as e:  # noqa: BLE001 - any engine failure must not kill the session
                log.warning("TTS %s failed (%s); degrading to %s", self.primary.name, e, self.fallback.name)
                self.degraded, self.last_error = True, str(e)
                if started:
                    return
        yield from self.fallback.stream(text, voice)

    def synthesize(self, text: str, voice: str | None) -> np.ndarray:
        return _concat(self.stream(text, voice))

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()


def get_tts(name: str | None = None) -> TtsEngine:
    """``mock`` | ``local`` | ``elevenlabs`` | ``auto`` (default: ``HEARING_TTS`` or auto)."""
    name = (name or os.environ.get("HEARING_TTS") or "auto").lower()
    if name == "mock":
        return MockTTS()
    if name in ("local", "kokoro"):
        return KokoroTTS()
    if name == "elevenlabs":
        return ElevenLabsTTS()
    if name == "elevenlabs-mock":
        return mock_elevenlabs()
    if name != "auto":
        raise ValueError(f"Unknown TTS {name!r}; use auto, mock, local, elevenlabs or elevenlabs-mock")
    local = local_available()
    if os.environ.get("ELEVENLABS_API_KEY"):
        log.info("tts: elevenlabs (fallback %s)", "kokoro" if local else "mock")
        return FallbackTTS(ElevenLabsTTS(), KokoroTTS() if local else MockTTS())
    if local:
        log.info("tts: kokoro (fallback mock)")
        return FallbackTTS(KokoroTTS(), MockTTS())
    log.info("tts: mock (no ELEVENLABS_API_KEY, Kokoro weights not in %s)", kokoro_dir())
    return MockTTS()


def phrase_latency(engine: TtsEngine, text: str, voice: str | None = None) -> dict:
    """Wall-clock time to the first audio chunk and to the whole phrase, plus RTF."""
    t = time.perf_counter()
    first = None
    n = 0
    for chunk in engine.stream(text, voice):
        if first is None:
            first = time.perf_counter() - t
        n += len(chunk)
    total = time.perf_counter() - t
    audio_s = n / engine.sample_rate
    return {
        "engine": engine.name,
        "first_audio_ms": 1000 * (first if first is not None else total),
        "total_ms": 1000 * total,
        "audio_s": audio_s,
        "rtf": total / audio_s if audio_s else float("inf"),
    }


SAMPLE_TEXT = "Meet me by the fountain at noon, and bring your umbrella."


def synth_test(engine: TtsEngine, text: str = SAMPLE_TEXT, buckets: list[str] | None = None,
               out_dir: Path | None = None) -> list[dict]:
    """Render ``text`` once per bucket; one row each: voice id, first-audio ms, RTF, wav path.
    Never opens an audio device. Raises TtsError when the engine fails (wrong key, no network)."""
    from hearing.audio.io import save as save_wav

    rows = []
    for bucket in buckets or bucket_names():
        row = phrase_latency(engine, text, bucket) | {"bucket": bucket}
        row["voice"] = resolve_voice(bucket, "elevenlabs" if engine.name == "elevenlabs" else "local")
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            row["wav"] = str(out_dir / f"{engine.name}_{bucket}.wav")
            save_wav(row["wav"], engine.synthesize(text, bucket), PIPELINE_SR)
        rows.append(row)
    return rows


def _bench() -> None:
    from hearing import device

    text = "see you at noon by the fountain"
    engines: list[TtsEngine] = [MockTTS()]
    if local_available():
        engines.append(KokoroTTS())
    if os.environ.get("ELEVENLABS_API_KEY"):
        engines.append(ElevenLabsTTS())
    print(f"device={device.pick()} text={text!r}")
    for e in engines:
        try:
            phrase_latency(e, "warm up", None)  # load weights / DNS
            runs = [phrase_latency(e, text, "neutral") for _ in range(3)]
        except TtsError as err:
            print(f"{e.name:<11} failed: {err}")
            continue
        first = np.median([r["first_audio_ms"] for r in runs])
        total = np.median([r["total_ms"] for r in runs])
        rtf = np.median([r["rtf"] for r in runs])
        print(f"{e.name:<11} first_audio_ms={first:8.1f} total_ms={total:8.1f} "
              f"audio_s={runs[0]['audio_s']:.2f} rtf={rtf:.3f}")


if __name__ == "__main__":
    _bench()
