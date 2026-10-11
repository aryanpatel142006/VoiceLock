"""Whole-clip reference transcriber for WER (eval only; product ASR lives in hearing/synth/asr.py)."""

import json
import logging
import os
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from eval.metrics import wer
from hearing import device, models
from hearing.audio import io

log = logging.getLogger(__name__)

WHISPER_SR = 16000
DEFAULT_MODEL = "small.en"


@runtime_checkable
class Transcriber(Protocol):
    name: str

    def transcribe(self, x: np.ndarray, sr: int) -> str: ...


class MockTranscriber:
    """Offline and deterministic: the given text, else a placeholder from the signal length."""

    name = "mock"

    def __init__(self, transcript: str | None = None):
        self.transcript = transcript

    def transcribe(self, x: np.ndarray, sr: int) -> str:
        if self.transcript is not None:
            return self.transcript
        return f"mock transcript of {len(x) / sr:.2f} seconds"


def _whisper_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


def _model_cached(model: str = DEFAULT_MODEL) -> bool:
    root = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    return (
        any(root.glob(f"models--*faster-whisper-{model}*")) if root.exists() else False
    )


def _faster_whisper_device() -> tuple[str, str]:
    """(device, compute_type); MPS is unsupported by CTranslate2, so Macs use CPU int8."""
    if device.pick() == "cuda":
        return "cuda", "float16"
    return "cpu", "int8"


def _load_whisper(model: str = DEFAULT_MODEL):
    from faster_whisper import WhisperModel

    dev, compute = _faster_whisper_device()
    return WhisperModel(model, device=dev, compute_type=compute)


def _loader_name(model: str) -> str:
    name = f"faster-whisper-{model}"
    try:
        models.register_loader(name)(lambda: _load_whisper(model))
    except ValueError:
        pass  # already registered
    return name


class WhisperTranscriber:
    name = "whisper"

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model_name = model
        self._key = _loader_name(model)

    def transcribe(self, x: np.ndarray, sr: int) -> str:
        model = models.load(self._key)
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.mean(axis=-1)
        if sr != WHISPER_SR:
            x = io.resample(x, sr, WHISPER_SR)
        segments, _ = model.transcribe(x, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segments).strip()


def get_transcriber(name: str = "auto", transcript: str | None = None) -> Transcriber:
    if name == "mock":
        return MockTranscriber(transcript)
    if name == "whisper":
        return WhisperTranscriber()
    if name != "auto":
        raise ValueError(f"Unknown transcriber {name!r}; use auto, mock or whisper")
    if _whisper_available() and _model_cached():
        log.info("transcriber: whisper (%s)", DEFAULT_MODEL)
        return WhisperTranscriber()
    log.info(
        "transcriber: mock (faster-whisper missing or %s not cached)", DEFAULT_MODEL
    )
    return MockTranscriber(transcript)


def _reference(manifest: dict, which: str) -> str:
    t = manifest.get("transcript", manifest.get("transcripts"))
    if isinstance(t, dict):
        t = t.get(which, t.get("target"))
    if not isinstance(t, str):
        raise KeyError(f"manifest has no transcript for {which!r}")
    return t


def wer_for_fixture(
    fixture_dir: str | os.PathLike,
    transcriber: Transcriber,
    which: Literal["mix", "target"] = "mix",
) -> dict:
    """Transcribe ``<which>.wav`` and score it against the manifest's target transcript."""
    d = Path(fixture_dir)
    ref = _reference(json.loads((d / "manifest.json").read_text()), which)
    x, sr = io.load(d / f"{which}.wav")
    hyp = transcriber.transcribe(x, sr)
    return {"wer": wer(hyp, ref), "hyp": hyp, "ref": ref}
