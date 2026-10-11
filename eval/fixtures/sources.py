"""Speech and noise sources for the fixture bank.

* :class:`TinySource` reads the committed clips in ``eval/fixtures/tiny/`` and imports
  nothing beyond numpy/soundfile, so ``--tiny`` is offline by construction.
* :class:`LibriSpeechSource` streams ``openslr/librispeech_asr`` (test-clean) through
  ``datasets`` (imported lazily, network) and decodes the FLAC bytes with soundfile.
* :func:`make_babble` sums several utterances into babble; :func:`pink_noise` is 1/f-shaped
  Gaussian noise.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.fixtures.mix import fit_length, rms
from eval.metrics import normalize_text
from hearing.audio import io as aio

log = logging.getLogger(__name__)

TINY_DIR = Path(__file__).resolve().parent / "tiny"
LIBRISPEECH_DATASET = "openslr/librispeech_asr"
LIBRISPEECH_SPLIT = "test-clean"
BABBLE_TALKERS = 6


@dataclass
class Utterance:
    """One speech clip: float32 mono at ``sr`` plus its provenance."""

    id: str
    speaker_id: str
    text: str
    audio: np.ndarray
    sr: int = aio.PIPELINE_SR
    sex: str = ""  # "F" / "M" when known, else ""
    source: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return len(self.audio) / float(self.sr)

    @property
    def transcript(self) -> str:
        return normalize_text(self.text)


# --- noise ---------------------------------------------------------------------------------


def make_babble(utterances: list[Utterance], n: int) -> np.ndarray:
    """Sum ``utterances`` (each RMS-normalised and trimmed/looped to ``n``) into unit-RMS babble."""
    if len(utterances) < 2:
        raise ValueError("babble needs at least two talkers")
    acc = np.zeros(n, dtype=np.float64)
    for u in utterances:
        x = fit_length(np.asarray(u.audio, dtype=np.float64), n)
        r = rms(x)
        if r > 0:
            acc += x / r
    r = rms(acc)
    return (acc / r if r > 0 else acc).astype(np.float32)


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS 1/f (pink) noise: white Gaussian shaped by 1/sqrt(f) in the frequency domain."""
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.fft.rfftfreq(n)
    scale = np.ones_like(f)
    scale[1:] = 1.0 / np.sqrt(f[1:])
    scale[0] = 0.0  # no DC
    x = np.fft.irfft(spec * scale, n=n)
    r = rms(x)
    return (x / r if r > 0 else x).astype(np.float32)


# --- tiny (offline) -------------------------------------------------------------------------


class TinySource:
    """The committed 3.0 s clips under ``eval/fixtures/tiny/`` and their manifest."""

    def __init__(self, root: str | os.PathLike = TINY_DIR):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())

    def speech(self) -> list[Utterance]:
        out = []
        for entry in self.manifest["speech"]:
            x, sr = aio.load(self.root / entry["file"], aio.PIPELINE_SR)
            out.append(
                Utterance(
                    id=entry["id"],
                    speaker_id=str(entry["speaker_id"]),
                    text=entry["text"],
                    audio=x,
                    sr=sr,
                    sex=entry.get("sex", ""),
                    source={
                        "dataset": self.manifest["dataset"],
                        "split": self.manifest["split"],
                        "utterance": entry["id"],
                        "file": entry["file"],
                    },
                )
            )
        return out

    def noise(self) -> tuple[str, np.ndarray, dict]:
        """``(noise_type, audio, source)`` for the committed babble clip."""
        entry = self.manifest["noise"]
        x, _ = aio.load(self.root / entry["file"], aio.PIPELINE_SR)
        source = {
            "dataset": self.manifest["dataset"],
            "split": self.manifest["split"],
            "utterances": list(entry["source_ids"]),
            "file": entry["file"],
        }
        return entry["type"], x, source


# --- LibriSpeech (network) ------------------------------------------------------------------


def _decode_flac(raw: bytes) -> tuple[np.ndarray, int]:
    x, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    x = x.mean(axis=1, dtype=np.float32) if x.shape[1] > 1 else x[:, 0]
    return np.ascontiguousarray(x, dtype=np.float32), int(sr)


def _flac_duration_s(raw: bytes) -> float:
    info = sf.info(io.BytesIO(raw))
    return info.frames / float(info.samplerate)


class LibriSpeechSource:
    """Stream LibriSpeech test-clean from the Hugging Face Hub (``streaming=True``).

    ``datasets`` is imported inside :meth:`stream` so the offline paths never touch it.
    Audio is taken as raw FLAC bytes and decoded with soundfile (no torchcodec needed).
    """

    def __init__(self, split: str = LIBRISPEECH_SPLIT, dataset: str = LIBRISPEECH_DATASET):
        self.dataset = dataset
        self.split = split

    def stream(self, min_s: float = 0.0, max_s: float = float("inf")) -> Iterator[Utterance]:
        """Yield utterances in dataset order whose duration lies in ``[min_s, max_s]``."""
        from datasets import Audio, load_dataset

        config, hf_split = {"test-clean": ("clean", "test"), "test-other": ("other", "test")}[
            self.split
        ]
        ds = load_dataset(self.dataset, config, split=hf_split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))
        for ex in ds:
            raw = ex["audio"]["bytes"]
            dur = _flac_duration_s(raw)
            if not (min_s <= dur <= max_s):
                continue
            x, sr = _decode_flac(raw)
            if sr != aio.PIPELINE_SR:
                x = aio.resample(x, sr, aio.PIPELINE_SR)
            yield Utterance(
                id=str(ex["id"]),
                speaker_id=str(ex["speaker_id"]),
                text=str(ex["text"]),
                audio=x,
                sr=aio.PIPELINE_SR,
                source={"dataset": self.dataset, "split": self.split, "utterance": str(ex["id"])},
            )

    def collect(
        self,
        min_s: float,
        max_s: float,
        per_speaker: int = 3,
        progress: bool = True,
    ) -> dict[str, list[Utterance]]:
        """Keep up to ``per_speaker`` utterances per speaker (stream order), grouped by speaker."""
        pool: dict[str, list[Utterance]] = {}
        for seen, u in enumerate(self.stream(min_s, max_s), start=1):
            bucket = pool.setdefault(u.speaker_id, [])
            if len(bucket) < per_speaker:
                bucket.append(u)
            if progress and seen % 50 == 0:
                print(
                    f"\r  streaming {self.split}: {seen} candidates, {len(pool)} speakers",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
        if progress:
            print(file=sys.stderr)
        return pool


def pick_pairs(
    pool: dict[str, list[Utterance]], n_pairs: int, rng: np.random.Generator
) -> list[tuple[Utterance, Utterance]]:
    """Choose ``n_pairs`` (target, interferer) pairs of different speakers, deterministically.

    Speakers are drawn without replacement while enough remain, then recycled, so the two
    talkers in a pair are always different speakers.
    """
    speakers = sorted(pool)
    if len(speakers) < 2:
        raise ValueError("need at least two speakers")
    order: list[str] = []
    while len(order) < 2 * n_pairs:
        order.extend(str(s) for s in rng.permutation(speakers))
    pairs = []
    for k in range(n_pairs):
        a, b = order[2 * k], order[2 * k + 1]
        if a == b:  # only possible at a recycle boundary
            b = order[2 * k + 2] if 2 * k + 2 < len(order) else next(s for s in speakers if s != a)
        ua = pool[a][int(rng.integers(len(pool[a])))]
        ub = pool[b][int(rng.integers(len(pool[b])))]
        pairs.append((ua, ub))
    return pairs


def pick_babble_talkers(
    pool: dict[str, list[Utterance]],
    exclude: set[str],
    k: int,
    rng: np.random.Generator,
) -> list[Utterance]:
    """``k`` utterances from ``k`` distinct speakers not in ``exclude``."""
    speakers = [s for s in sorted(pool) if s not in exclude]
    if len(speakers) < k:
        raise ValueError(f"need {k} babble speakers, have {len(speakers)}")
    chosen = rng.choice(len(speakers), size=k, replace=False)
    return [pool[speakers[int(i)]][int(rng.integers(len(pool[speakers[int(i)]])))] for i in chosen]
