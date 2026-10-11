"""Controlled voice-profile perturbations for offline identity-margin bounds.

The command writes rendered clips only below ``eval/shards/profile_perturbations`` (ignored by
git).  The committed artifact is a manifest containing source/render hashes and aggregate cosine
scores, never audio or embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import istft, resample, stft

SR = 16000
DEFAULT_GRID = (
    {"id": "baseline", "pitch_semitones": 0.0, "speed": 1.0, "formant_semitones": 0.0},
    {"id": "pitch-3", "pitch_semitones": -3.0, "speed": 1.0, "formant_semitones": 0.0},
    {"id": "pitch+3", "pitch_semitones": 3.0, "speed": 1.0, "formant_semitones": 0.0},
    {"id": "speed-10", "pitch_semitones": 0.0, "speed": 0.9, "formant_semitones": 0.0},
    {"id": "speed+10", "pitch_semitones": 0.0, "speed": 1.1, "formant_semitones": 0.0},
    {"id": "formant-2", "pitch_semitones": 0.0, "speed": 1.0, "formant_semitones": -2.0},
    {"id": "formant+2", "pitch_semitones": 0.0, "speed": 1.0, "formant_semitones": 2.0},
)


def fake_voice(hz: float, seconds: float = 1.0, seed: int = 0) -> np.ndarray:
    """Small deterministic voiced signal used by offline tests and dry runs."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * SR), dtype=np.float64) / SR
    x = 0.5 * np.sin(2 * np.pi * hz * t) + 0.2 * np.sin(4 * np.pi * hz * t)
    return np.asarray(x + rng.normal(0, 0.005, len(t)), dtype=np.float32)


def _bounded(x: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(np.nan_to_num(x), -1.0, 1.0), dtype=np.float32)


def _formant_warp(x: np.ndarray, semitones: float, sr: int) -> np.ndarray:
    if abs(semitones) < 1e-12:
        return x
    factor = 2.0 ** (float(semitones) / 12.0)
    _, _, z = stft(x, fs=sr, nperseg=512, noverlap=384, boundary="zeros", padded=True)
    freqs = np.arange(z.shape[0], dtype=np.float64)
    source = np.clip(freqs / factor, 0.0, freqs[-1])
    mag = np.abs(z)
    warped = np.vstack([np.interp(source, freqs, mag[:, i]) for i in range(z.shape[1])]).T
    _, y = istft(warped * np.exp(1j * np.angle(z)), fs=sr, nperseg=512, noverlap=384)
    return y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))


def transform_audio(
    audio: np.ndarray,
    *,
    pitch_semitones: float = 0.0,
    speed: float = 1.0,
    formant_semitones: float = 0.0,
    sr: int = SR,
) -> np.ndarray:
    """Apply independent pitch, speed, and formant controls without changing sample rate.

    ``speed`` changes duration (1.25 means 25% faster); pitch and formant shifts preserve
    duration so their cosine impact is not confounded by a shorter enrollment clip.
    """
    if speed <= 0 or not np.isfinite(speed):
        raise ValueError(f"speed must be finite and > 0, got {speed!r}")
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if not len(x):
        raise ValueError("audio must not be empty")
    if abs(speed - 1.0) > 1e-12:
        x = resample(x, max(1, round(len(x) / speed))).astype(np.float32)
    if abs(pitch_semitones) > 1e-12:
        ratio = 2.0 ** (float(pitch_semitones) / 12.0)
        shifted = resample(x, max(1, round(len(x) / ratio)))
        x = resample(shifted, len(x)).astype(np.float32)
    x = _formant_warp(x, formant_semitones, sr)
    return _bounded(x)


class Embedder:
    """Normalised embedding facade; the backend is injected for deterministic tests."""

    def __init__(self, backend):
        self.backend = backend

    def embed(self, audio: np.ndarray) -> np.ndarray:
        x = np.asarray(self.backend.embed(audio), dtype=np.float64).reshape(-1)
        if not x.size or not np.isfinite(x).all():
            raise ValueError("speaker encoder returned an invalid embedding")
        norm = np.linalg.norm(x)
        return x / norm if norm > 1e-12 else np.zeros_like(x)


class FakeBackend:
    """Deterministic spectral proxy, explicitly not evidence of speaker recognition."""

    def embed(self, audio):
        spectrum = np.abs(np.fft.rfft(audio, n=2048))
        return np.array([np.linalg.norm(b) for b in np.array_split(spectrum, 16)])


def measure_perturbations(audio: np.ndarray, embedder: Embedder, configs: list[dict]) -> list[dict]:
    reference = embedder.embed(audio)
    rows = []
    for config in configs:
        rendered = transform_audio(
            audio,
            pitch_semitones=config.get("pitch_semitones", 0.0),
            speed=config.get("speed", 1.0),
            formant_semitones=config.get("formant_semitones", 0.0),
        )
        cosine = float(np.dot(reference, embedder.embed(rendered)))
        rows.append({**config, "cosine": cosine, "cosine_degradation": 1.0 - cosine, "audio": rendered})
    return rows


def _sha256(audio: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(audio, np.float32).tobytes()).hexdigest()


def write_manifest(path: Path, rows: list[dict], *, source: str) -> dict:
    serial = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "audio"}
        item["sha256"] = _sha256(row["audio"])
        serial.append(item)
    manifest = {
        "schema": "profile-perturbation-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "sample_rate": SR,
        "source": source,
        "rows": serial,
    }
    source_path = Path(source)
    if source_path.is_file():
        manifest["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _real_embedder() -> Embedder:
    from hearing.models import load
    from hearing.tse.enrolled import MODEL, _unit_embedding

    backend = load(MODEL)
    return Embedder(type("UnitBackend", (), {"embed": lambda self, x: _unit_embedding(backend, x)})())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="profile WAV to perturb")
    parser.add_argument("--output", type=Path, default=Path("eval/shards/profile_perturbations"))
    parser.add_argument("--dry-run", action="store_true", help="use a deterministic proxy instead of ECAPA")
    args = parser.parse_args(argv)
    from hearing.audio.io import load, save

    audio, sr = load(args.profile, SR)
    embedder = Embedder(FakeBackend()) if args.dry_run else _real_embedder()
    rows = measure_perturbations(audio, embedder, list(DEFAULT_GRID))
    args.output.mkdir(parents=True, exist_ok=True)
    for row in rows:
        save(args.output / f"{row['id']}.wav", row["audio"], sr)
    write_manifest(args.output / "manifest.json", rows, source=str(args.profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
