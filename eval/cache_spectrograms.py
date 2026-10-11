"""Pre-compute reusable audio features for the benchmark fixture bank.

The cache contains one compressed ``npz`` per WAV stem and a small, committed
manifest of source/cache hashes.  Audio and arrays are intentionally written
under ``eval/shards/feature_cache`` (or an explicit output directory), where
the repository ignores them.

Examples::

    .venv/bin/python -m eval.cache_spectrograms
    .venv/bin/python -m eval.cache_spectrograms --fixtures /tmp/fixtures --out /tmp/cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = Path(os.environ.get("HEARING_FIXTURES_DIR", ROOT / "eval" / "fixtures" / "generated"))
DEFAULT_OUT = ROOT / "eval" / "shards" / "feature_cache"
N_FFT = 512
HOP = 160
N_MELS = 40
F_MIN = 0.0
F_MAX = 8_000.0
EPSILON = 1e-10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """Return Slaney-style triangular filters without a librosa dependency."""
    low = 2595.0 * np.log10(1.0 + F_MIN / 700.0)
    high = 2595.0 * np.log10(1.0 + min(F_MAX, sr / 2) / 700.0)
    points = np.linspace(low, high, n_mels + 2)
    hz = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for row, (left, centre, right) in enumerate(zip(bins[:-2], bins[1:-1], bins[2:])):
        centre = max(centre, left + 1)
        right = max(right, centre + 1)
        left = min(max(left, 0), filters.shape[1] - 1)
        centre = min(centre, filters.shape[1] - 1)
        right = min(right, filters.shape[1])
        if centre > left:
            filters[row, left:centre] = np.linspace(0.0, 1.0, centre - left, endpoint=False)
        if right > centre:
            filters[row, centre:right] = np.linspace(1.0, 0.0, right - centre, endpoint=False)
    return filters


def extract_features(audio: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    """Extract frame-aligned magnitude STFT, filterbank power, and log-mel arrays."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    signal = np.asarray(audio, dtype=np.float32).reshape(-1)
    pad = N_FFT // 2
    padded = np.pad(signal, (pad, pad), mode="reflect" if len(signal) > 1 else "constant")
    starts = np.arange(0, len(signal) + 1, HOP)
    frames = np.stack([padded[start : start + N_FFT] for start in starts if start + N_FFT <= len(padded)])
    windowed = frames * np.hanning(N_FFT).astype(np.float32)
    spectrum = np.fft.rfft(windowed, axis=1)
    stft = np.abs(spectrum).astype(np.float32)
    power = (stft * stft).astype(np.float32)
    filterbank = _mel_filterbank(sample_rate, N_FFT, N_MELS)
    energies = np.maximum(power @ filterbank.T, EPSILON).astype(np.float32)
    log_mel = np.log(energies).astype(np.float32)
    return {"stft": stft, "filterbank": energies, "log_mel": log_mel}


def _existing_entry(entries: list[dict], source: str, source_hash: str, output: Path) -> dict | None:
    for entry in entries:
        if (
            entry["source"] == source
            and entry["source_sha256"] == source_hash
            and entry["cache"] == str(output)
            and output.exists()
        ):
            return entry
    return None


def build_cache(fixtures: Path, output: Path) -> list[dict]:
    """Build or incrementally update the cache and return its manifest entries."""
    fixtures = Path(fixtures)
    output = Path(output)
    manifest_path = output / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old_entries = old.get("entries", [])
    parameters = {
        "n_fft": N_FFT,
        "hop_samples": HOP,
        "n_mels": N_MELS,
        "f_min_hz": F_MIN,
        "f_max_hz": F_MAX,
        "features": ["stft", "filterbank", "log_mel"],
    }
    if old.get("parameters") != parameters:
        old_entries = []
    entries: list[dict] = []
    fixture_dirs = sorted(path.parent for path in fixtures.glob("*/*/manifest.json"))
    for fixture_dir in fixture_dirs:
        relative_dir = fixture_dir.relative_to(fixtures)
        for source_path in sorted(fixture_dir.glob("*.wav")):
            source = str(source_path.relative_to(fixtures))
            source_hash = _sha256(source_path)
            cache_path = output / relative_dir / f"{source_path.stem}.npz"
            existing = _existing_entry(old_entries, source, source_hash, cache_path)
            if existing is None:
                audio, sample_rate = sf.read(source_path, dtype="float32", always_2d=False)
                features = extract_features(audio, int(sample_rate))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cache_path, **features)
                cache_hash = _sha256(cache_path)
                existing = {
                    "source": source,
                    "source_sha256": source_hash,
                    "cache": str(cache_path.relative_to(output)),
                    "cache_sha256": cache_hash,
                    "sample_rate": int(sample_rate),
                    "frames": int(features["stft"].shape[0]),
                }
            entries.append(existing)
    manifest = {
        "schema": 1,
        "parameters": parameters,
        "entries": entries,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    entries = build_cache(args.fixtures, args.out)
    print(f"cached {len(entries)} audio stems in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
