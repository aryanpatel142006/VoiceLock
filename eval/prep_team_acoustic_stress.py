"""Prepare deterministic room-acoustic stress pairs for the team demo.

The input is a profile directory containing ``aryan`` and ``joseph`` subdirectories
(version directories are allowed) and one venue/babble WAV.  Outputs are generated
WAVs plus an index manifest; no recordings are committed to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from hearing.audio import io as aio

SAMPLE_RATE = 16_000
RT60_SECONDS = (0.2, 0.5)
SNR_DB = (5.0, 10.0)
PROFILE_NAMES = ("aryan", "joseph")


def make_rir(
    rt60_s: float, sample_rate: int = SAMPLE_RATE, *, seed: int = 278
) -> np.ndarray:
    """Synthetic signed diffuse tail with direct path and unit total energy.

    The tail amplitude envelope falls 60 dB per RT60; this is not a measured
    venue response. Direct and diffuse components each carry half the energy.
    """
    if not np.isfinite(rt60_s) or rt60_s <= 0:
        raise ValueError("rt60_s must be finite and positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    length = max(2, round(rt60_s * sample_rate * 2))
    t = np.arange(1, length, dtype=np.float64) / sample_rate
    tail = np.random.default_rng(seed).standard_normal(length - 1)
    tail *= np.exp(-np.log(1000) * t / rt60_s)
    tail /= np.linalg.norm(tail)
    rir = np.concatenate(([1.0], tail)) / np.sqrt(2)
    return rir.astype(np.float32)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _fit(x: np.ndarray, size: int) -> np.ndarray:
    if not len(x):
        raise ValueError("audio must not be empty")
    return np.resize(np.asarray(x, dtype=np.float32), size)


def _scale_noise(target: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    target_rms, noise_rms = _rms(target), _rms(noise)
    if target_rms == 0 or noise_rms == 0:
        raise ValueError("target and noise must be non-silent")
    return noise * (target_rms / noise_rms / 10 ** (snr_db / 20))


def _profile_pieces(root: Path) -> list[tuple[str, Path]]:
    pieces: list[tuple[str, Path]] = []
    for name in PROFILE_NAMES:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"missing profile directory: {directory}")
        found = sorted(directory.rglob("*.wav"))
        if not found:
            raise FileNotFoundError(f"no WAV profile pieces under {directory}")
        pieces.extend((name, path) for path in found)
    return pieces


def prepare(profiles: Path, noise_path: Path, output: Path) -> Path:
    """Render all profile × RT60 × SNR combinations and return ``manifest.json``."""
    profiles, noise_path, output = Path(profiles), Path(noise_path), Path(output)
    if not noise_path.is_file():
        raise FileNotFoundError(f"noise file not found: {noise_path}")
    noise, noise_sr = aio.load(noise_path, SAMPLE_RATE)
    if noise_sr != SAMPLE_RATE:
        raise ValueError(f"noise must be {SAMPLE_RATE} Hz")
    pieces = _profile_pieces(profiles)
    output.mkdir(parents=True, exist_ok=True)
    pairs = []
    for speaker, source_path in pieces:
        source, source_sr = aio.load(source_path, SAMPLE_RATE)
        if source_sr != SAMPLE_RATE:
            raise ValueError(f"profile must be {SAMPLE_RATE} Hz: {source_path}")
        for rt60_s in RT60_SECONDS:
            # Stable across output locations and Python processes, unlike hash().
            source_id = source_path.relative_to(profiles).as_posix()
            digest = hashlib.sha256(source_id.encode()).hexdigest()[:16]
            rir_seed = int(digest, 16)
            reverberant = fftconvolve(source, make_rir(rt60_s, seed=rir_seed), mode="full")
            for snr_db in SNR_DB:
                scaled_noise = _scale_noise(reverberant, _fit(noise, len(reverberant)), snr_db)
                mix = reverberant + scaled_noise
                peak = max(
                    float(np.max(np.abs(mix))),
                    float(np.max(np.abs(reverberant))),
                    1e-12,
                )
                gain = min(1.0, 0.9 / peak)
                target, mix = reverberant * gain, mix * gain
                stem = f"{speaker}_{source_path.stem}_{digest}_rt60-{rt60_s:g}_snr-{snr_db:g}"
                target_path, mix_path = output / f"{stem}_target.wav", output / f"{stem}_mix.wav"
                aio.save(target_path, target.astype(np.float32), SAMPLE_RATE)
                aio.save(mix_path, mix.astype(np.float32), SAMPLE_RATE)
                # Report the SNR of the actual PCM16 pair, including quantization.
                saved_target, _ = aio.load(target_path, SAMPLE_RATE)
                saved_mix, _ = aio.load(mix_path, SAMPLE_RATE)
                measured = 20 * np.log10(_rms(saved_target) / _rms(saved_mix - saved_target))
                pairs.append(
                    {
                        "id": stem,
                        "speaker": speaker,
                        "source": str(source_path),
                        "rir_seed": rir_seed,
                        "rt60_s": rt60_s,
                        "snr_db": snr_db,
                        "measured_snr_db": round(float(measured), 6),
                        "target": target_path.name,
                        "mix": mix_path.name,
                    }
                )
    manifest = {
        "version": 2,
        "acoustics": "synthetic_not_measured_venue",
        "rir_method": "seeded_signed_exponential_unit_energy",
        "kind": "team_acoustic_stress",
        "sample_rate": SAMPLE_RATE,
        "rt60_s": list(RT60_SECONDS),
        "snr_db": list(SNR_DB),
        "noise": str(noise_path),
        "pairs": pairs,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.profiles, args.noise, args.out))


if __name__ == "__main__":
    main()
