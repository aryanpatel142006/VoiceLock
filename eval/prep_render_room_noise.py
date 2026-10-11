"""Prepare deterministic room-response plus additive-noise evaluation renders.

This is preparation only; it does not alter production profiles.  Inputs are local clean
recordings, noise files, and verified RIR files.  A synthetic RIR is available explicitly for
offline smoke experiments and is labelled as such in every manifest.

Example::

    python -m eval.prep_render_room_noise --clean-dir recordings/clean \
      --noise-dir cache/noise --rir-dir cache/rir --out /tmp/room-renders --initial 12
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from eval.fixtures.mix import fit_length, rms
from hearing.audio import io as aio

SNRS_DB = (0.0, 5.0, 10.0, 20.0)
ROLES = ("target_a", "target_b")
MAX_VARIANTS = 600
DEFAULT_IR = np.array([1.0, 0.35, 0.12, 0.04], dtype=np.float32)


def _audio_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".aiff", ".aif"})


def _scale_to_snr(target: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    target_rms = rms(target)
    noise_rms = rms(noise)
    if target_rms <= 0 or noise_rms <= 0:
        raise ValueError("target and noise must both be non-silent")
    return noise * (target_rms / noise_rms) / (10.0 ** (snr_db / 20.0))


def convolve_room(signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Causal full convolution, retaining the full tail and float32 determinism."""
    x = np.asarray(signal, dtype=np.float64)
    h = np.asarray(ir, dtype=np.float64)
    if x.ndim != 1 or h.ndim != 1 or not x.size or not h.size:
        raise ValueError("signal and IR must be non-empty mono vectors")
    return np.asarray(fftconvolve(x, h, mode="full"), dtype=np.float32)


def _fit_pair(target: np.ndarray, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = max(len(target), len(noise))
    return fit_length(target, n).astype(np.float32), fit_length(noise, n).astype(np.float32)


def render_variant(
    clean: np.ndarray,
    noise: np.ndarray,
    ir: np.ndarray,
    snr_db: float,
    *,
    sample_rate: int = aio.PIPELINE_SR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(mix, room_target, scaled_noise)`` with exact additive stems."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    target = convolve_room(clean, ir)
    target, noise = _fit_pair(target, noise)
    noise = _scale_to_snr(target, noise, snr_db)
    return (target + noise).astype(np.float32), target, noise.astype(np.float32)


def _synthetic_ir(sample_rate: int) -> np.ndarray:
    """Short deterministic acoustic placeholder; it is never labelled measured."""
    taps = DEFAULT_IR.copy()
    return taps / np.sqrt(np.sum(taps * taps)).astype(np.float32)


def _load_ir(path: Path, sample_rate: int) -> tuple[np.ndarray, dict]:
    ir, sr = aio.load(path, sr=None)
    if sr != sample_rate:
        ir = aio.resample(ir, sr, sample_rate)
    if rms(ir) <= 0:
        raise ValueError(f"RIR is silent: {path}")
    return ir.astype(np.float32), {
        "kind": "measured_or_cached",
        "path": str(path),
        "source_sample_rate": int(sr),
        "sample_rate": sample_rate,
        "normalization": "unit_energy",
    }


def _load_at_rate(path: Path, sample_rate: int) -> tuple[np.ndarray, int]:
    audio, source_rate = aio.load(path, sr=None)
    if source_rate != sample_rate:
        audio = aio.resample(audio, source_rate, sample_rate)
    return audio.astype(np.float32), int(source_rate)


def _normalise_ir(ir: np.ndarray) -> np.ndarray:
    energy = float(np.sqrt(np.sum(np.asarray(ir, dtype=np.float64) ** 2)))
    if energy <= 0:
        raise ValueError("IR is silent")
    return (np.asarray(ir, dtype=np.float32) / energy).astype(np.float32)


def render(
    clean_dir: Path,
    noise_dir: Path,
    out_dir: Path,
    *,
    rir_dir: Path | None = None,
    limit: int = 12,
    roles: tuple[str, ...] = ROLES,
    sample_rate: int = aio.PIPELINE_SR,
    synthetic_ir: bool = True,
) -> list[dict]:
    """Render a bounded, restartable shard and write one manifest per variant."""
    if limit < 1 or limit > MAX_VARIANTS:
        raise ValueError(f"limit must be between 1 and {MAX_VARIANTS}")
    clean_files, noise_files = _audio_files(clean_dir), _audio_files(noise_dir)
    if not clean_files or not noise_files:
        raise FileNotFoundError("clean and noise directories must contain local audio files")
    rir_files = _audio_files(rir_dir) if rir_dir else []
    if rir_files:
        ir, ir_source = _load_ir(rir_files[0], sample_rate)
    elif synthetic_ir:
        ir, ir_source = _synthetic_ir(sample_rate), {
            "kind": "SYNTHETIC_acoustic_IR",
            "reason": "no local verified RIR was available",
            "sample_rate": sample_rate,
        }
    else:
        raise FileNotFoundError("no local RIR found; pass synthetic_ir=True explicitly for a labelled fallback")
    ir = _normalise_ir(ir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    prior: dict[str, dict] = {}
    if index_path.exists():
        prior = {row["id"]: row for row in json.loads(index_path.read_text()).get("variants", [])}
    rows: list[dict] = list(prior.values())
    planned = [(c, n, role, snr) for c in clean_files for n in noise_files for role in roles for snr in SNRS_DB]
    for index, (clean_path, noise_path, role, snr_db) in enumerate(planned[:limit]):
        clean, clean_sr = _load_at_rate(clean_path, sample_rate)
        noise, noise_sr = _load_at_rate(noise_path, sample_rate)
        mix, target, scaled_noise = render_variant(clean, noise, ir, snr_db, sample_rate=sample_rate)
        variant_id = f"{index:04d}_{role}_snr{int(snr_db)}"
        tmp = out_dir / f".{variant_id}.tmp"
        final = out_dir / variant_id
        if final.exists():
            continue
        tmp.mkdir(exist_ok=False)
        aio.save(tmp / "mix.wav", mix, sample_rate)
        aio.save(tmp / "target.wav", target, sample_rate)
        aio.save(tmp / "noise.wav", scaled_noise, sample_rate)
        metadata = {
            "id": variant_id,
            "sample_rate": sample_rate,
            "duration_s": len(mix) / sample_rate,
            "snr_db_declared": snr_db,
            "snr_db_measured": 20.0 * np.log10(rms(target) / rms(scaled_noise)),
            "target_role": role,
            "source": {"clean": str(clean_path), "noise": str(noise_path)},
            "ir": ir_source | {"normalization": "unit_energy", "tap_count": len(ir)},
            "additivity": "mix = target + noise; no clipping or peak normalization",
            "input_sample_rates": {"clean": clean_sr, "noise": noise_sr},
            "simulated_distance": False,
            "provenance_note": "Room response is a digital convolution; it is not a measured distance.",
        }
        (tmp / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(tmp, final)
        prior[variant_id] = metadata
        rows = list(prior.values())
    index_tmp = out_dir / ".index.json.tmp"
    index_tmp.write_text(json.dumps({"count": len(rows), "variants": rows}, indent=2) + "\n")
    os.replace(index_tmp, out_dir / "index.json")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--noise-dir", type=Path, required=True)
    parser.add_argument("--rir-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--initial", type=int, default=12, help="initial checkpoint size (default: 12)")
    parser.add_argument("--limit", type=int, help="resume shard limit; never exceeds 600")
    parser.add_argument("--no-synthetic-ir", action="store_true")
    args = parser.parse_args()
    render(
        args.clean_dir,
        args.noise_dir,
        args.out,
        rir_dir=args.rir_dir,
        limit=args.limit or args.initial,
        synthetic_ir=not args.no_synthetic_ir,
    )


if __name__ == "__main__":
    main()
