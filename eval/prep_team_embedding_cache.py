"""Build an offline ECAPA crop cache and speaker-margin report.

The command never downloads weights.  Prepare the ECAPA snapshot separately with
``hearing.tse.enrolled.download_model()`` and point ``HEARING_MODELS_DIR`` at it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16_000
CROP_SECONDS = (0.25, 0.5, 1.0, 2.0)
SPEAKERS = ("aryan", "joseph")


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not vector.size or not np.isfinite(vector).all() or norm <= 1e-12:
        raise ValueError("embedding must be finite and non-zero")
    return vector / norm


def _audio(path: Path, sample_rate: int) -> np.ndarray:
    samples, rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim != 1:
        samples = samples.mean(axis=1)
    if rate != sample_rate:
        from scipy.signal import resample_poly

        gcd = np.gcd(rate, sample_rate)
        samples = resample_poly(samples, sample_rate // gcd, rate // gcd).astype(np.float32)
    return np.asarray(samples, dtype=np.float32)


def _paths(root: Path, speaker: str) -> list[Path]:
    paths = sorted(p for p in (root / speaker).rglob("*") if p.suffix.lower() in {".wav", ".flac"})
    if not paths:
        raise FileNotFoundError(f"no audio files found for {speaker} under {root / speaker}")
    return paths


def build_cache(
    root: str | Path,
    *,
    embedder,
    sample_rate: int = SR,
    crop_seconds: tuple[float, ...] = CROP_SECONDS,
) -> dict[str, np.ndarray | list[str] | int]:
    """Embed non-overlapping crops from each teammate's recordings.

    ``embeddings`` is padded with NaNs to ``[speaker, crop, max_windows, dim]``;
    ``counts`` identifies valid windows for each speaker/crop.
    """
    root = Path(root)
    grouped = []
    paths = []
    for speaker in SPEAKERS:
        speaker_audio = []
        for path in _paths(root, speaker):
            speaker_audio.append(_audio(path, sample_rate))
            paths.append(str(path))
        grouped.append(speaker_audio)
    all_embeds: list[list[list[np.ndarray]]] = []
    counts = np.zeros((len(SPEAKERS), len(crop_seconds)), dtype=np.int32)
    dim = None
    for si, recordings in enumerate(grouped):
        speaker_crops = []
        for ci, seconds in enumerate(crop_seconds):
            n = round(seconds * sample_rate)
            embeddings = []
            for recording in recordings:
                for start in range(0, len(recording) - n + 1, n):
                    embeddings.append(_unit(embedder.embed(recording[start : start + n])))
            if not embeddings:
                raise ValueError(f"{SPEAKERS[si]} has no complete {seconds:g}s crops")
            dim = dim or len(embeddings[0])
            if any(len(item) != dim for item in embeddings):
                raise ValueError("embedding dimension changed between crops")
            counts[si, ci] = len(embeddings)
            speaker_crops.append(embeddings)
        all_embeds.append(speaker_crops)
    max_windows = int(counts.max())
    result = np.full((len(SPEAKERS), len(crop_seconds), max_windows, dim), np.nan, dtype=np.float32)
    for si, speaker_crops in enumerate(all_embeds):
        for ci, embeddings in enumerate(speaker_crops):
            result[si, ci, : len(embeddings)] = embeddings
    return {
        "embeddings": result,
        "counts": counts,
        "crop_seconds": np.asarray(crop_seconds, dtype=np.float32),
        "speakers": np.asarray(SPEAKERS),
        "sample_rate": sample_rate,
        "source_paths": np.asarray(paths),
    }


def _valid(embeddings: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(embeddings[:count], dtype=np.float64)


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {f"p{p:02d}": float(np.percentile(values, p)) for p in (5, 25, 50, 75, 95)}


def margin_report(cache: dict) -> dict:
    """Summarize same-speaker, between-speaker, and closed-set margin cosines."""
    embeddings = np.asarray(cache["embeddings"])
    counts = np.asarray(cache["counts"])
    crops = np.asarray(cache["crop_seconds"])
    output = []
    for ci, crop_s in enumerate(crops):
        speakers = [_valid(embeddings[si, ci], int(counts[si, ci])) for si in range(2)]
        within = []
        for values in speakers:
            within.extend((values @ values.T)[np.triu_indices(len(values), 1)])
        between = (speakers[0] @ speakers[1].T).reshape(-1)
        # A margin compares each crop with its speaker's other crops and strongest negative.
        margins = []
        for si, own in enumerate(speakers):
            other = speakers[1 - si]
            own_scores = own @ own.T
            np.fill_diagonal(own_scores, np.nan)
            same = np.nanmean(own_scores, axis=1)
            same = np.nan_to_num(same, nan=1.0)
            margins.extend(same - np.max(own @ other.T, axis=1))
        margins = np.asarray(margins)
        output.append(
            {
                "crop_s": float(crop_s),
                "windows": [int(x) for x in counts[:, ci]],
                "similarity": {
                    "within": _percentiles(np.asarray(within) if within else np.array([1.0])),
                    "between": _percentiles(between),
                    "margin": _percentiles(margins),
                },
                "margin": _percentiles(margins),
            }
        )
    return {
        "sample_rate": int(cache["sample_rate"]),
        "speakers": [str(x) for x in cache["speakers"]],
        "crops": output,
    }


def _ecapa():
    from hearing.tse.enrolled import _load_encoder

    return _load_encoder()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("eval/fixtures/generated/team"))
    parser.add_argument("--output", type=Path, default=Path("eval/data/team_crop_embeddings.npz"))
    parser.add_argument("--margins", type=Path, default=Path("docs/research/TEAM_CROP_MARGINS.json"))
    args = parser.parse_args(argv)
    cache = build_cache(args.input, embedder=_ecapa())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **cache)
    args.margins.parent.mkdir(parents=True, exist_ok=True)
    args.margins.write_text(json.dumps(margin_report(cache), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
