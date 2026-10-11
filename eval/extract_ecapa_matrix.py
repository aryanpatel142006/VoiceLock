"""Extract frozen ECAPA embeddings and crop-wise cosine-distance matrices.

The input directory contains ``clean/`` and ``corrupted/`` subdirectories with WAV
segments.  This module deliberately does not download weights: prepare the
SpeechBrain checkpoint once with ``hearing.tse.enrolled.download_model`` and then
run this script offline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from hearing.audio.io import PIPELINE_SR, resample

DEFAULT_CROPS = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class Segment:
    condition: str
    speaker: str
    path: Path


def load_segments(root: str | Path) -> list[Segment]:
    """Discover WAVs below ``clean`` and ``corrupted``, in stable path order."""
    root = Path(root)
    segments = []
    for condition in ("clean", "corrupted"):
        directory = root / condition
        for path in sorted(directory.rglob("*.wav")):
            segments.append(Segment(condition, path.stem, path))
    if not segments:
        raise ValueError(f"no WAV segments found below {root}/clean or {root}/corrupted")
    return segments


def _read(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"{path}: expected mono WAV, got {audio.shape}")
    if sample_rate != PIPELINE_SR:
        audio = resample(audio, sample_rate, PIPELINE_SR)
    if not np.isfinite(audio).all():
        raise ValueError(f"{path}: audio contains non-finite samples")
    return np.asarray(audio, dtype=np.float32)


def _windows(audio: np.ndarray, crop_s: float, hop_s: float) -> list[tuple[int, np.ndarray]]:
    length = round(crop_s * PIPELINE_SR)
    hop = round(hop_s * PIPELINE_SR)
    if length <= 0 or hop <= 0:
        raise ValueError("crop and hop durations must be positive")
    return [
        (start, audio[start : start + length])
        for start in range(0, len(audio) - length + 1, hop)
    ]


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("ECAPA returned an invalid embedding")
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("ECAPA returned a zero embedding")
    return vector / norm


def _distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    similarities = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    distances = 1.0 - similarities
    np.fill_diagonal(distances, 0.0)
    return distances


def drift_summary(matrix: np.ndarray, names: list[str]) -> dict:
    """Mean clean-vs-corrupted cosine distance for windows with the same file name and offset.

    Pairing key is ``<file name>@<offset>``; corrupted files keep the clean file's name.
    """
    clean, corrupted = {}, {}
    for i, name in enumerate(names):
        condition, _, key = name.split("/", 2)
        (clean if condition == "clean" else corrupted)[key] = i
    pairs = [(clean[k], corrupted[k]) for k in clean.keys() & corrupted.keys()]
    if not pairs:
        return {"n_pairs": 0, "mean": None, "max": None}
    d = np.array([matrix[i, j] for i, j in pairs])
    return {"n_pairs": len(pairs), "mean": float(d.mean()), "max": float(d.max())}


def _ecapa_embedder() -> Callable[[np.ndarray], np.ndarray]:
    """Return the repository's local-only ECAPA backend."""
    from hearing import models
    from hearing.tse import enrolled

    backend = models.load(enrolled.MODEL)
    return backend.embed


def extract_matrix(
    root: str | Path,
    *,
    embedder: Callable[[np.ndarray], np.ndarray] | None = None,
    crops: tuple[float, ...] = DEFAULT_CROPS,
    hop_s: float | None = None,
    output: str | Path | None = None,
) -> dict:
    """Extract all complete crops and return/write pairwise cosine distances.

    ``output`` is an ``.npz`` containing one ``matrix_<crop>`` and ``labels_<crop>``
    pair per crop, plus a JSON sidecar with reproducibility metadata.
    """
    if not crops or any(c <= 0 for c in crops):
        raise ValueError("crops must contain positive durations")
    segments = load_segments(root)
    embed = embedder or _ecapa_embedder()
    arrays = {segment.path: _read(segment.path) for segment in segments}
    matrices: dict[float, np.ndarray] = {}
    labels: dict[float, list[str]] = {}
    for crop_s in crops:
        hop = hop_s if hop_s is not None else crop_s
        vectors, names = [], []
        for segment in segments:
            for start, audio in _windows(arrays[segment.path], crop_s, hop):
                vectors.append(_unit(embed(audio)))
                names.append(
                    f"{segment.condition}/{segment.speaker}/{segment.path.name}@"
                    f"{start / PIPELINE_SR:.3f}s"
                )
        if not vectors:
            raise ValueError(f"no complete {crop_s:g}s crops found")
        matrix = _distance_matrix(np.stack(vectors))
        matrices[crop_s] = matrix
        labels[crop_s] = names
    result = {
        "sample_rate": PIPELINE_SR,
        "crops": [float(c) for c in crops],
        "hop_s": hop_s,
        "n_segments": len(segments),
        "n_embeddings": {str(float(c)): len(labels[c]) for c in crops},
        "drift": {str(float(c)): drift_summary(matrices[c], labels[c]) for c in crops},
        "matrices": matrices,
        "labels": labels,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for crop_s in crops:
            key = str(float(crop_s))
            payload[f"matrix_{key}"] = matrices[crop_s]
            payload[f"labels_{key}"] = np.asarray(labels[crop_s])
        np.savez_compressed(path, **payload)
        metadata = {k: v for k, v in result.items() if k not in ("matrices", "labels")}
        metadata["output"] = str(path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory containing clean/ and corrupted/")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .npz path")
    parser.add_argument("--hop-s", type=float, default=None, help="crop hop (default: crop duration)")
    args = parser.parse_args()
    result = extract_matrix(args.root, hop_s=args.hop_s, output=args.output)
    print(f"{result['n_segments']} segments; embeddings per crop: {result['n_embeddings']}")
    for crop, d in result["drift"].items():
        print(f"crop {crop}s clean-vs-corrupted drift: {d}")


if __name__ == "__main__":
    main()
