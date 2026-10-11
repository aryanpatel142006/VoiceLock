"""Render deterministic two-speaker overlap stress shards.

Audio output is intentionally gitignored.  The JSON manifests are the durable
experiment record: they contain source references, rendered-stem hashes, exact
sample-aligned intervals, and measured ratios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
DEFAULT_RATIOS_DB = (0.0, -5.0, -10.0, -15.0)
PEAK = 0.9


def detect_overlaps(intervals: dict[str, Iterable[tuple[float, float]]]) -> list[dict]:
    """Return pairwise intersections, with deterministic speaker ordering."""
    spans: list[tuple[str, float, float]] = []
    for speaker in sorted(intervals):
        for start, end in intervals[speaker]:
            if end < start:
                raise ValueError(f"interval end precedes start for {speaker}: {(start, end)}")
            if end > start:
                spans.append((speaker, float(start), float(end)))
    overlaps: list[dict] = []
    for index, (left, left_start, left_end) in enumerate(spans):
        for right, right_start, right_end in spans[index + 1 :]:
            if left == right:
                continue
            start, end = max(left_start, right_start), min(left_end, right_end)
            if end > start:
                first, second = sorted((left, right))
                overlaps.append(
                    {"speakers": [first, second], "start_s": start, "end_s": end, "duration_s": end - start}
                )
    return sorted(overlaps, key=lambda item: (item["start_s"], item["end_s"], item["speakers"]))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0


def _read_source(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz; expected {SAMPLE_RATE} Hz")
    if audio.ndim != 1:
        raise ValueError(f"{path} is not mono")
    return audio


def render_overlap_ratio(
    output_dir: str | Path,
    *,
    ratio_db: float,
    target_path: str | Path,
    interferer_path: str | Path,
    seed: int = 0,
    duration_s: float = 4.0,
    target_interval: tuple[float, float] = (0.25, 2.75),
    interferer_interval: tuple[float, float] = (1.25, 3.75),
) -> dict:
    """Render one ratio shard and return its JSON-serialisable manifest."""
    if duration_s <= 0 or not math.isfinite(ratio_db) or not -100 <= ratio_db <= 100:
        raise ValueError("duration_s must be positive and ratio_db must be a finite dB value")
    target_path, interferer_path = Path(target_path), Path(interferer_path)
    target = _read_source(target_path)
    interferer = _read_source(interferer_path)
    n_samples = round(duration_s * SAMPLE_RATE)
    target_start, target_end = (round(v * SAMPLE_RATE) for v in target_interval)
    interferer_start, interferer_end = (round(v * SAMPLE_RATE) for v in interferer_interval)
    if not (0 <= target_start < target_end <= n_samples and 0 <= interferer_start < interferer_end <= n_samples):
        raise ValueError("speaker intervals must be inside the rendered duration")
    target_len, interferer_len = target_end - target_start, interferer_end - interferer_start
    target = np.resize(target, target_len).astype(np.float64)
    interferer = np.resize(interferer, interferer_len).astype(np.float64)
    overlap_start, overlap_end = max(target_start, interferer_start), min(target_end, interferer_end)
    target_overlap = target[overlap_start - target_start : overlap_end - target_start]
    interferer_overlap = interferer[overlap_start - interferer_start : overlap_end - interferer_start]
    target_rms, interferer_rms = _rms(target_overlap), _rms(interferer_overlap)
    if target_rms == 0 or interferer_rms == 0:
        raise ValueError("source overlap is silent")
    interferer *= target_rms / interferer_rms / (10.0 ** (ratio_db / 20.0))
    target_stem = np.zeros(n_samples, dtype=np.float64)
    interferer_stem = np.zeros(n_samples, dtype=np.float64)
    target_stem[target_start:target_end] = target
    interferer_stem[interferer_start:interferer_end] = interferer
    mix = target_stem + interferer_stem
    gain = PEAK / max(float(np.max(np.abs(mix))), 1e-12)
    target_stem, interferer_stem, mix = target_stem * gain, interferer_stem * gain, mix * gain
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"target": "target.wav", "interferer": "interferer.wav", "mix": "mix.wav"}
    for name, audio in (("target", target_stem), ("interferer", interferer_stem), ("mix", mix)):
        sf.write(output_dir / paths[name], audio.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    intervals = {
        "target": [[target_start / SAMPLE_RATE, target_end / SAMPLE_RATE]],
        "interferer": [[interferer_start / SAMPLE_RATE, interferer_end / SAMPLE_RATE]],
    }
    measured = 20.0 * np.log10(_rms(target_stem[overlap_start:overlap_end]) / _rms(interferer_stem[overlap_start:overlap_end]))
    manifest = {
        "schema": "overlap-ratio-stress/v1",
        "id": output_dir.name,
        "seed": seed,
        "sample_rate": SAMPLE_RATE,
        "duration_s": n_samples / SAMPLE_RATE,
        "ratio_db": float(ratio_db),
        "measured_overlap_ratio_db": float(measured),
        "speaker_intervals": intervals,
        "overlap_intervals": detect_overlaps(intervals),
        "source_stems": {
            "target": {"path": str(target_path), "sha256": _sha256(target_path)},
            "interferer": {"path": str(interferer_path), "sha256": _sha256(interferer_path)},
        },
        "paths": paths,
        "sha256": {name: _sha256(output_dir / path) for name, path in paths.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def render_set(output_root: str | Path, *, source_root: str | Path, seed: int = 0) -> dict:
    """Render the four requested ratios and write a set index."""
    output_root, source_root = Path(output_root), Path(source_root)
    manifests = []
    for ratio_db in DEFAULT_RATIOS_DB:
        shard = output_root / f"ratio_{ratio_db:g}db"
        manifests.append(
            render_overlap_ratio(
                shard,
                ratio_db=ratio_db,
                target_path=source_root / "speech_a.wav",
                interferer_path=source_root / "speech_b.wav",
                seed=seed,
            )
        )
    index = {"schema": "overlap-ratio-stress/index-v1", "seed": seed, "ratios_db": list(DEFAULT_RATIOS_DB), "shards": manifests}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("eval/shards/overlap_ratios"))
    parser.add_argument("--sources", type=Path, default=Path("eval/fixtures/tiny"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    render_set(args.out, source_root=args.sources, seed=args.seed)


if __name__ == "__main__":
    main()
