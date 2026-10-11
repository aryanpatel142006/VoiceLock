"""Build a deterministic, metadata-first overnight speech-pair catalog.

This module never downloads, mixes, transcribes, or infers speaker attributes.  It
only reads public-source manifests and existing WAVs, and emits plans for a later
evaluation job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

SIR_GRID_DB = [5, 0, -5, -10, -15]
SOURCE_OVERLAP_EXCLUDED = True


@dataclass(frozen=True)
class SourceClip:
    source_id: str
    speaker_id: str
    documented_sex: str
    transcript: str
    path: Path
    provenance: dict


def _pitch_hz(x: np.ndarray, sr: int) -> float | None:
    """Return a robust median autocorrelation pitch, or None when unvoiced."""
    frame = max(1, int(sr * 0.04))
    hop = max(1, int(sr * 0.02))
    values: list[float] = []
    for start in range(0, max(0, len(x) - frame + 1), hop):
        y = x[start : start + frame].astype(np.float64)
        y -= y.mean()
        energy = float(np.dot(y, y))
        if energy < 1e-8:
            continue
        lo, hi = int(sr / 400), int(sr / 60)
        corr = np.correlate(y, y, mode="full")[len(y) - 1 :]
        hi = min(hi, len(corr) - 1)
        if hi <= lo:
            continue
        lag = lo + int(np.argmax(corr[lo : hi + 1]))
        strength = float(corr[lag] / corr[0])
        if strength >= 0.35:
            values.append(sr / lag)
    return float(np.median(values)) if values else None


def source_stats(path: str | Path) -> dict:
    """Compute lightweight acoustic facts without identifying or classifying a voice."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return {
        "sample_rate": int(sr),
        "duration_s": round(len(x) / sr, 4),
        "frames": len(x),
        "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 3),
        "median_pitch_hz": None
        if (pitch := _pitch_hz(x, int(sr))) is None
        else round(pitch, 2),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }


def _group(a: SourceClip, b: SourceClip) -> str:
    if not a.documented_sex or not b.documented_sex:
        return "unknown_documented_sex"
    return (
        "same_documented_sex"
        if a.documented_sex == b.documented_sex
        else "different_documented_sex"
    )


def build_pair_plan(clips: list[SourceClip], max_pairs: int = 60) -> list[dict]:
    """Select distinct-speaker pairs, preferring close robust median pitch."""
    candidates = []
    for a in clips:
        for b in clips:
            if a.source_id >= b.source_id or a.speaker_id == b.speaker_id:
                continue
            sa, sb = source_stats(a.path), source_stats(b.path)
            pa, pb = sa["median_pitch_hz"], sb["median_pitch_hz"]
            pitch_gap = abs(pa - pb) if pa is not None and pb is not None else float("inf")
            candidates.append((pitch_gap, a.source_id, b.source_id, a, b, sa, sb))
    candidates.sort(key=lambda row: row[:3])
    result = []
    for _, _, _, target, interferer, target_stats, interferer_stats in candidates[:max_pairs]:
        result.append(
            {
                "pair_id": f"{target.source_id}__{interferer.source_id}",
                "target": {
                    "source_id": target.source_id,
                    "speaker_id": target.speaker_id,
                    "role": "target",
                    "path": str(target.path),
                    "transcript": target.transcript,
                    "stats": target_stats,
                    "provenance": target.provenance,
                },
                "interferer": {
                    "source_id": interferer.source_id,
                    "speaker_id": interferer.speaker_id,
                    "role": "interferer",
                    "path": str(interferer.path),
                    "transcript": interferer.transcript,
                    "stats": interferer_stats,
                    "provenance": interferer.provenance,
                },
                "speaker_group": _group(target, interferer),
                "sir_db": SIR_GRID_DB,
                "proposed_onset_s": 0.0,
                "proposed_crop_s": 3.0,
                "metadata_confidence": "documented" if target.provenance.get("dataset") else "unknown",
                "pitch_match_status": "acoustic_candidate",
                "source_overlap_excluded": SOURCE_OVERLAP_EXCLUDED,
            }
        )
    return result


def target_absent_plans() -> list[dict]:
    """Return 12 interval plans; timing is deliberately provenance-only."""
    specs = [
        ("pause", "target pauses before interferer"),
        ("pause", "target pauses mid-utterance"),
        ("pause", "target pauses after utterance"),
        ("pause", "target absent during lead-in"),
        ("target_absent", "target absent, one talker enters"),
        ("target_absent", "target absent, two talkers alternate"),
        ("target_absent", "target absent, three talkers alternate"),
        ("target_absent", "target leaves and returns"),
        ("two_vs_three_talker", "two talkers active"),
        ("two_vs_three_talker", "three talkers active"),
        ("two_vs_three_talker", "two then three talkers"),
        ("two_vs_three_talker", "three then two talkers"),
    ]
    plans = []
    for index, (condition, note) in enumerate(specs, 1):
        plans.append(
            {
                "plan_id": f"absent-{index:02d}",
                "condition": condition,
                "note": note,
                "intervals": [{"start_s": None, "end_s": None, "status": "proposed"}],
                "interval_provenance": "not collected: requires annotated multi-talker recording",
                "target_words_status": "unverified",
                "source_overlap_excluded": True,
            }
        )
    return plans


def write_plan(clips: list[SourceClip], output: str | Path) -> dict:
    plan = {
        "schema": "overnight-grunt-pairs/v1",
        "sample_rate": 16000,
        "source_overlap_excluded": True,
        "pairs": build_pair_plan(clips),
        "target_absent_plans": target_absent_plans(),
    }
    Path(output).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _clips_from_manifest(path: Path) -> list[SourceClip]:
    raw = json.loads(path.read_text())
    root = path.parent
    out = []
    for entry in raw.get("speech", []):
        out.append(
            SourceClip(
                source_id=str(entry["id"]),
                speaker_id=str(entry["speaker_id"]),
                documented_sex=str(entry.get("sex", "")),
                transcript=str(entry.get("text", "")).strip(),
                path=root / str(entry["file"]),
                provenance={
                    "dataset": raw.get("dataset", ""),
                    "split": raw.get("split", ""),
                    "license": raw.get("license", ""),
                    "manifest": str(path),
                },
            )
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clips = [clip for manifest in args.manifest for clip in _clips_from_manifest(manifest)]
    write_plan(clips, args.output)


if __name__ == "__main__":
    main()
