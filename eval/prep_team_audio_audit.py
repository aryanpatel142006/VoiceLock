"""Measure local Aryan/Joseph recordings and write an acoustic health catalog."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".aif", ".aiff", ".m4a", ".mp3"}
DEFAULT_ROOTS = (
    "profiles/aryan",
    "profiles/joseph",
    "eval/fixtures/generated/team/aryan",
    "eval/fixtures/generated/team/joseph",
    "eval/fixtures/demo_listen",
)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return samples.mean(axis=1), int(sample_rate)
    except (OSError, RuntimeError, ValueError):
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=sample_rate,channels",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        sample_rate = int(stream["sample_rate"])
        channels = int(stream["channels"])
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(path),
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        )
        values = np.frombuffer(decoded.stdout, dtype="<f4")
        return values.reshape(-1, channels).mean(axis=1), sample_rate


def _metrics(samples: np.ndarray, sample_rate: int) -> dict[str, float]:
    samples = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    frame_length = max(1, round(sample_rate * 0.01))
    count = max(1, samples.size // frame_length)
    frames = np.array_split(samples[: count * frame_length], count)
    frame_rms = np.array(
        [np.sqrt(np.mean(np.square(frame))) for frame in frames], dtype=np.float64
    )
    noise_rms = float(np.percentile(frame_rms, 10)) if frame_rms.size else 0.0
    threshold = max(noise_rms * 3.0, peak * 0.05, 1e-4)
    speech_fraction = float(np.mean(frame_rms >= threshold)) if frame_rms.size else 0.0
    snr_db = float(20.0 * np.log10(max(rms, 1e-12) / max(noise_rms, 1e-12)))
    return {
        "peak_amplitude": peak,
        "rms_energy": rms,
        "noise_floor_rms": noise_rms,
        "snr_db": snr_db,
        "speech_fraction": speech_fraction,
    }


def _speaker(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    for name in ("aryan", "joseph"):
        if name in parts:
            return name
    return "shared"


def audit_paths(roots: Iterable[str | Path]) -> list[dict[str, object]]:
    candidates: set[Path] = set()
    for root_value in roots:
        root = Path(root_value)
        if root.is_file() and root.suffix.lower() in AUDIO_SUFFIXES:
            candidates.add(root)
        elif root.is_dir():
            candidates.update(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
            )
    records = []
    for path in sorted(candidates):
        samples, sample_rate = _read_audio(path)
        records.append(
            {
                "path": str(path),
                "speaker": _speaker(path),
                "duration_s": float(samples.size / sample_rate),
                "sample_rate": sample_rate,
                "channels": 1,
                **_metrics(samples, sample_rate),
            }
        )
    return records


def write_markdown(records: list[dict[str, object]], destination: str | Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Team audio audit",
        "",
        "Generated from local recordings; speech is estimated from 10 ms frame energy.",
        "",
        "| speaker | recording | duration (s) | sample rate | peak | RMS | SNR vs silence (dB) | speech fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {speaker} | `{name}` | {duration_s:.3f} | {sample_rate} | "
            "{peak_amplitude:.4f} | {rms_energy:.4f} | {snr_db:.2f} | "
            "{speech_fraction:.3f} |".format(
                name=Path(str(record["path"])).name, **record
            )
        )
    lines.extend(
        [
            "",
            (
                "SNR uses the ratio of whole-recording RMS to the 10th-percentile 10 ms "
                "frame RMS. A floor of 1e-12 avoids undefined values for digitally silent "
                "recordings; speech fraction counts frames above max(3x noise floor, 5% "
                "of peak, 1e-4)."
            ),
            "",
        ]
    )
    destination.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", dest="roots", help="Recording root")
    parser.add_argument("--json", default="eval/data/team_audio_audit.json")
    parser.add_argument("--markdown", default="docs/research/TEAM_AUDIO_AUDIT.md")
    args = parser.parse_args()
    roots = args.roots or list(DEFAULT_ROOTS)
    records = audit_paths(roots)
    payload = {"recordings": records, "recording_count": len(records), "roots": roots}
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(records, args.markdown)
    print(json.dumps({"recording_count": len(records), "json": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
