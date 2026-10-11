"""Render deterministic offline gain-disparity and target-pause audio shards.

The renderer intentionally preserves over-range float samples until WAV serialization.  It is
preparation data, not a production mixer; each shard records its recipe and the serialized hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hearing.audio import io as audio_io

SR = audio_io.PIPELINE_SR
SIRS_DB = (0.0, -6.0, -10.0, -15.0)
SCHEDULES = ("target_on", "target_pause", "target_off")
ONSET_SHIFTS_S = (0.0, 0.05, 0.25)


@dataclass(frozen=True)
class Recipe:
    target_role: str
    interferer_role: str
    sir_db: float
    schedule: str
    onset_shift_s: float
    seed: int = 0

    @property
    def id(self) -> str:
        return (
            f"{self.target_role}_to_{self.interferer_role}"
            f"_sir{self.sir_db:g}_{self.schedule}_shift{self.onset_shift_s:g}"
        )


def _rms(audio: np.ndarray) -> float:
    values = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def _shift(audio: np.ndarray, samples: int) -> np.ndarray:
    output = np.zeros_like(audio)
    if samples >= 0:
        output[samples:] = audio[: len(audio) - samples] if samples < len(audio) else 0
    elif -samples < len(audio):
        output[: len(audio) + samples] = audio[-samples:]
    return output


def render_recipe(
    target: np.ndarray, interferer: np.ndarray, recipe: Recipe, sample_rate: int = SR
) -> tuple[np.ndarray, dict]:
    """Return an unnormalised float32 mix and metadata for one deterministic recipe."""
    target = np.asarray(target, dtype=np.float32)
    interferer = np.asarray(interferer, dtype=np.float32)
    if not target.size or not interferer.size:
        raise ValueError("target and interferer must be non-empty")
    target_rms, interferer_rms = _rms(target), _rms(interferer)
    if target_rms == 0.0 or interferer_rms == 0.0:
        raise ValueError("target and interferer must be non-silent")
    if recipe.schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {recipe.schedule!r}")
    if recipe.onset_shift_s < 0:
        raise ValueError("onset_shift_s must be non-negative")

    n = max(len(target), len(interferer))
    target = np.pad(target, (0, n - len(target)))
    interferer = np.pad(interferer, (0, n - len(interferer)))
    interferer = interferer * (target_rms / interferer_rms) * 10 ** (-recipe.sir_db / 20)
    interferer = _shift(interferer, round(recipe.onset_shift_s * sample_rate))
    target_silent: list[list[float]] = []
    if recipe.schedule == "target_off":
        target[:] = 0
        target_silent = [[0.0, round(n / sample_rate, 6)]]
    elif recipe.schedule == "target_pause":
        start = min(round(recipe.onset_shift_s * sample_rate), n)
        target[start:] = 0
        target_silent = [[round(start / sample_rate, 6), round(n / sample_rate, 6)]]
    mix = (target.astype(np.float64) + interferer.astype(np.float64)).astype(np.float32)
    metadata = {
        "target_role": recipe.target_role,
        "interferer_role": recipe.interferer_role,
        "sir_db": recipe.sir_db,
        "schedule": recipe.schedule,
        "onset_shift_s": recipe.onset_shift_s,
        "seed": recipe.seed,
        "sample_rate": sample_rate,
        "duration_s": round(len(mix) / sample_rate, 6),
        "target_silent_intervals": target_silent,
        "peak": float(np.max(np.abs(mix), initial=0.0)),
        "declared_overrange": bool(np.max(np.abs(mix), initial=0.0) > 1.0),
    }
    return mix, metadata


def recipes(
    roles: tuple[str, str] = ("aryan", "joseph"),
    seed: int = 0,
    sirs: tuple[float, ...] = SIRS_DB,
    shifts: tuple[float, ...] = ONSET_SHIFTS_S,
) -> list[Recipe]:
    """Build the stable recipe order, rendering each role as target in turn."""
    result = []
    for target, interferer in (roles, roles[::-1]):
        for sir in sirs:
            for schedule in SCHEDULES:
                for shift in shifts:
                    result.append(Recipe(target, interferer, sir, schedule, shift, seed))
    return result


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.wav")
    audio_io.save(temporary, audio, sample_rate)
    temporary.replace(path)


def write_shards(
    sources: dict[str, np.ndarray],
    render_recipes: list[Recipe],
    out: Path,
    sample_rate: int = SR,
) -> dict:
    """Write WAV shards and a sorted aggregate manifest; return the manifest value."""
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for recipe in render_recipes:
        try:
            target, interferer = sources[recipe.target_role], sources[recipe.interferer_role]
        except KeyError as exc:
            raise ValueError(f"missing source role {exc.args[0]!r}") from exc
        audio, metadata = render_recipe(target, interferer, recipe, sample_rate)
        wav = out / recipe.id / "mix.wav"
        _write_wav(wav, audio, sample_rate)
        metadata.update({"id": recipe.id, "sha256": hashlib.sha256(wav.read_bytes()).hexdigest()})
        (wav.parent / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        files.append(str(wav.relative_to(out)))
    manifest = {"sample_rate": sample_rate, "count": len(files), "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True, help="directory containing aryan.wav and joseph.wav")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)
    sources = {role: audio_io.load(args.sources / f"{role}.wav", SR)[0] for role in ("aryan", "joseph")}
    plan = recipes()
    if args.limit:
        plan = plan[: args.limit]
    write_shards(sources, plan, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
