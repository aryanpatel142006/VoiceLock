"""Render a deterministic, held-out compound challenge bank.

This is preparation only: it never trains a model and never changes the source bank.  A
source index is treated as frozen and every output records the source hash and exact
transformation recipe.  The default output is under ``HEARING_FIXTURES_DIR`` (usually the
shared ignored generated-fixtures directory).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from eval.fixtures.mix import (
    Manifest,
    Stems,
    fit_length,
    gain_for_level,
    rms,
    write_fixture,
)
from hearing.audio import io as aio

SAMPLE_RATE = aio.PIPELINE_SR
DEFAULT_CAP = 600
HARD_TARGET_DB = (-10.0, -15.0)
TRANSFORMS = ("pause_onset", "noise_reverb_pitch_bandlimit")
STEM_FILES = {"mix": "mix.wav", "target": "target.wav", "interferer": "interferer.wav", "noise": "noise.wav"}


def source_group(record: dict[str, Any]) -> str:
    """Stable group key; role swaps and conditions never cross source groups."""
    source = record.get("source", {})
    values = (
        source.get("dataset", ""),
        source.get("split", ""),
        source.get("target_speaker", ""),
        source.get("interferer_speaker", ""),
        record.get("target_id", ""),
        record.get("interferer_id", ""),
    )
    return hashlib.sha256("\x1f".join(map(str, values)).encode()).hexdigest()[:16]


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return fit_length(np.asarray(x, dtype=np.float32), n).astype(np.float32, copy=True)


def _pitch(x: np.ndarray, semitones: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Pitch challenge approximation: resample there and back, preserving clip duration."""
    if not semitones:
        return x.copy()
    up, down = (2, 1) if semitones > 0 else (1, 2)
    shifted = resample_poly(x, up, down)
    return _fit(resample_poly(shifted, down, up), len(x))


def _reverb(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    taps = max(1, int(0.035 * sr))
    impulse = np.zeros(taps, dtype=np.float32)
    impulse[0] = 1.0
    impulse[taps // 3] = 0.35
    impulse[-1] = 0.15
    return np.convolve(x, impulse, mode="full")[: len(x)].astype(np.float32)


def _bandlimit(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    sos = butter(6, [250.0, 3600.0], btype="bandpass", fs=sr, output="sos")
    if len(x) < 32:
        return x.copy()
    return sosfiltfilt(sos, x).astype(np.float32)


def _pause_onset(x: np.ndarray, role: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = x.copy()
    onset = min(int((0.11 if role == "target" else 0.0) * sr), max(0, len(y) - 1))
    if onset:
        y = np.concatenate((np.zeros(onset, dtype=np.float32), y[:-onset]))
    gap_start = int(0.43 * len(y))
    gap = int(0.12 * sr)
    y[gap_start : min(len(y), gap_start + gap)] = 0.0
    return y


def transform_audio(x: np.ndarray, recipe: str, role: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Apply one named deterministic compound transform without mutating ``x``."""
    y = np.asarray(x, dtype=np.float32).copy()
    if recipe == "pause_onset":
        return _pause_onset(y, role, sr)
    if recipe != "noise_reverb_pitch_bandlimit":
        raise ValueError(f"unknown transform recipe: {recipe}")
    y = _pitch(y, 2 if role == "target" else -2, sr)
    y = _reverb(y, sr)
    y = _bandlimit(y, sr)
    return _fit(y, len(x))


def _pink(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freq = np.fft.rfftfreq(n)
    spec[1:] *= 1.0 / np.sqrt(np.maximum(freq[1:], 1e-12))
    out = np.fft.irfft(spec, n=n).astype(np.float32)
    level = rms(out)
    return out / level if level else out


def _scale_noise(target: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if rms(noise) == 0:
        return np.zeros_like(target)
    return noise * gain_for_level(target, noise, snr_db)


def render_compound(
    record: dict[str, Any],
    source_dir: Path,
    out_root: Path,
    recipe: str,
    hard_target_db: float,
    role_swap: bool,
    seed: int,
) -> dict[str, Any]:
    """Render one case and return its manifest; source files are read-only."""
    fixture_dir = source_dir / record["id"]
    arrays: dict[str, np.ndarray] = {}
    for name in ("target", "interferer", "noise"):
        relative = Path(record["paths"][name])
        path = source_dir / relative if relative.parts[0] == record["id"] else fixture_dir / relative
        arrays[name], _ = aio.load(path, SAMPLE_RATE)
    n = len(arrays["target"])
    target_name, interferer_name = ("interferer", "target") if role_swap else ("target", "interferer")
    target = transform_audio(_fit(arrays[target_name], n), recipe, "target")
    interferer = transform_audio(_fit(arrays[interferer_name], n), recipe, "interferer")
    interferer = interferer * gain_for_level(target, interferer, hard_target_db)
    noise = _scale_noise(target, _fit(arrays["noise"], n), float(record.get("snr_db", 0.0)))
    if recipe == "noise_reverb_pitch_bandlimit":
        noise = noise + (0.15 * rms(target) * _pink(n, seed + 991)).astype(np.float32)
    mix = target.astype(np.float64) + interferer.astype(np.float64) + noise.astype(np.float64)
    gain = 0.9 / max(float(np.max(np.abs(mix))), 1e-12)
    target, interferer, noise = (v * gain for v in (target, interferer, noise))
    stems = Stems(
        mix=(target + interferer + noise).astype(np.float32),
        target=target.astype(np.float32),
        interferer=interferer.astype(np.float32),
        noise=noise.astype(np.float32),
    )
    suffix = f"{recipe}_sir{int(abs(hard_target_db))}_{'swap' if role_swap else 'base'}"
    case_id = f"{record['id']}_{suffix}"
    challenge_key = hashlib.sha256(
        json.dumps(
            {
                "source_id": record["id"],
                "source_group": source_group(record),
                "recipe": recipe,
                "hard_target_db": hard_target_db,
                "role_swap": role_swap,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    manifest = Manifest(
        id=case_id,
        set="compound_heldout",
        sample_rate=SAMPLE_RATE,
        duration_s=round(n / SAMPLE_RATE, 4),
        snr_db=float(record.get("snr_db", 0.0)),
        sir_db=float(hard_target_db),
        target_id=record.get("interferer_id", "") if role_swap else record.get("target_id", ""),
        interferer_id=record.get("target_id", "") if role_swap else record.get("interferer_id", ""),
        noise_type="compound_pink",
        transcript=(record.get("interferer_transcript", "") if role_swap else record.get("transcript", "")),
        paths=dict(STEM_FILES),
        seed=seed,
        source={
            "dataset": record.get("source", {}).get("dataset", ""),
            "split": record.get("source", {}).get("split", ""),
            "source_id": record["id"],
            "source_group": source_group(record),
            "role_swap": role_swap,
            "transform": recipe,
            "hard_target_db": hard_target_db,
            "transcript_available": bool(
                record.get("interferer_transcript" if role_swap else "transcript", "")
            ),
            "challenge_key": challenge_key,
        },
        interferer_transcript=(record.get("transcript", "") if role_swap else record.get("interferer_transcript", "")),
    )
    write_fixture(out_root, manifest, stems)
    return manifest.to_dict()


def build_plan(index: dict[str, Any], cap: int = DEFAULT_CAP) -> list[dict[str, Any]]:
    records = sorted(index.get("fixtures", []), key=lambda r: (source_group(r), r["id"]))
    if cap < 1:
        raise ValueError("cap must be positive")
    plan: list[dict[str, Any]] = []
    # Four cases per source record gives balanced role swaps and both hard levels.
    for record in records:
        for role_swap in (False, True):
            for hard in HARD_TARGET_DB:
                for recipe in TRANSFORMS:
                    plan.append(
                        {
                            "source_id": record["id"],
                            "source_group": source_group(record),
                            "role_swap": role_swap,
                            "hard_target_db": hard,
                            "transform": recipe,
                        }
                    )
    # Round-robin source groups prevents a cap from favouring the first source group.
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in plan:
        groups.setdefault(item["source_group"], []).append(item)
    balanced: list[dict[str, Any]] = []
    while groups and len(balanced) < cap:
        for key in sorted(groups):
            if groups[key]:
                balanced.append(groups[key].pop(0))
                if len(balanced) == cap:
                    break
            if not groups[key]:
                del groups[key]
    return balanced


def _checkpoint(path: Path, plan_hash: str, completed: list[str]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"plan_sha256": plan_hash, "completed": completed}, indent=2) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-set", default="tiny")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--cap", "--max-cases", type=int, default=DEFAULT_CAP)
    parser.add_argument("--seed", type=int, default=261)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, default=None)
    args = parser.parse_args(argv)
    source_dir = args.source_root / args.source_set
    index = json.loads((source_dir / "index.json").read_text())
    plan = build_plan(index, args.cap)
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    out = args.out or Path(os.environ.get("HEARING_FIXTURES_DIR", "eval/fixtures/generated")) / "compound_heldout"
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or out / "checkpoint.json"
    completed: set[str] = set()
    if args.resume and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        if state.get("plan_sha256") != plan_hash:
            raise ValueError("checkpoint plan does not match current source index/cap")
        completed = set(state.get("completed", []))
    pid_path = out / "render.pid"
    log_path = out / "render.log"
    out.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()) + "\n")
    try:
        with log_path.open("a") as log:
            log.write(f"start {time.time():.3f} plan_sha256={plan_hash}\n")
            for number, item in enumerate(plan, start=1):
                case_key = f"{item['source_id']}|{item['transform']}|{item['hard_target_db']}|{item['role_swap']}"
                if case_key in completed:
                    continue
                if args.stop_file and args.stop_file.exists():
                    log.write(f"stopped at {number}\n")
                    break
                record = next(r for r in index["fixtures"] if r["id"] == item["source_id"])
                rendered = render_compound(
                    record, source_dir, out, item["transform"], item["hard_target_db"], item["role_swap"], args.seed
                )
                completed.add(case_key)
                _checkpoint(checkpoint, plan_hash, sorted(completed))
                print(f"\r[{number}/{len(plan)}] {rendered['id']}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
            entries = []
            for manifest_path in sorted(out.glob("*/manifest.json")):
                entries.append(json.loads(manifest_path.read_text()))
            (out / "index.json").write_text(
                json.dumps(
                    {
                        "set": "compound_heldout",
                        "sample_rate": SAMPLE_RATE,
                        "count": len(entries),
                        "plan_sha256": plan_hash,
                        "fixtures": entries,
                    },
                    indent=2,
                )
                + "\n"
            )
    finally:
        pid_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
