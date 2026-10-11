"""Prepare an offline quiet-target headroom and gain-sweep experiment.

The command reads *clean* target/interferer/noise stems from an existing fixture
bank.  It never changes production settings and writes JSON/WAV derivatives only
under ``$HEARING_RESULTS_DIR/overnight_prep/T-238`` (or ``--out``).

Example::

    .venv/bin/python -m eval.prep_headroom --fixtures "$HEARING_FIXTURES_DIR/tiny" \
        --out "$HEARING_RESULTS_DIR/overnight_prep/T-238" --limit 24
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from hearing.audio import io as aio

TASK_ID = "T-238"
SR = aio.PIPELINE_SR
TARGET_DBFS = -20.0
SIR_SWEEP_DB = (0.0, -6.0, -10.0, -15.0)
MAX_GAIN_DB = 12.0
DEFAULT_LIMIT = 24
LICENCE_EVIDENCE = {
    "dataset": "LibriSpeech ASR corpus, OpenSLR SLR12, CC BY 4.0",
    "evidence": "eval/fixtures/README.md#licences",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def stats(x: np.ndarray) -> dict[str, float | int | bool]:
    """Return measured, unnormalised signal statistics."""
    a = np.asarray(x, dtype=np.float32)
    return {
        "samples": int(a.size),
        "rms": rms(a),
        "rms_dbfs": dbfs(rms(a)),
        "peak": float(np.max(np.abs(a))) if a.size else 0.0,
        "peak_dbfs": dbfs(float(np.max(np.abs(a))) if a.size else 0.0),
        "silent": bool(a.size == 0 or np.max(np.abs(a)) <= 1e-8),
    }


def smooth_gain_db(
    desired_db: np.ndarray,
    voiced: np.ndarray,
    *,
    attack_frames: int = 3,
    release_frames: int = 12,
    max_gain_db: float = MAX_GAIN_DB,
) -> np.ndarray:
    """Causal bounded gain envelope; silence holds zero gain.

    ``desired_db`` and ``voiced`` are 10 ms-frame values. Gain changes toward a
    voiced target at the attack rate and returns toward zero at the release rate.
    """
    desired = np.asarray(desired_db, dtype=np.float64)
    active = np.asarray(voiced, dtype=bool)
    if desired.shape != active.shape:
        raise ValueError("desired_db and voiced must have the same shape")
    if attack_frames < 1 or release_frames < 1 or max_gain_db < 0:
        raise ValueError("smoothing parameters must be positive")
    out = np.zeros(desired.shape, dtype=np.float64)
    for i, want in enumerate(np.clip(desired, -max_gain_db, max_gain_db)):
        if not active[i]:
            out[i] = 0.0
            continue
        target = float(want)
        previous = out[i - 1] if i else 0.0
        step = 1.0 / (attack_frames if target > previous else release_frames)
        if target < previous:
            step = 1.0 / release_frames
        out[i] = previous + np.clip(target - previous, -step * max_gain_db, step * max_gain_db)
    return out.astype(np.float32)


def frame_rms(x: np.ndarray, hop: int = 160) -> np.ndarray:
    padded = np.pad(np.asarray(x, dtype=np.float32), (0, (-len(x)) % hop))
    return np.sqrt(np.mean(padded.reshape(-1, hop).astype(np.float64) ** 2, axis=1))


def gain_forecast(
    target: np.ndarray,
    *,
    target_db: float = TARGET_DBFS,
    silence_db: float = -50.0,
    attack_frames: int = 3,
    release_frames: int = 12,
) -> dict:
    """Forecast target levelling without applying it to the source."""
    levels = frame_rms(target)
    desired = target_db - np.asarray([dbfs(v) for v in levels])
    voiced = np.asarray([dbfs(v) > silence_db for v in levels])
    gain = smooth_gain_db(
        desired, voiced, attack_frames=attack_frames, release_frames=release_frames
    )
    sample_gain = np.repeat(gain, 160)[: len(target)]
    post = np.asarray(target, dtype=np.float64) * 10 ** (sample_gain / 20.0)
    return {
        "target_dbfs": target_db,
        "silence_floor_dbfs": silence_db,
        "frames": len(gain),
        "gain_db_min": float(np.min(gain)) if len(gain) else 0.0,
        "gain_db_max": float(np.max(gain)) if len(gain) else 0.0,
        "gain_db_p50": float(np.percentile(gain, 50)) if len(gain) else 0.0,
        "silent_frames_no_gain": int(np.sum((~voiced) & (gain == 0))),
        "post_gain_stats": {
            "rms_dbfs": dbfs(rms(post)),
            "peak": float(np.max(np.abs(post))) if len(post) else 0.0,
            "peak_dbfs": dbfs(float(np.max(np.abs(post))) if len(post) else 0.0),
        },
        "clipping_forecast": {
            "float_overrange_samples": int(np.sum(np.abs(post) > 1.0)),
            "pcm_clip_samples": int(np.sum(np.abs(post) > 1.0)),
        },
    }


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        temp = Path(f.name)
    temp.replace(path)


def discover(fixtures: Path, limit: int) -> list[Path]:
    if (fixtures / "manifest.json").exists():
        paths = [fixtures]
    else:
        paths = sorted(p.parent for p in fixtures.glob("*/manifest.json"))
    return paths[:limit]


def analyse_case(case_dir: Path, out: Path, render: bool) -> dict:
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    stems = {}
    source_hashes = {}
    paths = manifest.get("paths", {})
    for name in ("target", "interferer", "noise"):
        relative = Path(paths.get(name, f"{name}.wav")).name
        source_path = case_dir / relative
        stems[name], source_sr = aio.load(source_path)
        source_hashes[name] = sha256(source_path)
        if source_sr != SR:
            raise ValueError(f"{case_dir}: expected {SR} Hz, got {source_sr}")
    n = min(map(len, stems.values()))
    stems = {k: v[:n] for k, v in stems.items()}
    target_gain = gain_forecast(stems["target"])
    sir_rows = []
    target_rms = rms(stems["target"])
    for sir in SIR_SWEEP_DB:
        factor = 10 ** (-sir / 20) * target_rms / max(rms(stems["interferer"]), 1e-12)
        competitor = stems["interferer"] * factor
        summed = stems["target"] + competitor + stems["noise"]
        sir_rows.append(
            {
                "sir_db": sir,
                "target_gain_db": TARGET_DBFS - dbfs(target_rms),
                "forecast_peak": float(np.max(np.abs(summed))),
                "float_overrange_samples": int(np.sum(np.abs(summed) > 1.0)),
                "pcm_clip_samples": int(np.sum(np.abs(np.clip(summed, -1, 1)) >= 1.0)),
                "mix_additivity_error_rms": rms(summed - stems["target"] - competitor - stems["noise"]),
            }
        )
    result = {
        "id": manifest.get("id", case_dir.name),
        "source": {
            "manifest_sha256": sha256(manifest_path),
            "stem_sha256": source_hashes,
            "fixture": manifest.get("set", ""),
            "target_id": manifest.get("target_id", ""),
            "interferer_id": manifest.get("interferer_id", ""),
            "target_speaker": manifest.get("source", {}).get("target_speaker", ""),
            "interferer_speaker": manifest.get("source", {}).get("interferer_speaker", ""),
            "utterance": manifest.get("transcript", ""),
            "interval": [0.0, round(n / SR, 6)],
            "licence": LICENCE_EVIDENCE,
        },
        "input_stats": {name: stats(value) for name, value in stems.items()},
        "prediction_vs_measurement": {
            "prediction": "float_overrange and pcm_clip counts forecast unprocessed summed stems; target gain is bounded/smoothed.",
            "measured": "input RMS/peak/silence and additive reconstruction are measured from decoded PCM16 stems.",
        },
        "target_gain_forecast": target_gain,
        "sir_sweep": sir_rows,
    }
    if render:
        wave_dir = out / "waveforms"
        wave_dir.mkdir(parents=True, exist_ok=True)
        for name, value in (
            ("target", stems["target"]),
            ("baseline_sum", stems["target"] + stems["interferer"] + stems["noise"]),
        ):
            aio.save(wave_dir / f"{result['id']}_{name}.wav", value, SR)
        result["diagnostic_waveforms"] = "waveforms/"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= DEFAULT_LIMIT:
        parser.error(f"--limit must be between 1 and {DEFAULT_LIMIT}")
    out = args.out or Path(os.environ.get("HEARING_RESULTS_DIR", "eval/results")) / "overnight_prep" / TASK_ID
    out.mkdir(parents=True, exist_ok=True)
    cases = discover(args.fixtures, args.limit)
    config = {"task": TASK_ID, "seed": args.seed, "sample_rate": SR, "sir_db": SIR_SWEEP_DB, "target_dbfs": TARGET_DBFS}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    results = []
    for case in cases:
        checkpoint = out / "checkpoints" / f"{case.name}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("status") == "complete" and saved.get("config_sha256") == config_hash:
                results.append(saved["result"])
                continue
        try:
            result = analyse_case(case, out, args.render)
            _atomic_json(checkpoint, {"status": "complete", "config_sha256": config_hash, "result": result})
            results.append(result)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            _atomic_json(checkpoint, {"status": "failed", "config_sha256": config_hash, "error": str(exc)})
            raise
    report = {"task": TASK_ID, "config": config, "config_sha256": config_hash, "planned_cases": args.limit, "available_cases": len(cases), "completed_cases": len(results), "cases": results, "listening_plan": "Level-match target and baseline_sum; shuffle SIR cases; record quiet-target clarity, clipping, overrange and wrong-speaker intrusions by interval."}
    _atomic_json(out / "report.json", report)
    print(json.dumps({"planned_cases": args.limit, "available_cases": len(cases), "completed_cases": len(results), "report": str(out / "report.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
