"""Bounded, CPU-only audit of delivered audio output alignment and artefacts.

The input manifest is JSON with ``cases`` containing ``id``, ``reference``,
``output``, and optional ``target_active`` WAV paths.  Paths are relative to
the manifest.  Optional ``source``/case ``provenance`` is copied verbatim so
the report can be traced to an original speaker, utterance, interval, and
hash without inventing metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SR = 16_000
FRAME = 160


@dataclass(frozen=True)
class AuditConfig:
    max_lag_ms: float = 300.0
    frame_ms: float = 10.0
    dropout_db: float = -20.0
    click_z: float = 8.0
    clip_epsilon: float = 1e-4
    max_cases: int = 300
    max_seconds: float = 3600.0
    max_bytes: int = 1_000_000_000


DEFAULT_CONFIG = AuditConfig()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"{path}: expected mono PCM16 WAV")
        sr = handle.getframerate()
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0, sr


def _lag_samples(output: np.ndarray, reference: np.ndarray, max_lag: int) -> int | None:
    n = min(len(output), len(reference))
    if n < 2:
        return None
    a, b = output[:n].astype(np.float64), reference[:n].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
        return None
    corr = np.correlate(a, b, mode="full")
    center = n - 1
    lo, hi = max(0, center - max_lag), min(len(corr), center + max_lag + 1)
    return int(np.argmax(np.abs(corr[lo:hi])) + lo - center)


def _aligned(a: np.ndarray, b: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        return a[lag:], b[: len(a) - lag]
    return a[: len(a) + lag], b[-lag:]


def _frames(x: np.ndarray, size: int = FRAME) -> np.ndarray:
    count = len(x) // size
    return x[: count * size].reshape(count, size)


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x.astype(np.float64)), axis=1))


def _db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _click_metrics(output: np.ndarray, reference: np.ndarray, z: float) -> dict:
    out_d = np.abs(np.diff(output.astype(np.float64)))
    ref_d = np.abs(np.diff(reference.astype(np.float64)))
    baseline = float(np.percentile(ref_d, 95)) if len(ref_d) else 0.0
    scale = max(float(np.median(np.abs(ref_d - np.median(ref_d)))) * 1.4826, 1e-6)
    candidates = np.flatnonzero(out_d > baseline + z * scale)
    ordinary_p95 = float(np.percentile(ref_d, 95)) if len(ref_d) else 0.0
    return {
        "candidate_boundaries": len(candidates),
        "suspect_boundaries": int(np.sum(out_d[candidates] > baseline + z * scale)),
        "ordinary_change_p95": ordinary_p95,
        "largest_change": float(np.max(out_d)) if len(out_d) else 0.0,
        "threshold": baseline + z * scale,
    }


def audit_case(case: dict, config: AuditConfig | None = None, *, base: Path | None = None,
               source: dict | None = None) -> dict:
    config = config or DEFAULT_CONFIG
    root = base or Path(".")
    ref_path, out_path = tuple((root / case[key]).resolve() for key in ("reference", "output"))
    reference, ref_sr = _read(ref_path)
    output, out_sr = _read(out_path)
    if ref_sr != SR or out_sr != SR:
        raise ValueError(f"{case.get('id', '<case>')}: WAVs must be {SR} Hz")
    max_lag = round(config.max_lag_ms * SR / 1000)
    detected_lag = _lag_samples(output, reference, max_lag)
    lag = 0 if detected_lag is None else detected_lag
    aligned_out, aligned_ref = _aligned(output, reference, lag)
    n = min(len(aligned_out), len(aligned_ref))
    aligned_out, aligned_ref = aligned_out[:n], aligned_ref[:n]
    frame_out, frame_ref = _frames(aligned_out), _frames(aligned_ref)
    ref_rms = _rms(frame_ref) if len(frame_ref) else np.empty(0)
    out_rms = _rms(frame_out) if len(frame_out) else np.empty(0)
    active = ref_rms > max(np.percentile(ref_rms, 25), 1e-6) if len(ref_rms) else np.empty(0, bool)
    mask_path = case.get("target_active")
    if mask_path:
        mask, mask_sr = _read((root / mask_path).resolve())
        if mask_sr != SR:
            raise ValueError(f"{case.get('id', '<case>')}: target mask must be {SR} Hz")
        active = (_frames(mask[: len(frame_out) * FRAME])[:, 0] > 0.5)[: len(out_rms)]
    matched_scale = float(np.sqrt(np.mean(aligned_ref**2)) / max(np.sqrt(np.mean(aligned_out**2)), 1e-12))
    matched_out = aligned_out * matched_scale
    matched_frame = _rms(_frames(matched_out))
    active = active[: len(matched_frame)]
    dropout = active & (matched_frame[: len(active)] < np.maximum(ref_rms[: len(active)], 1e-6) * 10 ** (config.dropout_db / 20))
    return {
        "id": case.get("id", out_path.stem),
        "provenance": case.get("provenance", source or {}),
        "inputs": {
            "reference": str(case["reference"]),
            "output": str(case["output"]),
            "reference_sha256": _sha256(ref_path),
            "output_sha256": _sha256(out_path),
        },
        "alignment": {
            "lag_ms": round(1000 * lag / SR, 3),
            "overlap_s": round(n / SR, 4),
            "reference_s": round(len(reference) / SR, 4),
            "output_s": round(len(output) / SR, 4),
            "length_delta_ms": round(1000 * (len(output) - len(reference)) / SR, 3),
            "unavailable": detected_lag is None,
        },
        "matched_level": {"scale_db": round(_db(matched_scale), 3), "scale": matched_scale},
        "clipping": {
            "clipped_samples": int(np.sum(np.abs(output) >= 1.0 - config.clip_epsilon)),
            "fraction": float(np.mean(np.abs(output) >= 1.0 - config.clip_epsilon)),
        },
        "dropouts": {
            "active_frames": int(np.sum(active)),
            "dropout_frames": int(np.sum(dropout)),
            "dropout_s": round(float(np.sum(dropout) * config.frame_ms / 1000), 4),
            "threshold_db": config.dropout_db,
            "mask": "target_active" if mask_path else "reference_rms_quartile",
        },
        "clicks": _click_metrics(aligned_out, aligned_ref, config.click_z),
    }


def _config_hash(config: AuditConfig) -> str:
    return hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def run_manifest(manifest_path: Path, results_dir: Path, *, config: AuditConfig | None = None,
                 resume: bool = False) -> dict:
    config = config or DEFAULT_CONFIG
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text())
    cases = payload.get("cases", [])
    if len(cases) > config.max_cases:
        raise ValueError(f"manifest has {len(cases)} cases; cap is {config.max_cases}")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_cases: list[dict] = []
    resumed = 0
    total_bytes = 0
    total_seconds = 0.0
    for case in cases:
        checkpoint = results_dir / f"{case.get('id', 'case')}.json"
        if resume and checkpoint.exists():
            report_cases.append(json.loads(checkpoint.read_text()))
            resumed += 1
            continue
        paths = [(manifest_path.parent / case[key]).resolve() for key in ("reference", "output")]
        total_bytes += sum(p.stat().st_size for p in paths)
        if case.get("target_active"):
            mask_path = (manifest_path.parent / case["target_active"]).resolve()
            total_bytes += mask_path.stat().st_size
        if total_bytes > config.max_bytes:
            raise ValueError(f"input byte cap exceeded ({config.max_bytes})")
        result = audit_case(case, config, base=manifest_path.parent, source=payload.get("source"))
        total_seconds += result["alignment"]["output_s"]
        if total_seconds > config.max_seconds:
            raise ValueError(f"audio duration cap exceeded ({config.max_seconds}s)")
        fd, temporary = tempfile.mkstemp(prefix=checkpoint.name, dir=results_dir)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(result, handle, indent=2)
            Path(temporary).replace(checkpoint)
        finally:
            Path(temporary).unlink(missing_ok=True)
        report_cases.append(result)
    report = {
        "schema": "overnight-prep-output-audit/v1",
        "manifest": str(manifest_path),
        "config": asdict(config),
        "config_sha256": _config_hash(config),
        "planned_cases": len(cases),
        "available_cases": len(report_cases),
        "resumed_cases": resumed,
        "source": payload.get("source", {}),
        "cases": report_cases,
    }
    (results_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    parser.add_argument("--max-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--max-lag-ms", type=float, default=300.0)
    parser.add_argument("--dropout-db", type=float, default=-20.0)
    parser.add_argument("--click-z", type=float, default=8.0)
    args = parser.parse_args(argv)
    config = AuditConfig(max_cases=args.max_cases, max_seconds=args.max_seconds, max_bytes=args.max_bytes,
                         max_lag_ms=args.max_lag_ms, dropout_db=args.dropout_db, click_z=args.click_z)
    report = run_manifest(args.manifest, args.results, config=config, resume=args.resume)
    print(json.dumps({"report": str(args.results / "report.json"), "available_cases": report["available_cases"],
                      "resumed_cases": report["resumed_cases"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
