"""Prepare deterministic local gain-disparity and pause/absence render cases.

This is intentionally an offline preparation tool, not a production mixer.  It reads
existing fixture/profile manifests and WAVs, never edits them, and writes a separate
ignored shard.  Rendered samples are float32 sums without peak normalisation; a
manifest records when a render can exceed [-1, 1].

Example (the first 12 cases are quick to inspect)::

    .venv/bin/python -m eval.prep_render_gain_pause --sources eval/fixtures/generated \
        --out /tmp/gain-pause --limit 12
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hearing.audio import io as audio_io

SR = audio_io.PIPELINE_SR
SIRS_DB = (0.0, -6.0, -10.0, -15.0)
ONSET_SHIFTS_S = (-0.25, 0.0, 0.25)
MAX_CASES = 600
MAX_SECONDS = 90 * 60
MAX_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class Source:
    source_id: str
    speaker_id: str
    role: str
    wav: Path
    transcript: str
    intervals: tuple[tuple[float, float], ...]
    group: str
    metadata: dict


@dataclass(frozen=True)
class Case:
    case_id: str
    target: Source
    interferer: Source
    third: Source | None
    sir_db: float
    schedule: str
    onset_shift_s: float
    target_on: bool
    pause_intervals: tuple[tuple[float, float], ...]
    seed: int


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _intervals(value: object, duration: float) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list):
        return ()
    result = []
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start, end = max(0.0, float(item[0])), min(duration, float(item[1]))
            if end > start:
                result.append((start, end))
    return tuple(result)


def _source_from_manifest(manifest_path: Path, role: str = "target") -> Source | None:
    data = _json(manifest_path)
    root = manifest_path.parent
    paths = data.get("paths", {})
    name = paths.get(role, f"{role}.wav")
    if isinstance(name, dict):
        name = name.get("path", "")
    wav = root / str(name)
    if not wav.exists():
        return None
    source = data.get("source", {})
    speaker = source.get(f"{role}_speaker") or source.get("speaker_id") or data.get(f"{role}_id")
    if not speaker:
        speaker = data.get(f"{role}_id") or f"{data.get('id', root.name)}:{role}"
    duration = float(data.get("duration_s", 0.0))
    intervals = data.get(f"{role}_intervals", data.get("intervals", []))
    return Source(
        source_id=str(data.get(f"{role}_id") or data.get("id") or root.name),
        speaker_id=str(speaker),
        role=role,
        wav=wav.resolve(),
        transcript=str(data.get("transcript", "")) if role == "target" else str(data.get("interferer_transcript", "")),
        intervals=_intervals(intervals, duration),
        group=str(data.get("set") or source.get("group") or "unknown"),
        metadata={"manifest": str(manifest_path.resolve()), "source": source},
    )


def discover_sources(root: Path) -> list[Source]:
    """Discover target/interferer stems without modifying the source tree."""
    found: dict[tuple[str, str], Source] = {}
    for manifest in sorted(root.rglob("manifest.json")):
        for role in ("target", "interferer"):
            source = _source_from_manifest(manifest, role)
            if source and source.speaker_id and source.wav.exists():
                found.setdefault((source.source_id, role), source)
    return sorted(found.values(), key=lambda s: (s.speaker_id, s.source_id, s.role))


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64))))) if x.size else 0.0


def _scale_to_sir(target: np.ndarray, interferer: np.ndarray, sir_db: float) -> np.ndarray:
    """Return a scaled copy; zero/silent sources are rejected rather than fabricated."""
    target_rms, interferer_rms = _rms(target), _rms(interferer)
    if target_rms == 0.0 or interferer_rms == 0.0:
        raise ValueError("cannot set SIR with a silent target or interferer")
    return interferer * (target_rms / interferer_rms) * 10 ** (-sir_db / 20)


def _shift(x: np.ndarray, samples: int) -> np.ndarray:
    out = np.zeros_like(x)
    if samples >= 0:
        if samples < len(x):
            out[samples:] = x[: len(x) - samples]
    elif -samples < len(x):
        out[: len(x) + samples] = x[-samples:]
    return out


def _mask(n: int, intervals: Iterable[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(n, dtype=np.float32)
    for start, end in intervals:
        mask[round(start * SR) : min(n, round(end * SR))] = 1.0
    return mask


def _pause_schedule(duration: float) -> tuple[str, tuple[tuple[float, float], ...]]:
    pause_start = max(0.5, duration * 0.55)
    pause_end = min(duration, pause_start + max(0.25, duration * 0.2))
    return "target_pause", ((pause_start, pause_end),)


def plan_cases(sources: list[Source], seed: int = 0, limit: int = MAX_CASES) -> list[Case]:
    """Create a stable, source-disjoint-aware case plan; no audio is loaded."""
    targets = [s for s in sources if s.role == "target"]
    cases: list[Case] = []
    for target in targets:
        others = [s for s in sources if s.speaker_id != target.speaker_id and s.wav != target.wav]
        for interferer in others:
            if interferer.speaker_id == target.speaker_id:
                continue
            third_candidates = [s for s in others if s.speaker_id != interferer.speaker_id]
            third = third_candidates[0] if len({x.speaker_id for x in [target, interferer, *third_candidates]}) >= 3 else None
            try:
                duration = float(target.metadata.get("duration_s", 0.0)) or 3.0
            except (TypeError, ValueError):
                duration = 3.0
            pause_name, pause = _pause_schedule(duration)
            for sir in SIRS_DB:
                for schedule, on, pause_intervals in (
                    ("target_on", True, ()),
                    ("target_off", False, ()),
                    (pause_name, True, pause),
                ):
                    for shift in ONSET_SHIFTS_S:
                        case_id = f"{len(cases):04d}_{target.source_id}_{interferer.source_id}_sir{sir:g}_{schedule}_shift{shift:g}"
                        cases.append(Case(case_id, target, interferer, third, sir, schedule, shift, on, pause_intervals, seed))
                        if len(cases) >= min(limit, MAX_CASES):
                            return cases
    return cases


def _read(source: Source) -> np.ndarray:
    audio, _ = audio_io.load(source.wav, SR)
    return np.asarray(audio, dtype=np.float32)


def render_case(case: Case) -> tuple[np.ndarray, dict]:
    target = _read(case.target)
    interferer = _scale_to_sir(target, _read(case.interferer), case.sir_db)
    interferer = _shift(interferer, round(case.onset_shift_s * SR))
    n = max(len(target), len(interferer))
    target = np.pad(target, (0, n - len(target)))
    interferer = np.pad(interferer, (0, n - len(interferer)))
    target = target * (1.0 if case.target_on else 0.0)
    if case.pause_intervals:
        target = target * (1.0 - _mask(n, case.pause_intervals))
    stems = {"target": target.astype(np.float32), "interferer": interferer.astype(np.float32)}
    if case.third is not None:
        third = _read(case.third)
        third = np.pad(third, (0, max(0, n - len(third))))[:n]
        stems["third"] = third.astype(np.float32)
    mix = np.sum(np.stack(list(stems.values())), axis=0, dtype=np.float64).astype(np.float32)
    return mix, {
        "declared_overrange": bool(np.max(np.abs(mix), initial=0.0) > 1.0),
        "peak": float(np.max(np.abs(mix), initial=0.0)),
        "duration_s": len(mix) / SR,
        "transcript": case.target.transcript if case.target_on and not case.pause_intervals else "",
    }


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def _atomic_wav(path: Path, audio: np.ndarray) -> None:
    temp = path.with_name(f".{path.name}.tmp.wav")
    audio_io.save(temp, audio, SR)
    temp.replace(path)


def render_batch(cases: list[Case], out: Path, checkpoint: Path | None = None, max_seconds: float = MAX_SECONDS) -> int:
    started = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint or out / "checkpoint.json"
    state = _json(checkpoint) if checkpoint.exists() else {"completed": [], "case_count": len(cases)}
    completed = set(state.get("completed", []))
    for case in cases:
        if case.case_id in completed:
            continue
        if time.monotonic() - started >= max_seconds or sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) >= MAX_BYTES:
            break
        mix, render_meta = render_case(case)
        case_dir = out / case.case_id
        case_dir.mkdir(exist_ok=True)
        _atomic_wav(case_dir / "mix.wav", mix)
        manifest = {
            "id": case.case_id,
            "sample_rate": SR,
            "seed": case.seed,
            "source_crops": {
                "target": {"path": str(case.target.wav), "role": "target", "source_id": case.target.source_id, "speaker_id": case.target.speaker_id},
                "interferer": {"path": str(case.interferer.wav), "role": "interferer", "source_id": case.interferer.source_id, "speaker_id": case.interferer.speaker_id},
                **({"third": {"path": str(case.third.wav), "role": "third", "source_id": case.third.source_id, "speaker_id": case.third.speaker_id}} if case.third else {}),
            },
            "enrolment_group": case.target.group,
            "final_group": case.interferer.group,
            "sir_db": case.sir_db,
            "schedule": case.schedule,
            "onset_shift_s": case.onset_shift_s,
            "pause_intervals": case.pause_intervals,
            "stems": "float32; mix is an unnormalised sum; WAV serialization clips only at [-1, 1]",
            **render_meta,
        }
        _atomic_json(case_dir / "manifest.json", manifest)
        completed.add(case.case_id)
        _atomic_json(checkpoint, {"completed": sorted(completed), "case_count": len(cases), "updated_at": time.time()})
    return len(completed)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    if not 1 <= args.limit <= MAX_CASES:
        parser.error(f"--limit must be between 1 and {MAX_CASES}")
    sources = discover_sources(args.sources)
    if len({s.speaker_id for s in sources}) < 2:
        raise SystemExit("need at least two distinct source speakers")
    return 0 if render_batch(plan_cases(sources, args.seed, args.limit), args.out, max_seconds=args.max_seconds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
