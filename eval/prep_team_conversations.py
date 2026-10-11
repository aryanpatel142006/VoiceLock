"""Build deterministic, offline two-speaker conversation fixtures.

The input is a directory containing ``aryan/fixture*.wav`` and
``joseph/fixture*.wav``.  Profile takes and the existing ``conversation/``
recording are deliberately not considered.  Generated WAVs are written next
to the manifest and are ignored by git; the manifest is the portable index
used by benchmark jobs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
DURATION_S = 10.0
_N_SAMPLES = int(SAMPLE_RATE * DURATION_S)
_EPS = 1e-12


def discover_takes(root: str | Path) -> dict[str, list[Path]]:
    """Return sorted, distinct fixture takes for Aryan and Joseph.

    A byte-identical take in both speaker directories is rejected: benchmark
    separation must not accidentally train/test on the same recording.
    """
    root = Path(root)
    result: dict[str, list[Path]] = {}
    for speaker in ("aryan", "joseph"):
        directory = next((p for p in root.iterdir() if p.is_dir() and p.name.lower() == speaker), None)
        if directory is None:
            raise FileNotFoundError(f"missing speaker take directory: {root / speaker}")
        takes = sorted(directory.glob("fixture*.wav"))
        if not takes:
            raise FileNotFoundError(f"no fixture*.wav takes in {directory}")
        result[speaker] = takes
    hashes = {
        speaker: {hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        for speaker, paths in result.items()
    }
    shared = hashes["aryan"] & hashes["joseph"]
    if shared:
        raise ValueError("Aryan and Joseph fixture takes must be disjoint")
    return result


def _read(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"{path} is {sample_rate} Hz; expected {SAMPLE_RATE} Hz")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if not np.any(np.abs(audio) > _EPS):
        raise ValueError(f"{path} is silent")
    return audio


def _fit(audio: np.ndarray, length: int) -> np.ndarray:
    if not len(audio):
        raise ValueError("cannot fit an empty take")
    return np.resize(audio, length).astype(np.float32, copy=False)


def _place(destination: np.ndarray, source: np.ndarray, start: float, end: float) -> None:
    a, b = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
    if a < 0 or b > len(destination) or b <= a:
        raise ValueError(f"invalid interval [{start}, {end})")
    segment = _fit(source, b - a)
    destination[a:b] += segment


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _scale_sir(target: np.ndarray, interferer: np.ndarray, sir_db: float) -> np.ndarray:
    target_rms, interferer_rms = _rms(target), _rms(interferer)
    if target_rms <= _EPS or interferer_rms <= _EPS:
        raise ValueError("cannot measure SIR from a silent stem")
    return interferer * (target_rms / interferer_rms) / (10.0 ** (sir_db / 20.0))


def _source_paths(takes: dict[str, list[Path]], seed: int) -> tuple[Path, Path]:
    rng = np.random.default_rng(seed)
    return tuple(paths[int(rng.integers(len(paths)))] for paths in (takes["aryan"], takes["joseph"]))


def build_scenario(
    takes: dict[str, list[Path]],
    kind: str,
    *,
    seed: int = 0,
    sir_db: float | None = None,
) -> dict:
    """Render one 10-second scenario and return stems plus timing metadata."""
    if kind not in {"alternating", "overlap", "interruption"}:
        raise ValueError(f"unknown scenario kind: {kind}")
    aryan_path, joseph_path = _source_paths(takes, seed)
    aryan, joseph = _read(aryan_path), _read(joseph_path)
    target = np.zeros(_N_SAMPLES, np.float32)
    interferer = np.zeros(_N_SAMPLES, np.float32)
    turns: list[dict] = []
    if kind == "alternating":
        intervals = [(0.5, 2.0, "aryan"), (2.2, 3.7, "joseph"), (3.9, 5.4, "aryan"),
                     (5.6, 7.1, "joseph"), (7.3, 8.8, "aryan")]
    elif kind == "overlap":
        intervals = [(0.5, 3.0, "aryan"), (2.0, 4.5, "joseph"), (5.0, 7.5, "aryan"),
                     (7.0, 9.5, "joseph")]
    else:
        intervals = [(0.5, 3.0, "aryan"), (1.0, 3.0, "joseph"), (5.0, 7.5, "aryan"),
                     (7.0, 9.0, "joseph")]
    for start, end, speaker in intervals:
        audio = aryan if speaker == "aryan" else joseph
        stem = target if speaker == "aryan" else interferer
        _place(stem, audio, start, end)
        turns.append({"speaker": speaker, "start_s": start, "end_s": end})
    if kind == "overlap":
        if sir_db is None:
            raise ValueError("overlap scenarios require sir_db")
        interferer = _scale_sir(target, interferer, sir_db)
    elif kind == "interruption":
        # The second speaker begins 500 ms into Aryan's first turn.
        if turns[1]["start_s"] - turns[0]["start_s"] != 0.5:
            raise AssertionError("interruption timing must remain 500 ms after turn onset")
    return {
        "duration_s": DURATION_S,
        "sample_rate": SAMPLE_RATE,
        "target": target,
        "interferer": interferer,
        "turns": turns,
        "sir_db": sir_db,
        "source": {"aryan": str(aryan_path), "joseph": str(joseph_path)},
    }


def _condition_rows() -> list[tuple[str, str, float | None]]:
    return [("alternating", "alternating", None)] + [
        (f"overlap_{sir:g}db", "overlap", sir) for sir in (-3.0, 0.0, 3.0)
    ] + [("interruption", "interruption", None)] * 3


def generate_manifest(source_root: str | Path, manifest_path: str | Path, *, seed: int = 0) -> dict:
    """Render the benchmark bank and write its deterministic JSON manifest."""
    takes = discover_takes(source_root)
    source_root = Path(source_root)
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for index, (condition, kind, sir_db) in enumerate(_condition_rows()):
        scenario = build_scenario(takes, kind, seed=seed + index, sir_db=sir_db)
        scenario_id = f"{index:02d}_{condition}"
        scenario_dir = manifest_path.parent / scenario_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        sf.write(scenario_dir / "target.wav", scenario["target"], SAMPLE_RATE)
        sf.write(scenario_dir / "interferer.wav", scenario["interferer"], SAMPLE_RATE)
        sf.write(scenario_dir / "mix.wav", scenario["target"] + scenario["interferer"], SAMPLE_RATE)
        row = {
            "id": scenario_id,
            "condition": condition,
            "duration_s": DURATION_S,
            "sample_rate": SAMPLE_RATE,
            "sir_db": sir_db,
            "paths": {
                "mix": f"{scenario_id}/mix.wav",
                "target": f"{scenario_id}/target.wav",
                "interferer": f"{scenario_id}/interferer.wav",
            },
            "turns": scenario["turns"],
            "source": {
                speaker: str(Path(path).relative_to(source_root))
                for speaker, path in scenario["source"].items()
            },
            "seed": seed + index,
        }
        scenarios.append(row)
    result = {
        "set": "team_conversations",
        "sample_rate": SAMPLE_RATE,
        "duration_s": DURATION_S,
        "seed": seed,
        "speakers": ["aryan", "joseph"],
        "scenarios": scenarios,
    }
    manifest_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("eval/fixtures/generated/team_conversations/manifest.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    generate_manifest(args.source_root, args.manifest, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
