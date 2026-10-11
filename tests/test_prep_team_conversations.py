from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.prep_team_conversations import (
    SAMPLE_RATE,
    build_scenario,
    discover_takes,
    generate_manifest,
)


def _take(root: Path, speaker: str, name: str, frequency: float) -> None:
    out = root / speaker
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(SAMPLE_RATE * 3, dtype=np.float32) / SAMPLE_RATE
    sf.write(out / name, (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32), SAMPLE_RATE)


def test_discover_requires_distinct_speaker_takes(tmp_path):
    _take(tmp_path, "aryan", "fixture-01.wav", 180)
    _take(tmp_path, "joseph", "fixture-01.wav", 240)

    takes = discover_takes(tmp_path)

    assert set(takes) == {"aryan", "joseph"}
    assert takes["aryan"][0].name == "fixture-01.wav"


def test_scenario_is_ten_seconds_and_has_timed_turns(tmp_path):
    _take(tmp_path, "aryan", "fixture-01.wav", 180)
    _take(tmp_path, "joseph", "fixture-01.wav", 240)
    takes = discover_takes(tmp_path)

    scenario = build_scenario(takes, "alternating", seed=7)

    assert scenario["duration_s"] == 10.0
    assert scenario["sample_rate"] == SAMPLE_RATE
    assert scenario["turns"][0]["speaker"] == "aryan"
    assert abs(scenario["turns"][1]["start_s"] - scenario["turns"][0]["end_s"] - 0.2) < 1e-6
    assert len(scenario["target"]) == 10 * SAMPLE_RATE
    assert len(scenario["interferer"]) == 10 * SAMPLE_RATE


def test_interruption_starts_after_half_second(tmp_path):
    _take(tmp_path, "aryan", "fixture-01.wav", 180)
    _take(tmp_path, "joseph", "fixture-01.wav", 240)
    turns = build_scenario(discover_takes(tmp_path), "interruption")["turns"]

    assert turns[1]["start_s"] - turns[0]["start_s"] == 0.5


def test_overlap_sir_is_measured_from_nonzero_stems(tmp_path):
    _take(tmp_path, "aryan", "fixture-01.wav", 180)
    _take(tmp_path, "joseph", "fixture-01.wav", 240)
    scenario = build_scenario(discover_takes(tmp_path), "overlap", seed=2, sir_db=-3.0)

    target_rms = np.sqrt(np.mean(scenario["target"] ** 2))
    interferer_rms = np.sqrt(np.mean(scenario["interferer"] ** 2))
    measured = 20 * np.log10(target_rms / interferer_rms)
    assert abs(measured - (-3.0)) < 0.05
    assert scenario["sir_db"] == -3.0


def test_manifest_is_json_and_records_all_benchmark_conditions(tmp_path):
    _take(tmp_path, "aryan", "fixture-01.wav", 180)
    _take(tmp_path, "joseph", "fixture-01.wav", 240)
    out = tmp_path / "manifest.json"

    manifest = generate_manifest(tmp_path, out, seed=11)

    assert len(manifest["scenarios"]) == 7
    assert {row["condition"] for row in manifest["scenarios"]} == {
        "alternating",
        "overlap_-3db",
        "overlap_0db",
        "overlap_3db",
        "interruption",
    }
    assert json.loads(out.read_text()) == manifest
    assert all(row["source"]["aryan"] != row["source"]["joseph"] for row in manifest["scenarios"])
