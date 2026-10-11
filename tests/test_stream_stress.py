"""Tests for streaming stress-test corpus generation (T-293)."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from eval import stream_stress


def test_build_scenario_definitions_count_and_uniqueness():
    """Verify that build_scenario_definitions creates exactly 40 unique scenarios."""
    scenarios = stream_stress.build_scenarios()
    assert len(scenarios) == 40, f"Expected 40 scenarios, got {len(scenarios)}"

    ids = [s.id for s in scenarios]
    assert len(set(ids)) == 40, "Scenario IDs must be unique"

    seeds = [s.seed for s in scenarios]
    assert len(set(seeds)) == 40, "Scenario seeds must be unique"

    categories = {s.category for s in scenarios}
    assert "rapid_turn_taking" in categories
    assert "simultaneous_onset" in categories
    assert "micro_pause_200ms" in categories
    assert "gain_asymmetry_6db" in categories
    assert "compound_stress" in categories


def test_scenario_rendering_single(tmp_path):
    """Test rendering of a single scenario to a temporary directory."""
    scenarios = stream_stress.build_scenarios()
    spec = scenarios[0]

    out_dir = tmp_path / spec.id
    record = stream_stress.render_scenario(spec, out_dir)
    assert record["id"] == spec.id
    assert record["num_turns"] == len(spec.turns)

    assert (out_dir / "mix.wav").is_file()
    assert (out_dir / "turn_labels.json").is_file()
    assert (out_dir / "scenario_card.md").is_file()

    # Verify audio properties
    audio, sr = sf.read(out_dir / "mix.wav", dtype="float32")
    assert sr == 16000
    assert audio.ndim == 1
    assert len(audio) == round(spec.duration_s * sr)
    assert np.max(np.abs(audio)) <= 1.0

    # Verify turn labels JSON
    labels = json.loads((out_dir / "turn_labels.json").read_text())
    assert labels["scenario_id"] == spec.id
    assert labels["sample_rate"] == 16000
    assert len(labels["turns"]) == len(spec.turns)

    for turn in labels["turns"]:
        assert turn["start_s"] < turn["end_s"]
        assert turn["start_sample"] < turn["end_sample"]
        assert 0 <= turn["start_sample"] <= len(audio)
        assert 0 <= turn["end_sample"] <= len(audio)

    # Verify scenario card
    card = (out_dir / "scenario_card.md").read_text()
    assert spec.id in card
    assert "EMA" in card
    assert "Ground-Truth Timeline" in card


def test_spot_check_accuracy(tmp_path):
    """Spot-check 3 scenarios for label accuracy: audio energy in turns vs silence."""
    scenarios = stream_stress.build_scenarios()
    # Pick 3 representative scenarios from different categories
    picked_indices = [0, 10, 20]  # rapid_turn_taking, simultaneous_onset, micro_pause_200ms
    results = []

    for idx in picked_indices:
        spec = scenarios[idx]
        out_dir = tmp_path / spec.id
        record = stream_stress.render_scenario(spec, out_dir)
        assert record["id"] == spec.id
        check = stream_stress.spot_check_scenario(out_dir)
        results.append(check)
        assert check["passed"] is True, f"Spot check failed for {spec.id}: {check}"

    assert len(results) == 3


def test_corpus_manifest_and_files_exist():
    """Verify that eval/stream_stress/T-293/ has all 40 scenarios and valid manifest."""
    corpus_dir = stream_stress.DEFAULT_OUT_DIR
    manifest_path = corpus_dir / "manifest.json"
    assert manifest_path.is_file(), f"Missing {manifest_path}"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["task"] == "T-293"
    assert manifest["total_scenarios"] == 40
    assert len(manifest["scenarios"]) == 40

    # Ensure mix.wav files exist; render if absent (since *.wav is gitignored per repo rules)
    if not (corpus_dir / "scenario_01" / "mix.wav").is_file():
        stream_stress.render_corpus(corpus_dir)

    # Ensure every scenario directory and file exists
    for s in manifest["scenarios"]:
        s_dir = corpus_dir / s["id"]
        assert s_dir.is_dir(), f"Missing directory {s_dir}"
        assert (s_dir / "mix.wav").is_file(), f"Missing mix.wav in {s_dir}"
        assert (s_dir / "turn_labels.json").is_file(), f"Missing turn_labels.json in {s_dir}"
        assert (s_dir / "scenario_card.md").is_file(), f"Missing scenario_card.md in {s_dir}"

        # Check recorded seeds
        assert s["seed"] >= 29301


def test_spot_check_artifact():
    """Verify spot_check.json exists and all spot-checked scenarios passed."""
    corpus_dir = stream_stress.DEFAULT_OUT_DIR
    spot_check_path = corpus_dir / "spot_check.json"
    assert spot_check_path.is_file(), f"Missing {spot_check_path}"

    checks = json.loads(spot_check_path.read_text())
    assert len(checks) >= 3, "Spot check must cover at least 3 scenarios"
    for check in checks:
        assert check["passed"] is True, f"Scenario {check.get('scenario_id')} failed spot check"


def test_deterministic_rendering_reproducibility(tmp_path):
    """Verify that rendering a scenario twice produces identical sha256 and audio."""
    scenarios = stream_stress.build_scenarios()
    spec = scenarios[0]

    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"

    rec1 = stream_stress.render_scenario(spec, dir1)
    rec2 = stream_stress.render_scenario(spec, dir2)

    assert rec1["audio_metrics"]["sha256"] == rec2["audio_metrics"]["sha256"]
    a1, _ = sf.read(dir1 / "mix.wav")
    a2, _ = sf.read(dir2 / "mix.wav")
    np.testing.assert_array_equal(a1, a2)

