from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from eval.overnight_pair_catalog import (
    SourceClip,
    build_pair_plan,
    source_stats,
    target_absent_plans,
)


def _wav(path: Path, frequency: float, amplitude: float, seconds: float = 1.0) -> None:
    t = np.arange(round(16000 * seconds), dtype=np.float32) / 16000
    sf.write(path, (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32), 16000)


def test_source_stats_are_deterministic_and_include_pitch_and_level(tmp_path: Path):
    path = tmp_path / "a.wav"
    _wav(path, 180, 0.2)
    stats = source_stats(path)
    assert stats["sample_rate"] == 16000
    assert stats["duration_s"] == 1.0
    assert 170 < stats["median_pitch_hz"] < 190
    assert stats["rms_dbfs"] < -10


def test_pair_plan_excludes_same_speaker_and_carries_sir_grid(tmp_path: Path):
    for name, frequency in (("a.wav", 180), ("b.wav", 190), ("c.wav", 200)):
        _wav(tmp_path / name, frequency, 0.2)
    clips = [
        SourceClip("a", "spk-a", "F", "one", tmp_path / "a.wav", {"dataset": "tiny"}),
        SourceClip("b", "spk-b", "M", "two", tmp_path / "b.wav", {"dataset": "tiny"}),
        SourceClip("c", "spk-a", "F", "three", tmp_path / "c.wav", {"dataset": "tiny"}),
    ]
    plan = build_pair_plan(clips, max_pairs=10)
    assert len(plan) == 2
    ab = next(p for p in plan if p["pair_id"] == "a__b")
    assert ab["sir_db"] == [5, 0, -5, -10, -15]
    assert ab["speaker_group"] == "different_documented_sex"
    assert all(p["target"]["speaker_id"] != p["interferer"]["speaker_id"] for p in plan)


def test_target_absent_plans_have_interval_provenance_and_are_complete():
    plans = target_absent_plans()
    assert len(plans) == 12
    assert {p["condition"] for p in plans} == {"pause", "target_absent", "two_vs_three_talker"}
    for plan in plans:
        assert plan["intervals"]
        assert plan["interval_provenance"]
        assert plan["target_words_status"] in {"unverified", "complete"}
        assert plan["source_overlap_excluded"] is True
