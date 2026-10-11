"""Offline tests for overlap stress manifests and exact ratio rendering."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.render_overlap_ratios import detect_overlaps, render_set


def test_detect_overlaps_returns_pairwise_intersections():
    result = detect_overlaps({"b": [(1.0, 3.0)], "a": [(0.0, 2.0), (2.5, 4.0)]})
    assert result == [
        {"speakers": ["a", "b"], "start_s": 1.0, "end_s": 2.0, "duration_s": 1.0},
        {"speakers": ["a", "b"], "start_s": 2.5, "end_s": 3.0, "duration_s": 0.5},
    ]


def test_render_set_is_deterministic_and_hits_ratio_grid(tmp_path: Path):
    first = render_set(tmp_path / "first", source_root=Path("eval/fixtures/tiny"), seed=7)
    second = render_set(tmp_path / "second", source_root=Path("eval/fixtures/tiny"), seed=7)
    assert first["ratios_db"] == [0.0, -5.0, -10.0, -15.0]
    assert [s["measured_overlap_ratio_db"] for s in first["shards"]] == [
        s["measured_overlap_ratio_db"] for s in second["shards"]
    ]
    for expected, shard in zip(first["ratios_db"], first["shards"]):
        assert abs(shard["measured_overlap_ratio_db"] - expected) < 0.01
        assert shard["overlap_intervals"] == [
            {"speakers": ["interferer", "target"], "start_s": 1.25, "end_s": 2.75, "duration_s": 1.5}
        ]
        manifest_path = tmp_path / "first" / shard["id"] / "manifest.json"
        assert json.loads(manifest_path.read_text()) == json.loads(
            (tmp_path / "second" / shard["id"] / "manifest.json").read_text()
        )
    rendered, _ = sf.read(tmp_path / "first" / "ratio_0db" / "mix.wav", dtype="float32")
    assert np.max(np.abs(np.asarray(rendered))) <= 0.9001
