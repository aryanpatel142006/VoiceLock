"""Vision scoring: real tracking/selection, only detection is mocked offline."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from hearing.vision.faces import MediaPipeBackend, MockBackend, weights_available


def manifest():
    return {
        "id": "fake_two_face", "duration_s": 2.0, "fps": 25.0,
        "n_frames": 50, "frame_size": [640, 360], "layout": "two_face",
        "target_id": "grid-s1-utterance", "interferer_id": "grid-s4-utterance",
        "paths": {"video": "video.mp4"}, "target_side": "right",
        "faces": [
            {"actor": "grid-s4", "role": "interferer", "side": "left", "box": [0.05, 0.3, 0.1, 0.4]},
            {"actor": "grid-s1", "role": "target", "side": "right", "box": [0.5, 0.3, 0.1, 0.4]},
        ],
        "target_intervals": [[0.4, 1.2]], "interferer_intervals": [[0.0, 1.8]],
    }


def test_scoring_maps_track_ids_to_talkers_and_keeps_silent_target():
    from eval.vision_eval import evaluate_frames

    m = manifest()
    # Interferer gets track 1: equating detector ids to expected ids would fail.
    backend = MockBackend([[f["box"] for f in m["faces"]]], loop=True, landmarks=True)
    result = evaluate_frames(m, np.zeros((50, 8, 8, 3), np.uint8), backend)
    assert result["accuracy"] == pytest.approx(46 / 50)  # 0.04 s confirmation + 0.12 s acquisition
    assert result["frames_scored"] == 50
    assert result["false_switches"] == 0
    assert result["switch_latency_s"] == []
    assert result["mean_switch_latency_s"] is None
    assert result["frames_with_faces"] == 49
    assert result["fps"] > 0


def test_missing_faces_score_zero_and_empty_video_is_rejected():
    from eval.vision_eval import evaluate_frames

    m = manifest()
    result = evaluate_frames(m, np.zeros((50, 8, 8, 3), np.uint8), MockBackend())
    assert result["accuracy"] == 0
    assert result["frames_with_faces"] == 0
    with pytest.raises(ValueError, match="frame count"):
        evaluate_frames(m, [], MockBackend())


def test_cli_writes_results_and_enforces_threshold(tmp_path, monkeypatch):
    from eval import vision_eval

    fixture = tmp_path / "set" / "fake"
    fixture.mkdir(parents=True)
    (fixture / "manifest.json").write_text(json.dumps(manifest()))
    monkeypatch.setattr(vision_eval, "MediaPipeBackend", MockBackend)
    monkeypatch.setattr(vision_eval, "read_video", lambda *a, **kw: (np.zeros((50, 8, 8, 3), np.uint8), 25))
    monkeypatch.setattr(vision_eval.device, "describe", lambda: {"device": "cpu"})
    out = tmp_path / "results"
    assert vision_eval.main([str(fixture.parent), "--output-dir", str(out), "--min-accuracy", "0.9"]) == 1
    result = json.loads(next(out.glob("vision_*.json")).read_text())
    assert result["accuracy"] == 0
    assert result["device"] == {"device": "cpu"}
    assert result["per_fixture"][0]["frames"] == 50


@pytest.mark.slow
def test_real_grid_fixture_end_to_end():
    from eval.vision_eval import evaluate_fixture

    root = Path(os.environ.get("HEARING_FIXTURES_DIR") or "eval/fixtures/generated")
    fixtures = [p for p in sorted(root.glob("*/*/manifest.json"))
                if json.loads(p.read_text()).get("layout") == "two_face"]
    if not fixtures or not weights_available():
        pytest.skip("requires generated GRID two-face fixture and MediaPipe weights")
    pytest.importorskip("mediapipe")
    with MediaPipeBackend() as backend:
        result = evaluate_fixture(fixtures[0], backend)
    assert result["frames"] > 0
    assert 0 <= result["accuracy"] <= 1
    assert result["fps"] > 0


def test_central_interferer_is_not_relabelled_as_target():
    from eval.vision_eval import evaluate_frames

    m = manifest()
    m["faces"][0]["role"] = "target"
    m["faces"][1]["role"] = "interferer"
    backend = MockBackend([[f["box"] for f in m["faces"]]], loop=True)
    result = evaluate_frames(m, np.zeros((50, 8, 8, 3), np.uint8), backend)
    assert result["accuracy"] == 0
    assert result["false_switches"] == 1
