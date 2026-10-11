from pathlib import Path

import pytest

from eval.scene_score import (
    expected_target,
    frame_times,
    load_dir,
    load_scene,
    oracle_predict,
    score,
    score_many,
    synthesize_tracks,
)

SCENES = Path(__file__).resolve().parent.parent / "eval" / "scenes"


def scene(name):
    return load_scene(SCENES / f"{name}.yaml")


def test_at_least_six_scenes_load_and_synthesise_25fps():
    scenes = load_dir(SCENES)
    assert len(scenes) >= 6
    for s in scenes:
        frames = synthesize_tracks(s)
        assert len(frames) == round(25 * s["duration_s"])
        for tr in (t for f in frames for t in f):
            assert set(tr) == {"track_id", "box", "p_speaking", "last_seen_ts"}
            assert all(0 <= v <= 1 for v in tr["box"])


def test_occlusion_omits_actor():
    s = scene("c_occlusion")
    frames = synthesize_tracks(s)
    assert [t["track_id"] for t in frames[round(3.2 * 25)]] == [2]
    assert len(frames[round(3.6 * 25)]) == 2


@pytest.mark.parametrize("name", ["a_turn_to_b", "d_moving_target"])
def test_oracle_perfect(name):
    s = scene(name)
    assert score(s, oracle_predict(s))["accuracy"] == 1.0


def test_oracle_cli_scenes_have_no_errors():
    for s in load_dir(SCENES):
        assert 0 <= score(s, oracle_predict(s))["accuracy"] <= 1


def test_always_wrong_scores_zero():
    s = scene("a_turn_to_b")
    wrong = [(t, 2 if expected_target(s, t) == 1 else 1) for t in frame_times(s)]
    r = score(s, wrong)
    assert r["accuracy"] == 0.0


def test_late_prediction_latency_and_grace():
    s = scene("a_turn_to_b")
    late = [(t, expected_target(s, t - 0.5)) for t in frame_times(s)]
    r = score(s, late)
    assert r["accuracy"] == 1.0
    assert r["switch_latency_s"][0] == pytest.approx(0.5, abs=0.02)
    assert r["false_switches"] == 0


def test_grace_excludes_one_second():
    s = scene("a_turn_to_b")
    # wrong from t=4 to t=5.5: frames in [4,5) excluded, [5,5.5) counted
    pred = [(t, 1 if 4 <= t < 5.5 else expected_target(s, t)) for t in frame_times(s)]
    r = score(s, pred)
    assert r["frames_scored"] == 7 * 25
    assert r["accuracy"] == pytest.approx(1 - 13 / 175, abs=1e-6)


def test_false_switches_counted():
    s = scene("b_talk_over")
    pred = [(t, 2 if 3 <= t < 3.4 or 5 <= t < 5.2 else 1) for t in frame_times(s)]
    r = score(s, pred)
    assert r["false_switches"] == 2
    assert r["switch_latency_s"] == []


def test_never_switching_gives_none_latency():
    s = scene("a_turn_to_b")
    r = score(s, [(t, 1) for t in frame_times(s)])
    assert r["switch_latency_s"] == [None]


@pytest.mark.slow
def test_oracle_over_all_scenes_end_to_end():
    result = score_many(load_dir(SCENES), oracle_predict)
    assert result["accuracy"] > 0.95
