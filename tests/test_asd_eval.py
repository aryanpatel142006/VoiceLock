"""ASD accuracy scorer (T-038): exact metrics on a scripted two-track scene with the mock detector."""

import json

import numpy as np
import pytest

from eval import asd_eval as A
from hearing.types import FaceTrack
from hearing.vision.asd import LipMotionASD, get_asd

FPS = 10  # Frame centres at 0.05, 0.15, ..., 0.95 s keep the hand-computed labels readable.
BOX_A = (0.1, 0.2, 0.3, 0.5)
BOX_B = (0.6, 0.2, 0.3, 0.5)
FACES = [
    A.FaceLabel("target", BOX_A, [(0.2, 0.6)]),
    A.FaceLabel("interferer", BOX_B, [(0.0, 0.3), (0.7, 1.0)]),
]
# Track ids change half-way (as after a tracker loss); face B has no track in the last frame.
MOCK_P = {0: 0.2, 1: 0.9, 2: 0.8, 3: 0.4}


def script() -> list[A.FrameInput]:
    frames = []
    for i in range(10):
        ts = i / FPS
        a, b = (0, 1) if i < 5 else (2, 3)
        # Detector boxes are shifted a little from the manifest boxes, like real detections.
        tracks = [FaceTrack(a, (0.12, 0.22, 0.28, 0.45), None, 0.0, ts)]
        if i < 9:
            tracks.append(FaceTrack(b, (0.62, 0.25, 0.25, 0.45), None, 0.0, ts))
        frames.append(A.FrameInput(ts, tracks, np.zeros(1600, np.float32)))
    return frames


def scored():
    series, update_ms = A.score_frames(get_asd("mock", script=MOCK_P), script(), FACES, fps=FPS)
    return series, update_ms


def test_series_follow_matched_tracks_and_labels():
    series, update_ms = scored()
    a, b = series
    assert len(update_ms) == 10
    np.testing.assert_allclose(a.t, np.arange(10) / FPS + 0.05)
    np.testing.assert_array_equal(a.y, [0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(b.y, [1, 1, 1, 0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_allclose(a.p, [0.2] * 5 + [0.8] * 5)
    np.testing.assert_allclose(b.p, [0.9] * 5 + [0.4] * 4 + [0.0])  # Unmatched frame scores 0.
    assert a.matched.all() and b.matched[:9].all() and not b.matched[9]


def test_metrics_are_exact():
    series, _ = scored()
    r = A.summarize(series)
    assert r["face_frames"] == 20 and r["speaking_frames"] == 10
    assert r["matched_frac"] == pytest.approx(19 / 20)
    assert r["accuracy"] == pytest.approx(8 / 20)
    assert r["precision"] == pytest.approx(4 / 10)  # TP 4 (A: 1, B: 3), FP 6.
    assert r["recall"] == pytest.approx(4 / 10)  # FN 6.
    assert r["auc"] == pytest.approx(42 / 100)  # Mann-Whitney with ties counted half.
    # A: first p >= 0.5 inside [0.2, 0.6) at 0.55 s; B: 0.05 s, then never in [0.7, 1.0).
    assert r["onset_latency_s"] == pytest.approx([0.35, 0.05, None])
    assert r["onsets"] == 3 and r["onsets_missed"] == 1
    assert r["onset_latency_median_s"] == pytest.approx(0.2)


def test_collar_drops_boundary_frames():
    series, _ = scored()
    assert A.summarize(series, collar_s=0.0)["accuracy_collar"] == pytest.approx(8 / 20)
    # Collar 0.1 s keeps A frames 0, 3, 4, 7, 8, 9 (1 right) and B frames 1, 4, 5, 8 (2 right).
    assert A.summarize(series, collar_s=0.1)["accuracy_collar"] == pytest.approx(3 / 10)


def test_roc_auc_edge_cases():
    assert A.roc_auc(np.array([0.1, 0.9]), np.array([False, True])) == 1.0
    assert A.roc_auc(np.array([0.9, 0.1]), np.array([False, True])) == 0.0
    assert A.roc_auc(np.array([0.5, 0.5, 0.5]), np.array([True, False, True])) == 0.5
    assert A.roc_auc(np.array([0.3, 0.4]), np.array([True, True])) is None


def test_match_faces_is_one_to_one_and_thresholded():
    tracks = [FaceTrack(7, BOX_B, None, 0.0, 0.0), FaceTrack(8, (0.11, 0.2, 0.3, 0.5), None, 0.0, 0.0)]
    assert A.match_faces(tracks, FACES) == {0: 1, 1: 0}
    far = [FaceTrack(9, (0.45, 0.0, 0.05, 0.05), None, 0.0, 0.0)]
    assert A.match_faces(far, FACES) == {}


def test_rows_group_by_condition_plus_overall():
    series, _ = scored()
    results = [A.FixtureResult("x", "two_face snr0 sir0", series), A.FixtureResult("y", "two_face snr5 sir0", series[:1])]
    rows = A.rows_for(results)
    assert list(rows) == ["two_face snr0 sir0", "two_face snr5 sir0", "overall"]
    assert rows["overall"]["face_frames"] == 30
    text = A.table("mock", rows)
    assert "== mock" in text and "overall" in text and "acc±c" in text
    assert set(A.by_role(results)) == {"target", "interferer"}


def test_lipmotion_receives_audio_in_10ms_chunks():
    det = LipMotionASD()
    frame = A.FrameInput(0.0, [], np.zeros(640, np.float32))
    assert A._feed_audio(det, frame.audio, 0.0, 0) is None
    assert len(det.audio_energies) == 4  # Four 10 ms energies from one 40 ms frame.


def test_light_asd_unavailable_is_reported(monkeypatch):
    from hearing.vision import light_asd

    monkeypatch.setattr(light_asd, "available", lambda: False)
    det, reason = A.build_detector("light_asd")
    assert det is None and "download" in reason


def test_faces_from_manifest_and_condition():
    m = {
        "layout": "two_face", "snr_db": 0.0, "sir_db": 5.0,
        "faces": [{"role": "target", "box": [0.1, 0.2, 0.3, 0.4]}, {"role": "interferer", "box": [0.6, 0.2, 0.2, 0.4]}],
        "target_intervals": [[0.5, 1.5]], "interferer_intervals": [[0.2, 0.9], [1.2, 2.0]],
    }  # fmt: skip
    faces = A.faces_from_manifest(m)
    assert [f.name for f in faces] == ["target", "interferer"]
    assert faces[1].intervals == [(0.2, 0.9), (1.2, 2.0)]
    assert A.condition(m) == "two_face snr0 sir5"


def test_write_results_names_file(tmp_path):
    path = A.write_results(tmp_path, {"stamp": "20261005-000000", "x": float("nan")})
    assert path.name == "asd_20261005-000000.json"
    assert json.loads(path.read_text())["x"] is None
    assert A.write_results(tmp_path, {"stamp": "20261005-000000"}).name == "asd_20261005-000000-1.json"


def test_main_without_fixtures_says_how_to_make_them(tmp_path, capsys):
    assert A.main(["--set", "tiny", "--fixtures", str(tmp_path)]) == 2
    assert "make_fixtures --av --tiny" in capsys.readouterr().err


@pytest.mark.slow
def test_real_run_on_av_tiny(tmp_path):
    from hearing.vision import faces

    set_dir = A.fixture_set_dir("tiny")
    if not set_dir.is_dir() or not faces.weights_available():
        pytest.skip("needs eval/fixtures/generated/av_tiny (make_fixtures --av --tiny) and MediaPipe weights")
    assert A.main(["--set", "tiny", "--limit", "2", "--out", str(tmp_path)]) == 0
    report = json.loads(next(tmp_path.glob("asd_*.json")).read_text())
    lip = report["backends"]["lipmotion"]
    assert lip["available"] and 0.0 <= lip["rows"]["overall"]["accuracy"] <= 1.0
    assert lip["rows"]["overall"]["matched_frac"] > 0.9
    assert report["machine"]["device"]
    light = report["backends"]["light_asd"]
    assert light["available"] or light["reason"]
