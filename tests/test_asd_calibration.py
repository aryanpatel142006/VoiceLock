"""T-051 operating points: chosen detector defaults, Light-ASD calibration, and the eval's fit helpers."""

import math

import numpy as np
import pytest

from eval import asd_eval as A
from hearing.types import AudioChunk, FaceTrack, VideoFrame
from hearing.vision import light_asd
from hearing.vision.asd import LipMotionASD
from hearing.vision.light_asd import LightASD, MockBackend, synthetic_face_frame

BOX = (0.3, 0.2, 0.4, 0.5)


def test_lipmotion_defaults_are_the_av_full_operating_point():
    det = LipMotionASD()
    assert det.window_s == 0.25 and det.activity_window_s == 1.0 and det.motion_ref == 0.01
    assert det.ema_s == 0.1 and det.min_frames == 4
    assert (det.bias, det.w_motion, det.w_coupling, det.w_activity, det.w_motion_activity) == (
        -1.75, 0.5, 5.0, -1.15, 3.1,
    )  # fmt: skip
    assert LipMotionASD(activity_window_s=None).activity_window_s == 0.25  # None: same as window_s.


def test_lipmotion_logit_is_calibrated_at_the_corners():
    """Moving mouth in silence stays well under even odds; moving mouth + sound clears them; static face never."""
    det = LipMotionASD()

    def p(motion, activity, coupling=0.0):
        logit = (det.bias + det.w_motion * motion + det.w_coupling * coupling + det.w_activity * activity
                 + det.w_motion_activity * motion * activity)  # fmt: skip
        return 1 / (1 + math.exp(-logit))

    assert p(1.0, 0.0) < 0.25  # Chewing in a silent room.
    assert p(0.0, 1.0) < 0.1  # Someone else talks, this mouth is still.
    assert p(1.0, 1.0) > 0.5  # Mouth moving with sound, even with no measurable coupling.
    assert p(1.0, 1.0, coupling=0.8) > 0.95


class LogitBackend:
    """Fixed logit for every track; optional ``calibration`` like TorchBackend."""

    device = "test"
    needs_audio = False

    def __init__(self, logit, calibration=None):
        self.logit = logit
        if calibration is not None:
            self.calibration = calibration

    def scores(self, video, mfcc):
        return np.full(video.shape[0], self.logit, np.float32)


def one_update(det: LightASD) -> float:
    t = FaceTrack(1, BOX, None, 0.0, 0.0)
    det.update([t], 0.0, AudioChunk(np.zeros(640, np.float32), 16000, 0, 0.0),
               VideoFrame(synthetic_face_frame([BOX], [0.3]), 0, 0.0))  # fmt: skip
    return t.p_speaking


def test_light_asd_defaults_and_torch_calibration():
    assert light_asd.DEFAULT_WINDOW == 12
    assert light_asd.TORCH_CALIBRATION[12] == (0.937, -1.826)
    assert light_asd.TorchBackend.calibration is light_asd.TORCH_CALIBRATION
    det = LightASD(backend=MockBackend())
    assert det.ema_s == 0.0 and det.window == 12
    assert det.calibration == (1.0, 0.0)  # Mocks and test backends keep identity.


def test_light_asd_picks_the_nearest_fitted_window():
    table = light_asd.TORCH_CALIBRATION
    assert LightASD(backend=LogitBackend(0.0, table)).calibration == table[12]
    assert LightASD(window=25, backend=LogitBackend(0.0, table)).calibration == table[25]
    assert LightASD(window=30, backend=LogitBackend(0.0, table)).calibration == table[25]
    assert light_asd.calibration_for({8: (1, 0), 12: (2, 0)}, 10) == (1, 0)  # Tie: the shorter window.


def test_light_asd_applies_backend_calibration_and_keeps_raw_logit():
    scale, bias = light_asd.TORCH_CALIBRATION[light_asd.DEFAULT_WINDOW]
    det = LightASD(backend=LogitBackend(5.0, calibration=(scale, bias)))
    assert one_update(det) == pytest.approx(1 / (1 + math.exp(-(scale * 5.0 + bias))))
    assert det.scores[1] == pytest.approx(5.0)  # Raw logit for the eval's refit.
    # The decision point sits at logit -bias/scale (1.95), not 0.
    assert one_update(LightASD(backend=LogitBackend(1.5, calibration=(scale, bias)))) < 0.5
    assert one_update(LightASD(backend=LogitBackend(2.5, calibration=(scale, bias)))) > 0.5
    # An explicit calibration overrides the backend's; no EMA means p follows the logit at once.
    assert one_update(LightASD(backend=LogitBackend(0.0, (scale, bias)), calibration=(1.0, 0.0))) == pytest.approx(0.5)


# --------------------------------------------------------------------- eval helpers
def test_fit_logistic_recovers_known_parameters():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 3, 20000)
    y = rng.random(20000) < 1 / (1 + np.exp(-(0.6 * x - 2.0)))
    w = A.fit_logistic(x[:, None], y)
    assert w == pytest.approx([0.6, -2.0], abs=0.08)
    off = 1.5 * np.ones_like(x)  # A fixed offset is not refitted into the bias.
    y2 = rng.random(20000) < 1 / (1 + np.exp(-(0.6 * x - 2.0 + 1.5)))
    assert A.fit_logistic(x[:, None], y2, off) == pytest.approx([0.6, -2.0], abs=0.08)


def series(logits, y):
    t = np.arange(len(y)) / 25
    return A.FaceSeries(A.FaceLabel("target", BOX, []), t, np.zeros(len(y)), np.asarray(y, bool),
                        np.ones(len(y), bool), np.asarray(logits, float)[:, None])  # fmt: skip


def test_calibrate_light_asd_shape_reports_fit_and_leave_one_pair_out():
    rng = np.random.default_rng(1)
    results = []
    for k in range(4):
        y = rng.random(200) < 0.5
        logits = np.where(y, 6.0, 2.0) + rng.normal(0, 0.5, 200)
        results.append(A.FixtureResult(f"f{k}", "two_face snr0 sir0", [series(logits, y)], snr_db=[0.0, 5.0, 10.0, 0.0][k],
                                       pair=f"p{k}"))  # fmt: skip
    cal = A.calibrate("light_asd", None, results)
    assert set(cal["params"]) == {"scale", "bias"} and cal["pairs"] == 4
    assert -cal["params"]["bias"] / cal["params"]["scale"] == pytest.approx(4.0, abs=0.3)  # Midway.
    assert cal["lopo_accuracy"] > 0.99 and cal["lopo_accuracy_snr0+5"] > 0.99
    # One pair: fit only, no held-out score.
    assert "lopo_accuracy" not in A.calibrate("light_asd", None, results[:1])
    assert A.calibrate("mock", None, [A.FixtureResult("x", "c", [])]) is None


def test_calibration_design_holds_lipmotion_motion_and_coupling():
    det = LipMotionASD()
    x = np.array([[1.0, 0.5, 0.2]])  # motion, coupling, activity
    cols, off, names = A.calibration_design("lipmotion", det, x)
    assert names == ["w_motion_activity", "w_activity", "bias"]
    np.testing.assert_allclose(cols, [[0.2, 0.2]])
    np.testing.assert_allclose(off, [det.w_motion * 1.0 + det.w_coupling * 0.5])


def test_raw_scores_reads_features_or_logits():
    lip = LipMotionASD()
    lip.features[3] = {"motion": 0.1, "coupling": 0.2, "activity": 0.3}
    assert A.raw_scores(lip, 3) == [0.1, 0.2, 0.3] and A.raw_scores(lip, 4) is None
    det = LightASD(backend=MockBackend())
    det.scores[7] = -1.5
    assert A.raw_scores(det, 7) == [-1.5]


def test_by_snr_pools_conditions_and_acceptance_row():
    y = np.array([1, 1, 0, 0], bool)
    good = A.FaceSeries(A.FaceLabel("t", BOX, []), np.arange(4) / 25, np.array([0.9, 0.8, 0.1, 0.2]), y, np.ones(4, bool))
    bad = A.FaceSeries(A.FaceLabel("t", BOX, []), np.arange(4) / 25, np.array([0.1, 0.2, 0.9, 0.8]), y, np.ones(4, bool))
    res = [A.FixtureResult("a", "x", [good], snr_db=0.0), A.FixtureResult("b", "x", [good], snr_db=5.0),
           A.FixtureResult("c", "x", [bad], snr_db=-5.0)]  # fmt: skip
    rows = A.by_snr(res)
    assert list(rows) == ["snr-5", "snr0", "snr5", "snr0+5"]
    assert rows["snr0+5"]["accuracy"] == 1.0 and rows["snr0+5"]["face_frames"] == 8
    assert rows["snr-5"]["accuracy"] == 0.0


def test_parse_params_and_set_aliases(tmp_path):
    items = ["lipmotion.window_s=0.3", "light_asd.calibration=[1, 0]", "lipmotion.min_frames=4", "light_asd.backend=torch"]
    assert A.parse_params(items) == {
        "lipmotion": {"window_s": 0.3, "min_frames": 4},
        "light_asd": {"calibration": [1, 0], "backend": "torch"},
    }
    with pytest.raises(ValueError, match="BACKEND.KEY=VALUE"):
        A.parse_params(["window_s=0.3"])
    assert A.fixture_set_dir("av_full", tmp_path) == tmp_path / "av_full"
    assert A.fixture_set_dir("full", tmp_path) == tmp_path / "av_full"
    assert A.fixture_set_dir("av_tiny", tmp_path) == tmp_path / "av_tiny"


def test_main_av_full_without_fixtures_says_full(tmp_path, capsys):
    assert A.main(["--set", "av_full", "--fixtures", str(tmp_path)]) == 2
    assert "make_fixtures --av --full" in capsys.readouterr().err


def test_score_frames_keeps_raw_scores_per_face():
    det = LipMotionASD()
    frames = [A.FrameInput(i / 25, [FaceTrack(0, BOX, None, 0.0, i / 25)], np.zeros(640, np.float32)) for i in range(3)]
    faces = [A.FaceLabel("target", BOX, [(0.0, 1.0)]), A.FaceLabel("interferer", (0.8, 0.8, 0.1, 0.1), [])]
    s, _ = A.score_frames(det, frames, faces)
    assert s[0].x.shape == (3, 3) and not np.isnan(s[0].x).any()
    assert np.isnan(s[1].x).all()  # Never matched.
