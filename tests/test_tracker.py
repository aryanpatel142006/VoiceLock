"""Synthetic box sequences for :class:`hearing.vision.tracker.IouTracker`."""

from __future__ import annotations

import time

import numpy as np
import pytest
from conftest import scaled

from hearing.types import FaceTrack
from hearing.vision import Detection, IouTracker, iou_matrix
from hearing.vision.tracker import _greedy_assign

FPS = 25
DT = 1.0 / FPS
W, H = 0.15, 0.25  # Typical face box at 640×360, normalised.


def det(x: float, y: float, w: float = W, h: float = H, **kw) -> Detection:
    return Detection(box=(x, y, w, h), **kw)


def run(tracker: IouTracker, frames: list[list[Detection]], t0: float = 0.0) -> list[list[FaceTrack]]:
    return [tracker.update(dets, t0 + i * DT) for i, dets in enumerate(frames)]


def ids(tracks: list[FaceTrack]) -> set[int]:
    return {t.track_id for t in tracks}


def nearest(tracks: list[FaceTrack], x: float, y: float) -> FaceTrack:
    return min(tracks, key=lambda t: (t.box[0] - x) ** 2 + (t.box[1] - y) ** 2)


@pytest.fixture(params=["hungarian", "greedy"])
def matcher(request) -> bool:
    if request.param == "hungarian":
        pytest.importorskip("scipy")
        return True
    return False


# --------------------------------------------------------------------------- iou
def test_iou_matrix_basic():
    a = np.array([[0.0, 0.0, 0.2, 0.2], [0.5, 0.5, 0.1, 0.1]])
    b = np.array([[0.0, 0.0, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]])
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1.0)
    assert m[0, 1] == pytest.approx(0.01 / 0.07)
    assert m[0, 2] == 0.0
    assert m[1, 2] == 0.0
    assert iou_matrix(np.zeros((0, 4)), b).shape == (0, 3)
    assert iou_matrix(a, np.zeros((0, 4))).shape == (2, 0)


def test_iou_matrix_degenerate_boxes_do_not_nan():
    a = np.array([[0.1, 0.1, 0.0, 0.0]])
    assert iou_matrix(a, a)[0, 0] == 0.0


def test_greedy_assign_respects_threshold_and_uniqueness():
    iou = np.array([[0.9, 0.8], [0.85, 0.1]])
    assert sorted(_greedy_assign(iou, 0.3)) == [(0, 0)]  # row 1 only has 0.85 for col 0 (taken) / 0.1.
    assert _greedy_assign(np.zeros((2, 2)), 0.3) == []


# ----------------------------------------------------------------------- basics
def test_detection_defaults():
    d = Detection((0.1, 0.2, 0.3, 0.4))
    assert d.score == 1.0 and d.landmarks is None


def test_min_hits_suppresses_single_frame_flicker(matcher):
    tr = IouTracker(min_hits=2, use_hungarian=matcher)
    assert tr.update([det(0.1, 0.1)], 0.0) == []  # Tentative: not reported yet.
    assert tr.update([], DT) == []  # Vanished before confirmation: dropped.
    assert tr.update([det(0.1, 0.1)], 2 * DT) == []  # New tentative track.
    out = tr.update([det(0.1, 0.1)], 3 * DT)
    assert len(out) == 1
    assert out[0].p_speaking == 0.0
    assert out[0].last_seen_ts == pytest.approx(3 * DT)


def test_face_track_fields_and_landmark_passthrough(matcher):
    tr = IouTracker(use_hungarian=matcher)
    lm = np.zeros((468, 3), dtype=np.float32)
    tr.update([det(0.3, 0.3, score=0.9, landmarks=None)], 0.0)
    out = tr.update([det(0.3, 0.3, score=0.95, landmarks=lm)], DT)
    assert isinstance(out[0], FaceTrack)
    assert out[0].landmarks is lm
    assert isinstance(out[0].box, tuple) and len(out[0].box) == 4
    assert all(isinstance(v, float) for v in out[0].box)


def test_box_smoothing_follows_motion_without_overshoot(matcher):
    tr = IouTracker(smoothing_alpha=0.5, use_hungarian=matcher)
    tr.update([det(0.10, 0.2)], 0.0)
    out = tr.update([det(0.14, 0.2)], DT)
    # EMA with alpha 0.5: halfway between the previous (0.10) and the new (0.14) box.
    assert out[0].box[0] == pytest.approx(0.12)
    out = tr.update([det(0.14, 0.2)], 2 * DT)
    assert out[0].box[0] == pytest.approx(0.13)
    assert out[0].box[2] == pytest.approx(W) and out[0].box[3] == pytest.approx(H)


# ------------------------------------------------------------------- stability
def test_two_faces_moving_for_two_seconds_keep_ids(matcher):
    tr = IouTracker(use_hungarian=matcher)
    n = 2 * FPS
    frames = []
    for i in range(n):
        # Face A drifts right, face B drifts down; both ~1 px/frame at 640×360.
        frames.append([det(0.10 + 0.002 * i, 0.30), det(0.60, 0.10 + 0.003 * i)])
    outs = run(tr, frames)
    first = outs[1]
    assert len(first) == 2
    id_a = nearest(first, 0.10, 0.30).track_id
    id_b = nearest(first, 0.60, 0.10).track_id
    assert id_a != id_b
    for i, out in enumerate(outs[1:], start=1):
        assert ids(out) == {id_a, id_b}, f"frame {i}: {ids(out)}"
        assert nearest(out, 0.10 + 0.002 * i, 0.30).track_id == id_a
        assert nearest(out, 0.60, 0.10 + 0.003 * i).track_id == id_b
    assert tr._next_id == 3  # Exactly two ids were ever allocated.


def test_half_second_disappearance_keeps_id_and_holds_box(matcher):
    tr = IouTracker(max_missing_s=0.7, use_hungarian=matcher)
    frames = [[det(0.4, 0.3)] for _ in range(10)]
    outs = run(tr, frames)
    tid = outs[-1][0].track_id
    held_box = outs[-1][0].box
    last_seen = outs[-1][0].last_seen_ts
    gap = 12  # 0.48 s at 25 fps.
    for k in range(gap):
        out = tr.update([], (10 + k) * DT)
        assert ids(out) == {tid}, "missing track must still be reported inside max_missing_s"
        assert out[0].box == held_box
        assert out[0].last_seen_ts == pytest.approx(last_seen)
    out = tr.update([det(0.42, 0.31)], (10 + gap) * DT)  # Back, slightly moved.
    assert ids(out) == {tid}
    assert out[0].last_seen_ts == pytest.approx((10 + gap) * DT)
    assert out[0].box != held_box


def test_gap_longer_than_max_missing_yields_new_id(matcher):
    # reattach_s=0: no memory of dropped tracks (T-119 re-attaches by default; tests/test_tracker_session.py)
    tr = IouTracker(max_missing_s=0.7, min_hits=2, use_hungarian=matcher, reattach_s=0.0)
    outs = run(tr, [[det(0.4, 0.3)] for _ in range(5)])
    old = outs[-1][0].track_id
    gap = int(0.9 * FPS)  # 22 frames = 0.88 s > 0.7 s.
    saw_dropout = False
    for k in range(gap):
        out = tr.update([], (5 + k) * DT)
        if not out:
            saw_dropout = True
    assert saw_dropout, "track should have expired during the gap"
    tr.update([det(0.4, 0.3)], (5 + gap) * DT)
    out = tr.update([det(0.4, 0.3)], (6 + gap) * DT)
    assert len(out) == 1 and out[0].track_id != old


def test_track_survives_exactly_until_max_missing(matcher):
    tr = IouTracker(max_missing_s=0.2, use_hungarian=matcher)
    tr.update([det(0.4, 0.3)], 0.0)
    tr.update([det(0.4, 0.3)], 0.04)
    assert ids(tr.update([], 0.24)) == {1}  # dt == max_missing_s: still held.
    assert tr.update([], 0.2401) == []


def test_crossing_faces_do_not_swap_ids(matcher):
    tr = IouTracker(use_hungarian=matcher)
    n = 2 * FPS
    y = 0.3
    frames = []
    for i in range(n):
        xa = 0.10 + 0.012 * i  # Left → right.
        xb = 0.70 - 0.012 * i  # Right → left; they meet around frame 25.
        frames.append([det(xa, y), det(xb, y)])
    outs = run(tr, frames)
    id_a = nearest(outs[1], 0.10 + 0.012, y).track_id
    id_b = nearest(outs[1], 0.70 - 0.012, y).track_id
    assert id_a != id_b
    for i, out in enumerate(outs[1:], start=1):
        assert ids(out) == {id_a, id_b}, f"frame {i}: ids {ids(out)}"
    # After the crossing, A is on the right and B on the left: no swap.
    final = outs[-1]
    xa = 0.10 + 0.012 * (n - 1)
    xb = 0.70 - 0.012 * (n - 1)
    assert nearest(final, xa, y).track_id == id_a
    assert nearest(final, xb, y).track_id == id_b


def test_reset_clears_ids(matcher):
    tr = IouTracker(use_hungarian=matcher)
    run(tr, [[det(0.1, 0.1), det(0.6, 0.6)] for _ in range(3)])
    assert tr._next_id == 3
    tr.reset()
    assert tr.tracks == []
    out = run(tr, [[det(0.1, 0.1)] for _ in range(2)])[-1]
    assert ids(out) == {1}


def test_out_of_order_timestamp_is_tolerated(matcher):
    tr = IouTracker(use_hungarian=matcher)
    tr.update([det(0.1, 0.1)], 1.0)
    out = tr.update([det(0.1, 0.1)], 0.5)  # Earlier than the last frame.
    assert ids(out) == {1}
    assert out[0].last_seen_ts == pytest.approx(1.0)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        IouTracker(iou_threshold=0.0)
    with pytest.raises(ValueError):
        IouTracker(min_hits=0)
    with pytest.raises(ValueError):
        IouTracker(smoothing_alpha=1.5)
    with pytest.raises(ValueError):
        IouTracker(max_missing_s=-1)


# -------------------------------------------------------------------- timing
def test_1000_frames_three_faces_under_budget(matcher):
    tr = IouTracker(use_hungarian=matcher)
    rng = np.random.default_rng(0)
    frames = []
    for i in range(1000):
        jitter = rng.normal(0.0, 0.002, size=(3, 2))
        frames.append(
            [
                det(0.10 + 0.0003 * i + jitter[0, 0], 0.30 + jitter[0, 1]),
                det(0.45 + jitter[1, 0], 0.20 + 0.0002 * i + jitter[1, 1]),
                det(0.75 - 0.0002 * i + jitter[2, 0], 0.40 + jitter[2, 1]),
            ]
        )
    # CPU time, best of 5 passes: wall clock (and a single pass) flakes when other verifies share the machine
    elapsed = float("inf")
    for _ in range(5):
        tr = IouTracker(use_hungarian=matcher)
        t0 = time.process_time()
        for i, dets in enumerate(frames):
            out = tr.update(dets, i * DT)
        elapsed = min(elapsed, time.process_time() - t0)
    per_frame_ms = 1000.0 * elapsed / len(frames)
    label = "hungarian" if matcher else "greedy"
    print(f"\nIouTracker[{label}]: 1000 frames × 3 faces in {elapsed * 1000:.1f} ms "
          f"({per_frame_ms * 1000:.1f} µs/frame)")
    assert len(out) == 3 and tr._next_id == 4
    assert elapsed < scaled(1.0)
    assert per_frame_ms < scaled(0.5), f"{per_frame_ms:.3f} ms per frame"


def test_redetection_after_a_miss_keeps_its_id_when_iou_fails():
    """T-107: a face that moved during a short miss (no IoU overlap left) keeps its id by centre."""
    tr = IouTracker(min_hits=1)
    tr.update([Detection((0.10, 0.3, 0.10, 0.2)), Detection((0.70, 0.3, 0.10, 0.2))], 0.0)
    tr.update([Detection((0.70, 0.3, 0.10, 0.2))], 0.04)  # face 1 missed
    out = tr.update([Detection((0.17, 0.3, 0.10, 0.2)), Detection((0.70, 0.3, 0.10, 0.2))], 0.4)
    assert next(t.track_id for t in out if t.box[0] < 0.5) == 1
    assert {t.track_id for t in out} == {1, 2}
    # without the centre pass it would have been a new id
    tr = IouTracker(min_hits=1, centre_gate=0.0)
    tr.update([Detection((0.10, 0.3, 0.10, 0.2))], 0.0)
    fresh = [t.track_id for t in tr.update([Detection((0.17, 0.3, 0.10, 0.2))], 0.4) if t.last_seen_ts == 0.4]
    assert fresh == [2]


def test_centre_pass_does_not_steal_a_far_or_differently_sized_face():
    tr = IouTracker(min_hits=1)
    tr.update([Detection((0.10, 0.3, 0.10, 0.2))], 0.0)
    out = tr.update([Detection((0.40, 0.3, 0.10, 0.2)), Detection((0.12, 0.3, 0.30, 0.5))], 0.2)
    assert 1 not in {t.track_id for t in out if t.last_seen_ts == 0.2}
