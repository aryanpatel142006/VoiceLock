import itertools
from pathlib import Path

import pytest

from eval.target_eval import evaluate, main
from hearing.types import FaceTrack, TargetState
from hearing.vision.target import TargetSelector

SCENES = Path(__file__).resolve().parent.parent / "eval" / "scenes"
FPS = 25
DT = 1 / FPS


def face(track_id, cx, cy=0.5, w=0.12, h=0.2, p=1.0, ts=0.0):
    return FaceTrack(track_id, (cx - w / 2, cy - h / 2, w, h), None, p, ts)


def run(sel, frames, t0=0.0):
    """frames: list of callables t -> list[FaceTrack]; returns [(t, state)]."""
    out = []
    for i, make in enumerate(frames):
        t = t0 + i * DT
        out.append((t, sel.update(make(t), t)))
    return out


def first_time(states, track_id):
    return next((t for t, s in states if s.track_id == track_id), None)


def test_acquires_central_face_and_reports_state():
    sel = TargetSelector()
    states = run(sel, [lambda t: [face(1, 0.5), face(2, 0.85)]] * 10)
    _t, s = states[-1]
    assert isinstance(s, TargetState)
    assert s.track_id == 1 and not s.locked
    assert 0.9 <= s.confidence <= 1.0
    assert s.since_ts == pytest.approx(first_time(states, 1))
    assert first_time(states, 1) <= 0.12


def test_dwell_no_switch_before_dwell_then_switch_within_1s():
    sel = TargetSelector()
    run(sel, [lambda t: [face(1, 0.5), face(2, 0.8)]] * 25)  # A is target
    # Head turns instantly at t=1: B now centred, A off to the side.
    states = run(sel, [lambda t: [face(1, 0.2), face(2, 0.5)]] * 30, t0=1.0)
    switch = first_time(states, 2)
    assert switch is not None
    assert switch - 1.0 >= sel.dwell_s - 1e-6
    assert switch - 1.0 < 1.0
    assert all(s.track_id == 1 for t, s in states if t < switch)
    # Confidence in A drops while B dwells, then B is chosen at `switch`.
    mid = [s.confidence for t, s in states if 1.0 < t < switch]
    assert mid[-1] < mid[0]
    assert states[-1][1].since_ts == pytest.approx(switch)


def test_hysteresis_jitter_near_boundary_does_not_flip():
    sel = TargetSelector()
    run(sel, [lambda t: [face(1, 0.5), face(2, 0.8)]] * 10)
    # A drifts to 0.58 (d=0.08), B jitters around 0.46: sometimes better by more
    # than the margin, never for the whole dwell; and ties stay with A.
    jitter = [0.43, 0.49, 0.52, 0.45, 0.50, 0.42, 0.53, 0.48]

    def frames(i):
        return lambda t: [face(1, 0.58), face(2, jitter[i % len(jitter)])]

    states = run(sel, [frames(i) for i in range(100)], t0=0.4)
    assert {s.track_id for _, s in states} == {1}


def test_hold_through_half_second_gap_then_drop_after_hold():
    sel = TargetSelector()
    chosen = run(sel, [lambda t: [face(1, 0.5), face(2, 0.8)]] * 25)[-1][1].since_ts
    gap = run(sel, [lambda t: [face(2, 0.8)]] * 12, t0=1.0)  # 0.48 s without A
    assert all(s.track_id == 1 for _, s in gap)
    confs = [s.confidence for _, s in gap]
    assert all(b < a for a, b in itertools.pairwise(confs))
    back = run(sel, [lambda t: [face(1, 0.5), face(2, 0.8)]] * 5, t0=1.48)
    assert all(s.track_id == 1 for _, s in back)
    assert back[-1][1].confidence > 0.9
    assert back[-1][1].since_ts == chosen  # same target, never re-chosen

    # A long absence: after hold_s the target is released to the next candidate.
    last_seen = back[-1][0]
    gone = run(sel, [lambda t: [face(2, 0.8)]] * 80, t0=last_seen + DT)
    assert all(s.track_id == 1 for t, s in gone if t < last_seen + sel.hold_s - 1e-6)
    released = first_time(gone, 2)
    assert released - last_seen == pytest.approx(sel.hold_s + sel.acquire_s, abs=DT + 1e-6)


def test_target_drops_to_none_after_hold_with_no_faces():
    sel = TargetSelector()
    run(sel, [lambda t: [face(1, 0.5)]] * 10)
    states = run(sel, [lambda t: []] * 60, t0=0.4)
    assert states[10][1].track_id == 1
    assert states[-1][1] == TargetState(None, 0.0, False, pytest.approx(0.36 + sel.hold_s))


def test_lock_overrides_centre_of_view_until_unlock():
    sel = TargetSelector()
    run(sel, [lambda t: [face(1, 0.5), face(2, 0.85)]] * 10)
    s = sel.lock(2)
    assert s.track_id == 2 and s.locked
    states = run(sel, [lambda t: [face(1, 0.5), face(2, 0.85)]] * 50, t0=0.4)
    assert all(s.track_id == 2 and s.locked for _, s in states)
    sel.unlock()
    after = run(sel, [lambda t: [face(1, 0.5), face(2, 0.85)]] * 25, t0=2.4)
    assert not after[0][1].locked
    assert first_time(after, 1) is not None and first_time(after, 1) - 2.4 < 1.0


def _locked_left(sel):
    """Two faces, the locked one on the left and the other MORE central (free selection would pick 2)."""
    run(sel, [lambda t: [face(1, 0.2), face(2, 0.6)]] * 10)
    sel.lock(1)
    return run(sel, [lambda t: [face(1, 0.2), face(2, 0.6)]] * 10, t0=0.4)


@pytest.mark.parametrize("gap_s", [1.0, 5.0])
def test_lock_survives_track_loss_and_never_switches_to_the_more_central_face(gap_s):
    sel = TargetSelector()
    _locked_left(sel)
    gone = run(sel, [lambda t: [face(2, 0.6)]] * int(gap_s * FPS), t0=0.8)
    assert all(s.track_id == 1 and s.locked for _, s in gone)  # reacquiring, never face 2
    assert sel.missing_s == pytest.approx(gap_s, abs=1e-6)
    assert 0.0 < gone[-1][1].confidence < 1.0
    back = run(sel, [lambda t: [face(1, 0.2), face(2, 0.6)]] * 5, t0=0.8 + gap_s)
    assert all(s.track_id == 1 and s.locked and s.confidence == 1.0 for _, s in back)
    assert sel.missing_s == 0.0


def test_lock_reattaches_to_new_track_id_at_the_same_position_without_a_switch():
    sel = TargetSelector()
    before = _locked_left(sel)[-1][1]
    run(sel, [lambda t: [face(2, 0.6)]] * 10, t0=0.8)
    states = run(sel, [lambda t: [face(7, 0.23), face(2, 0.6)]] * 5, t0=1.2)
    assert all(s.track_id == 7 and s.locked for _, s in states)
    assert states[-1][1].switch_seq == before.switch_seq  # same person: enrollment is kept
    assert states[-1][1].since_ts == before.since_ts


def test_lock_never_reattaches_to_a_face_seen_next_to_it_or_far_away():
    sel = TargetSelector()
    _locked_left(sel)
    # the other person (id 2) walks into the locked face's spot, and a stranger appears on the far right
    states = run(sel, [lambda t: [face(2, 0.21), face(9, 0.8)]] * 25, t0=0.8)
    assert all(s.track_id == 1 and s.locked for _, s in states)


def test_lost_lock_keeps_searching_and_never_falls_back_to_free_selection():
    sel = TargetSelector(hold_lock_s=2.0)
    _locked_left(sel)
    gone = run(sel, [lambda t: [face(2, 0.6)]] * 150, t0=0.8)  # 6 s, hold_lock_s = 2
    lost = [s for t, s in gone if t >= 0.8 + 2.0]
    assert lost and all(s.track_id is None and s.locked and s.confidence == 0.0 for s in lost)
    assert sel.lock_lost and sel.missing_s > 2.0
    # the same face returns (new id, same place): the lock re-attaches
    back = run(sel, [lambda t: [face(5, 0.19), face(2, 0.6)]] * 3, t0=6.8)
    assert all(s.track_id == 5 and s.locked for _, s in back) and not sel.lock_lost


def test_lost_lock_is_released_only_by_the_wearer():
    sel = TargetSelector(hold_lock_s=1.0)
    _locked_left(sel)
    run(sel, [lambda t: [face(2, 0.6)]] * 50, t0=0.8)
    assert sel.lock_lost
    s = sel.lock(2)  # the wearer re-locks someone else
    assert s.track_id == 2 and s.locked and not sel.lock_lost
    sel.unlock()  # or unlocks: free selection is back
    assert run(sel, [lambda t: [face(2, 0.6)]] * 10, t0=2.8)[-1][1].track_id == 2


def test_forget_ids_keeps_lock_and_reattaches_by_position_even_to_a_reused_id():
    sel = TargetSelector()
    _locked_left(sel)
    sel.forget_ids()  # vision went dark: tracker ids restart, so id 1 may be anyone now
    assert sel.state.track_id is None and sel.state.locked
    # restarted tracker gives the OTHER face id 1 first: must not be taken for the locked one
    wrong = run(sel, [lambda t: [face(1, 0.6)]] * 5, t0=2.0)
    assert all(s.track_id is None and s.locked for _, s in wrong)
    right = run(sel, [lambda t: [face(1, 0.6), face(2, 0.2)]] * 5, t0=2.2)
    assert all(s.track_id == 2 and s.locked for _, s in right)


def test_forget_ids_without_lock_is_a_reset():
    sel = TargetSelector()
    run(sel, [lambda t: [face(1, 0.5)]] * 10)
    sel.forget_ids()
    assert sel.update([], 1.0).track_id is None and not sel.state.locked


def test_hold_lock_s_is_an_attention_field():
    sel = TargetSelector()
    assert sel.attention["hold_lock_s"] == 10.0
    assert sel.set_attention(hold_lock_s=20)["hold_lock_s"] == 20.0
    with pytest.raises(ValueError):
        sel.set_attention(hold_lock_s=0.1)


def test_silent_faces_are_not_acquired_without_speech_gate_they_are():
    frames = [lambda t: [face(1, 0.5, p=0.0)]] * 10
    assert run(TargetSelector(), frames)[-1][1].track_id is None
    assert run(TargetSelector(speech_memory_s=None), frames)[-1][1].track_id == 1


def test_reset_clears_state():
    sel = TargetSelector()
    run(sel, [lambda t: [face(1, 0.5)]] * 10)
    sel.lock(1)
    sel.reset()
    assert sel.update([], 0.0) == TargetState(None, 0.0, False, 0.0)


def test_scene_set_scores():
    res = evaluate(SCENES)
    per = {p["scene"]: p for p in res["per_scene"]}
    assert len(per) >= 6
    assert res["accuracy"] >= 0.9
    assert all(p["accuracy"] >= 0.9 for p in per.values())
    assert res["missed_switches"] == 0
    assert res["max_switch_latency_s"] < 1.0
    assert res["mean_switch_latency_s"] < 1.0
    assert per["c_occlusion"]["false_switches"] == 0
    assert per["d_moving_target"]["false_switches"] == 0


def test_cli_exit_codes(capsys):
    assert main([str(SCENES)]) == 0
    assert "overall accuracy" in capsys.readouterr().out
    assert main([str(SCENES), "--min-accuracy", "1.01"]) == 1
