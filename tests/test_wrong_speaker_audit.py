import numpy as np

from eval import wrong_speaker_audit as wsa


def _row(**kw):
    row = {"k": 0, "margin": 1.0, "hold": 0.4, "held": False, "vetoed": False, "selected": 0, "oracle": 0,
           "follow": None, "target_active": True, "wrong": True, "dom_s": 0.2,
           "dom_stream_s": 0.2}
    row.update(kw)
    row["cause"] = wsa.cause(row) if row["wrong"] else None
    return row


def test_cause_buckets():
    assert wsa.cause(_row()) == "separator"
    assert wsa.cause(_row(dom_stream_s=0.0)) == "mix_back"
    assert wsa.cause(_row(selected=1)) == "high_margin"
    assert wsa.cause(_row(selected=1, margin=None)) == "high_margin"
    assert wsa.cause(_row(selected=1, margin=0.1)) == "low_margin"
    assert wsa.cause(_row(selected=1, margin=0.1, held=True)) == "low_margin_held"
    assert wsa.cause(_row(selected=1, vetoed=True)) == "vetoed"


def test_summarize_counts_and_sticky():
    rows = [_row(selected=1, follow=0), _row(selected=1, margin=0.1, held=True, follow=1), _row(),
            _row(wrong=False, selected=1)]
    s = wsa.summarize(rows)
    assert s["wrong"] == 3 and s["wrong_picks"] == 2
    assert s["causes"]["high_margin"] == 1 and s["causes"]["low_margin_held"] == 1 and s["causes"]["separator"] == 1
    assert (s["above_hold"], s["below_hold"], s["held_fired"]) == (2, 1, 1)
    assert s["sticky_fixes"] == 1
    assert s["pick_errors_all"] == 3


def test_window_of_frames_maps_streamed_hops():
    win, hop = 16000, 8000
    owner = wsa.window_of_frames(10, win, hop)  # 2 s of 200 ms frames
    assert list(owner[:3]) == [0, 0, 0]  # [0, 0.6 s): window 0 owns [0, 1.0 s)
    assert owner[5] == 1 and owner[9] == 2  # 1.1 s -> window 1 ([1.0, 1.5)), 1.9 s -> window 2


def test_other_dominant_flags_the_other_voice():
    rng = np.random.default_rng(0)
    n = wsa.LS_FRAME * 4
    t, o, nz = rng.standard_normal(n) * 0.1, rng.standard_normal(n) * 0.1, rng.standard_normal(n) * 0.001
    x = t + o + nz
    y = np.concatenate([t[: n // 2], o[n // 2 :]])  # right talker for 2 frames, then the other one
    assert list(wsa.other_dominant(y, x, t, o, nz)) == [False, False, True, True]


def test_true_dropout_counts_silent_target_runs():
    rng = np.random.default_rng(1)
    t = rng.standard_normal(16000) * 0.1
    y = t.copy()
    y[8000:9600] = 0.0  # 100 ms of silence while the target talks (5 frames of 20 ms)
    assert abs(wsa.true_dropout_s(y, t) - 0.1) < 1e-9
    assert wsa.true_dropout_s(t, t) == 0.0


def test_kill_rule_and_compare_line():
    def res(p, gated=3, down=0.4):
        return {"wrong_speaker_pct": p, "gated_windows": gated, "target_muted_s": {"target_down_10dB_s": down},
                "learned_windows": 5}

    assert not wsa.kill_rule(res(10.0), res(10.9))
    assert wsa.kill_rule(res(10.0), res(11.1))
    assert not wsa.kill_rule(res(10.0), res(2.0))
    assert "+1.10 pt" in wsa.compare_line(res(10.0), res(11.1))
