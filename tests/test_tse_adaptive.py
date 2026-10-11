"""T-089: adaptive hop on overruns, health recovery after N clean windows, compute headroom."""

import numpy as np
import pytest

from hearing.fusion.tiers import TierLadder, compute_headroom
from hearing.tse.stage import HOP_DOWN_AFTER, RECOVER_WINDOWS, FocusHQ, _Window
from hearing.types import AudioChunk, StageContext

SR = 16000
HOP = 160


def _stage(**kw) -> FocusHQ:
    st = FocusHQ(candidate="identity", window_s=1.0, hop_s=0.5, threaded=True, **kw)
    st.load()
    return st


def _settle(st: FocusHQ, *outcomes: str, ms: float = 100.0) -> None:
    """Resolve windows in order: "" = clean, else the skip/late reason."""
    for reason in outcomes:
        w = _Window(st._next_k, "skip" if reason not in ("", "late") else "ok", reason)
        st._next_k += 1
        w.ms = ms
        st._recent.append(w)
        st._account(w)


def test_hop_widens_on_two_overruns_in_four_and_the_stage_stays_healthy():
    st = _stage()
    assert [h / SR for h in st.hops] == [0.5, 0.75, 1.0]
    _settle(st, "", "")
    assert st.health()["ok"] and st.latency_ms == pytest.approx(1500.0)
    _settle(st, "overrun")  # one overrun: tolerated, the neighbours cover it
    assert st.health()["ok"] and st.hop_s == pytest.approx(0.5)
    _settle(st, "", "overrun")  # 2 of the last 4: widen
    h = st.health()
    assert h["ok"] and h["hop_s"] == pytest.approx(0.75) and h["hop_changes"] == 1
    assert h["delay_ms"] == pytest.approx(1750.0)
    _settle(st, "late", "", "overrun")
    assert st.hop_s == pytest.approx(1.0) and st.health()["ok"]
    _settle(st, "overrun")  # at the widest hop: 2 of 4 still plays (one second of mix each)
    assert st.health()["ok"]
    _settle(st, "overrun")
    assert st.health()["ok"]
    _settle(st, "overrun")  # 3 of 4: unhealthy, the ladder drops a tier
    h = st.health()
    assert not h["ok"] and h["reason"] == "overrun"


def test_health_recovers_after_exactly_n_clean_windows():
    assert RECOVER_WINDOWS == 4
    st = _stage(adapt_hop=False)
    _settle(st, "", "no lips")
    assert st.health()["reason"] == "no lips"
    for i in range(RECOVER_WINDOWS - 1):
        _settle(st, "")
        assert not st.health()["ok"], i
    _settle(st, "")
    assert st.health()["ok"] and st.health()["reason"] == ""
    _settle(st, "overrun", "overrun", "overrun")  # no wider hop: unhealthy, 4 clean windows to recover
    assert not st.health()["ok"]
    _settle(st, "", "", "")
    assert not st.health()["ok"]
    _settle(st, "")
    assert st.health()["ok"]
    _settle(st, "enrolling")  # T-088: the enrollment arriving is not a failure to outlive
    assert st.health()["reason"] == "enrolling"
    _settle(st, "")
    assert st.health()["ok"]


def test_hop_steps_back_after_ten_clean_windows_that_fit():
    st = _stage()
    _settle(st, "", "overrun", "overrun")
    assert st.hop_s == pytest.approx(0.75)
    _settle(st, *[""] * (HOP_DOWN_AFTER - 1), ms=300.0)
    assert st.hop_s == pytest.approx(0.75)
    _settle(st, "", ms=300.0)  # 10 clean, p95 300 ms < 0.8 x 500 ms
    assert st.hop_s == pytest.approx(0.5)
    st2 = _stage()
    _settle(st2, "", "overrun", "overrun")
    _settle(st2, *[""] * (2 * HOP_DOWN_AFTER), ms=450.0)  # would not fit in 0.5 s with margin
    assert st2.hop_s == pytest.approx(0.75)


def test_utilisation_is_compute_over_hop_and_headroom_takes_the_slowest():
    st = _stage()
    _settle(st, "", "", ms=200.0)
    h = st.health()
    assert h["compute_p50_ms"] == pytest.approx(200.0) and h["utilisation"] == pytest.approx(0.4)
    assert compute_headroom({"a": [h], "b": [{"utilisation": 0.7}], "c": [{"ok": True}]}) == pytest.approx(0.3)
    assert compute_headroom({"gate": [{"ok": True}]}) is None


def test_hop_change_dips_and_keeps_the_stream_continuous():
    """A real stream: the delay grows by 0.25 s through a dip, no clicks, nothing lost."""
    st = _stage()
    ctx = StageContext.default()
    x = (0.3 * np.sin(2 * np.pi * 220 * np.arange(4 * SR) / SR)).astype(np.float32)
    ys = []
    for i in range(0, len(x), HOP):
        if i == 2 * SR:
            _settle(st, "overrun", "overrun")  # force a hop change mid-stream
            assert st._jump == round(0.25 * SR)
        ys.append(st.process(AudioChunk(x[i : i + HOP], SR, i // HOP, i / SR), ctx).samples)
    st.close()
    y = np.concatenate(ys)
    assert st.hop_s == pytest.approx(0.75) and st._jump == 0 and st._stall == 0
    assert np.max(np.abs(np.diff(y))) < 0.1  # ramps only (a 220 Hz sine moves < 0.026 per sample)
    # After the jump the output is the input 1.75 s late (identity candidate).
    tail = y[-SR // 2 :]
    ref = x[len(x) - SR // 2 - st.delay : len(x) - st.delay]
    np.testing.assert_allclose(tail, ref, atol=1e-4)


def test_ladder_name_lists_built_tiers_and_status_carries_headroom():
    cfg = {"tiers": [
        {"name": "focus_hq", "pipeline": "tse(candidate=\"identity\")", "devices": ["cuda"]},
        {"name": "enrolled", "pipeline": "tse(candidate=\"identity\", window_s=0.2, hop_s=0.1)",
         "needs": ["healthy"]},
        {"name": "audio-only", "pipeline": "passthrough"},
    ]}
    ladder = TierLadder(cfg)
    built = [n for n in ("focus_hq", "enrolled", "audio-only") if n in ladder.built]
    assert ladder.name == "tiers(" + ",".join(built) + ")"
    ctx = StageContext.default()
    x = np.zeros(SR, np.float32)
    for i in range(0, len(x), HOP):
        ladder.process(AudioChunk(x[i : i + HOP], SR, i // HOP, i / SR), ctx)
    ladder.close()
    status = ctx.telemetry.extras["tiers"]
    assert "compute_headroom" in status
    assert status["compute_headroom"] is None or 0.0 <= status["compute_headroom"] <= 1.0


def test_the_cold_first_window_neither_widens_the_hop_nor_counts_in_compute():
    st = _stage()
    w = _Window(0, "pending")
    w.cold = True
    st._windows[0] = w
    assert st._plan(1, StageContext.default()).reason == "warming up"
    w.status, w.ms = "ok", 7900.0
    st._account(w)
    _settle(st, "", ms=250.0)
    h = st.health()
    assert h["hop_s"] == pytest.approx(0.5) and h["compute_p95_ms"] == pytest.approx(250.0)


def test_pinned_hop_never_changes_and_fails_like_the_widest_rung():
    """T-135: pin_hop() keeps the current hop (no widening, no stepping back); overruns then make the
    stage unhealthy as at the widest hop."""
    st = _stage()
    _settle(st, "", "overrun", "overrun")
    assert st.hop_s == pytest.approx(0.75)
    st.pin_hop()
    _settle(st, *[""] * (HOP_DOWN_AFTER + 2), ms=100.0)  # would step back unpinned
    assert st.hop_s == pytest.approx(0.75) and st.hop_changes == 1
    _settle(st, "overrun", "overrun", "overrun")
    h = st.health()
    assert st.hop_s == pytest.approx(0.75) and not h["ok"] and h["hop_pinned"]
    st.reset()
    assert st.hop_pinned  # a re-primed tier stays pinned
