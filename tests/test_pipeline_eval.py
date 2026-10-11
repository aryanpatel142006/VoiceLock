"""Scoring of eval/pipeline_eval.py (T-132) on synthetic stems, no weights."""
import numpy as np

from eval import pipeline_eval as pe

SR = pe.SR


def _talk(n, spans, f0, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    x = (0.1 * np.sin(2 * np.pi * f0 * t) * (1 + 0.3 * rng.standard_normal(n))).astype(np.float32)
    m = np.zeros(n, np.float32)
    for a, b in spans:
        m[round(a * SR) : round(b * SR)] = 1
    return x * m


def _scene(intended):
    n = 20 * SR
    a = _talk(n, [(0, 20)], 220, 0)
    b = _talk(n, [(0, 20)], 330, 1)
    nz = (0.001 * np.random.default_rng(2).standard_normal(n)).astype(np.float32)
    return {"stems": {"A": a, "B": b}, "noise": nz, "mix": a + b + nz, "intended": intended,
            "names": [(1.0, 1.6, "A")]}


def test_routing_counts_other_talker_dominant_as_wrong():
    sc = _scene([(1.6, "A")])
    a, b = sc["stems"]["A"], sc["stems"]["B"]
    assert pe.routing(sc, a + 0.1 * b, 2.0)["wrong_pct"] == 0.0
    y = a + 0.1 * b
    y[10 * SR : 12 * SR] = b[10 * SR : 12 * SR]  # 2 s of B
    r = pe.routing(sc, y, 2.0)
    assert r["wrong_s"] == 2.0
    assert r["longest_wrong_s"] == 2.0
    assert r["dominant_flips"] == 2
    assert r["other_leak_db_median"] < -15


def test_routing_after_retarget_old_target_is_wrong():
    sc = _scene([(1.6, "A"), (10.0, "B")])
    a = sc["stems"]["A"]
    r = pe.routing(sc, a, 2.0)  # keeps playing A after the re-target
    assert r["wrong_s"] == 10.0
    assert r["wrong_pct"] == 55.6  # 10 of the 18 target-active seconds


def test_trigger_score_hits_misses_false():
    names = [(7.0, 7.6, "A"), (24.5, 25.1, "B")]
    dets = [{"t": 8.4, "retarget": False}, {"t": 15.0, "retarget": True}]
    s = pe.trigger_score(names, dets)
    assert (s["hits"], s["misses"], s["false_triggers"]) == (1, 1, 1)


def test_detections_from_ticks():
    ticks = [{"t": 1.0, "detections": 0, "events": [], "last": None},
             {"t": 8.0, "detections": 1, "events": [{"event": "heard"}], "last": {"word": "Aryan", "latency_ms": 400}},
             {"t": 25.5, "detections": 2, "events": [{"event": "heard"}, {"event": "retarget"}], "last": {"word": "Aryan"}}]
    d = pe.detections(ticks)
    assert [x["t"] for x in d] == [8.0, 25.5]
    assert [x["retarget"] for x in d] == [False, True]
