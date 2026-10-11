"""Offline tests for the T-252 latency controller prototype (synthetic traces only)."""
import importlib.util
import json
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "experiments/theory/T-252/latency_controller.py"
_spec = importlib.util.spec_from_file_location("t252_lc", _p)
lc = importlib.util.module_from_spec(_spec)
import sys

sys.modules["t252_lc"] = lc
_spec.loader.exec_module(lc)

SEEDS = [252, 253, 254]


def _case(scenario, policy):
    return lc.run_case(scenario, policy, SEEDS)


def test_trace_deterministic():
    assert lc.make_trace("combined", 7) == lc.make_trace("combined", 7)
    assert lc.make_trace("combined", 7) != lc.make_trace("combined", 8)


def test_m2e_splits_algorithmic_from_slack():
    d = lc.m2e(lc.LADDER[0])
    assert d["algorithmic"] > 2.0 and abs(d["total"] - (d["algorithmic"] + d["slack"])) < 1e-9


def test_remote_loss_hurts_fixed_not_local_tier():
    s = lc.make_trace("remote_loss", 1)
    lost = next(x for x in s if x["lost"])
    assert lc.window_latency(lc.LADDER[0], lost) == float("inf")
    assert lc.window_latency(lc.LADDER[2], lost) < lc.LADDER[2].slack


def test_adaptive_beats_fixed_on_misses_under_stress():
    for sc in ("overload", "remote_loss", "combined"):
        a = _case(sc, "adaptive")["mean"]["miss_rate"]
        assert a < 0.3 * _case(sc, "fixed1s")["mean"]["miss_rate"]
        assert a < 0.3 * _case(sc, "fixed2s")["mean"]["miss_rate"]


def test_hysteresis_reduces_churn_vs_naive():
    for sc in ("steady", "flap", "overload"):
        assert _case(sc, "adaptive")["mean"]["switches_per_min"] < 0.2 * _case(sc, "naive")["mean"]["switches_per_min"]


def test_adaptive_meets_qos_in_all_scenarios():
    for sc in lc.SCENARIOS:
        assert lc.run_case(sc, "adaptive", [252 + i for i in range(20)])["qos_pass_share"] >= 0.8, sc


def test_rollback_pins_safe_tier():
    p = lc.Adaptive()
    p.reset()
    p.rollback()
    assert all(p.pick(t) == lc.SAFE for t in (0, 50, 1e6))


def test_checkpoint_resume_skips_complete_and_reruns_on_hash_change(tmp_path):
    a = lc.run(tmp_path, 2)
    assert a["cases_ran"] == a["cases_planned"] and a["cases_skipped_resume"] == 0
    b = lc.run(tmp_path, 2)
    assert b["cases_ran"] == 0 and b["cases_skipped_resume"] == b["cases_planned"]
    assert a["results"] == b["results"]
    c = lc.run(tmp_path, 3)  # different seeds -> different config hash -> recompute
    assert c["cases_ran"] == c["cases_planned"]
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "complete"
