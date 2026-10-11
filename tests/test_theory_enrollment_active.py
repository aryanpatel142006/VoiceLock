import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

P = Path(__file__).resolve().parents[1] / "experiments/theory/T-250/enroll_sim.py"
spec = importlib.util.spec_from_file_location("enroll_sim", P)
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def test_store_requires_confirmation_and_reverts():
    s = sim.ProfileStore()
    a, b = np.eye(4)[0], np.eye(4)[1]
    with pytest.raises(ValueError):
        s.add_version("x", a, confirmed=False)
    s.add_version("x", a, True)
    s.add_version("x", b, True)
    assert np.allclose(s.swap("x"), b)
    assert s.revert("x") and np.allclose(s.swap("x"), a)
    assert not s.revert("x")


def test_robust_aggregation_rejects_outlier():
    rng = np.random.default_rng(0)
    c = sim.unit(rng.normal(size=sim.DIM))
    E = np.stack([sim.clip(rng, c, 0.3) for _ in range(5)] + [sim.unit(rng.normal(size=sim.DIM))])
    assert sim.agg_robust(E) @ c > sim.agg_mean(E) @ c


def test_run_deterministic_and_resumes(tmp_path):
    out = tmp_path / "r.json"
    assert sim.main(["--seed", "3", "--out", str(out)]) == 0
    d1 = json.loads(out.read_text())
    assert d1["status"] == "complete" and d1["cases"] > 0
    m = out.stat().st_mtime_ns
    assert sim.main(["--seed", "3", "--out", str(out)]) == 0  # skipped
    assert out.stat().st_mtime_ns == m
    e = d1["experiments"]
    assert e["E3_drift_auc_after_10_steps"]["static"] < e["E3_drift_auc_after_10_steps"]["confirmed_ema"]
    assert e["E5_negative_profile_auc_similar_interferer"]["with_neg"] > e["E5_negative_profile_auc_similar_interferer"]["no_neg"]
    assert e["E4_open_vs_closed_set"]["open_set_false_accept"] < 1.0


def test_cap_stops():
    assert sim.run(0, max_seconds=0.0)["status"] == "stopped_cap"
