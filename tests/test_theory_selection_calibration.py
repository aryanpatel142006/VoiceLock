"""T-249: offline checks of the selection-calibration prototype (synthetic traces, numpy only)."""
import importlib.util
import json
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[1] / "experiments/theory/T-249/selection_calibration.py"
_spec = importlib.util.spec_from_file_location("t249_sc", _p)
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

CFG = {**sc.DEFAULT, "n_groups": 8}


def test_group_holdout_is_disjoint_and_capped():
    rows = sc.make_dataset({**CFG, "max_cases": 20})
    assert len(rows) <= 20
    dev = {r["group"] for r in rows if r["split"] == "dev"}
    hold = {r["group"] for r in rows if r["split"] == "holdout"}
    assert dev and hold and not dev & hold


def test_isotonic_monotone_and_platt_reduces_ece():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.5
    s = np.where(y, 0.6, 0.2) + 0.15 * rng.standard_normal(4000)
    iso, _ = sc.fit_isotonic(s, y.astype(float))
    g = iso(np.linspace(-0.2, 1, 50))
    assert np.all(np.diff(g) >= -1e-12)
    platt, _ = sc.fit_platt(s, y.astype(float))
    assert sc.ece(platt(s), y.astype(float))[0] < sc.ece(np.clip(s, 0, 1), y.astype(float))[0]


def test_hmm_bridges_pause_but_dwell_rejects_short_burst():
    pol = sc.DEFAULT["policy"]
    burst = np.r_[np.full(30, 0.05), np.full(2, 0.95), np.full(30, 0.05)]
    assert not sc.run_policy(burst, pol, hmm=False).any()
    assert not sc.run_policy(np.full(40, 0.02), pol, hmm=True).any()
    assert sc.run_policy(np.full(40, 0.95), pol, hmm=True)[-1]


def test_cli_deterministic_and_resumes(tmp_path, capsys):
    assert sc.main(["--out", str(tmp_path)]) == 0
    a = (tmp_path / "result.json").read_text()
    assert sc.main(["--out", str(tmp_path)]) == 0
    assert "resume" in capsys.readouterr().out
    assert sc.main(["--out", str(tmp_path), "--force"]) == 0
    assert (tmp_path / "result.json").read_text() == a
    r = json.loads(a)
    assert r["status"] == "complete" and r["counts"]["groups_dev"] == r["counts"]["groups_holdout"]
