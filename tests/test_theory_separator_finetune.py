import importlib.util
import json
from pathlib import Path

_p = Path(__file__).resolve().parents[1] / "experiments/theory/T-247/plan.py"
_s = importlib.util.spec_from_file_location("t247_plan", _p)
plan = importlib.util.module_from_spec(_s)
_s.loader.exec_module(plan)

GOOD = {
    "baseline": {
        "dev_si_sdri": 5,
        "heldout_team_si_sdri": 4,
        "heldout_public_si_sdri": 6,
        "weak_tercile_si_sdri": 2,
        "absent_out_db": -40,
        "wrong_speaker_pct": 15,
    },
    "candidate": {
        "dev_si_sdri": 6,
        "heldout_team_si_sdri": 5,
        "heldout_public_si_sdri": 6,
        "weak_tercile_si_sdri": 2,
        "absent_out_db": -39,
        "wrong_speaker_pct": 12,
    },
}


def test_split_disjoint_deterministic():
    groups = [f"g{i}" for i in range(200)] * 2
    a, b = plan.split(groups, 1), plan.split(groups, 1)
    assert a == b
    assert not set(a["dev"]) & set(a["heldout"])
    assert len(a["dev"]) + len(a["heldout"]) == 200
    assert 15 <= len(a["heldout"]) <= 70


def test_estimate_ordering_and_a100_fit():
    t = {k: plan.estimate(k)["total_gb"] for k in plan.ARMS}
    assert t["selector_only"] < t["adapter"] < t["top_layers"] < t["full"]
    assert plan.estimate("adapter")["fits_a100_40gb"]


def test_gate_success_and_rollback():
    assert plan.gate(GOOD)["verdict"] == "success"
    bad = json.loads(json.dumps(GOOD))
    bad["candidate"]["heldout_public_si_sdri"] = 4  # forgetting
    assert plan.gate(bad)["verdict"] == "rollback"
    flat = json.loads(json.dumps(GOOD))
    flat["candidate"]["dev_si_sdri"] = 5.1
    assert plan.gate(flat)["verdict"] == "inconclusive"


def test_manifest_hash_stable_and_not_deploy():
    m = plan.manifest(plan.CONFIG)
    assert m["config_hash"] == plan.manifest(dict(plan.CONFIG))["config_hash"]
    assert m["launched"] is False and plan.CONFIG["deploy"] is False
