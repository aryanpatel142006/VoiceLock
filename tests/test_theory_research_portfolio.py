import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("t253", ROOT / "experiments/theory/T-253/portfolio.py")
t253 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t253)
CAND = ROOT / "experiments/theory/T-253/candidates.json"


def test_ranking_deterministic_and_ordered(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    assert t253.main(["--out-dir", str(a)]) == 0
    assert t253.main(["--out-dir", str(b)]) == 0
    assert (a / "ranked.json").read_text() == (b / "ranked.json").read_text()
    ranked = [r for r in json.loads((a / "ranked.json").read_text())["ranked"] if r["rank"]]
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in ranked] == list(range(1, len(ranked) + 1))


def test_at_least_three_independent_falsifiable_paths():
    rows = t253.rank(t253.load(CAND))
    ok = [r for r in rows if r["independent_path"] and r["rank"] and r["falsifier"]]
    assert len(ok) >= 3


def test_proposals_cpu_only_no_production(tmp_path):
    t253.main(["--out-dir", str(tmp_path)])
    props = json.loads((tmp_path / "proposals.json").read_text())["proposals"]
    assert props
    for p in props:
        assert p["caps"]["gpu_cases"] == 0 and p["production_adoption"] is False


def test_doc_mentions_every_ranked_id():
    doc = (ROOT / "docs/research/theory/research_portfolio.md").read_text()
    for r in t253.rank(t253.load(CAND)):
        assert r["id"] in doc


def test_validation_rejects_bad_input(tmp_path):
    d = json.loads(CAND.read_text())
    d["candidates"][0]["impact"] = 9
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        t253.load(bad)
