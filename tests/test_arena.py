"""Arena pipeline: fixed-mixture runner, ASHA scheduler decisions, canonical records, A/B batch + server."""
from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from experiments.arena import arena, result, stats
from experiments.arena.runner import Candidate, Scorers, load_mixtures
from experiments.arena.scheduler import Config, Study, check_reference, decide
from experiments.arena.serve import make_server

SR = 16000


def _write_set(root, n_clusters=2, per_cluster=4, seconds=1.0):
    """Synthetic fixtures: target tone + interferer tone + noise, two source clusters."""
    rng = np.random.default_rng(0)
    t = np.arange(int(SR * seconds)) / SR
    fixtures = []
    for c in range(n_clusters):
        for k in range(per_cluster):
            fid = f"{c}{k:02d}_fx"
            d = root / fid
            d.mkdir(parents=True)
            target = 0.3 * np.sin(2 * np.pi * (220 + 40 * c) * t) * (t < 0.7)
            interferer = 0.2 * np.sin(2 * np.pi * (330 + 25 * k) * t)
            noise = 0.02 * rng.standard_normal(len(t))
            stems = {"target": target, "interferer": interferer, "noise": noise,
                     "mix": target + interferer + noise}
            for name, x in stems.items():
                sf.write(d / f"{name}.wav", x.astype(np.float32), SR, subtype="FLOAT")
            man = {"id": fid, "set": "syn", "sample_rate": SR, "duration_s": seconds, "snr_db": 0.0,
                   "sir_db": 0.0, "target_id": f"T{c}", "interferer_id": "I", "noise_type": "white",
                   "transcript": "", "paths": {n: f"{n}.wav" for n in stems}, "seed": 0}
            (d / "manifest.json").write_text(json.dumps(man))
            fixtures.append(man)
    (root / "index.json").write_text(json.dumps({"fixtures": fixtures}))
    return root


def test_bootstrap_resamples_clusters_not_rows():
    one = stats.paired_cluster_bootstrap([1.0, 1.2, 0.8], ["a", "a", "a"])
    assert one["lo"] is None and stats.verdict(one, +1) == "uncertain"  # one cluster: no CI, more eval
    two = stats.paired_cluster_bootstrap([1.0, 1.2, 0.9, 1.1], ["a", "a", "b", "b"])
    assert two["clusters"] == 2 and stats.verdict(two, +1) == "better" and stats.verdict(two, -1) == "worse"
    mixed = stats.paired_cluster_bootstrap([1.0, -1.0], ["a", "b"])
    assert stats.verdict(mixed, +1) == "uncertain"


def test_stop_never_rests_on_si_sdr_alone():
    cfg = Config()
    ci = lambda v: {"verdict": v}
    base = {m: ci("tied") for m in (*result.QUALITY, *result.PERCEPTUAL)}
    rows = [{"algo_latency_ms": 10.0}]
    only_sdr = {**base, "si_sdri": ci("worse"), "dnsmos_ovrl": ci("better")}
    assert decide(only_sdr, False, rows, cfg)[0] == "CONTINUE"
    assert decide(only_sdr, True, rows, cfg)[0] == "HUMAN_REVIEW"
    uncertain = {**base, "si_sdri": ci("worse"), "estoi": ci("uncertain")}
    assert decide(uncertain, False, rows, cfg)[0] == "CONTINUE"
    silent = {**base, "si_sdri": ci("worse"), "estoi": ci("worse"), "dnsmos_bak": ci("better"),
              "present_miss_frac": {"delta": 0.9, "verdict": "worse"}}
    assert decide(silent, False, rows, cfg)[0] == "STOP"  # DNSMOS likes silence; it cannot veto
    dominated = {**base, "si_sdri": ci("worse"), "estoi": ci("worse")}
    assert decide(dominated, False, rows, cfg)[0] == "STOP"


def test_reference_pins_then_detects_drift(tmp_path):
    p = tmp_path / "ref.json"
    assert check_reference({"m": {"si_sdri": 1.0}}, p) == (True, 0.0)
    assert check_reference({"m": {"si_sdri": 1.0}}, p)[0]
    ok, drift = check_reference({"m": {"si_sdri": 1.5}}, p)
    assert not ok and drift == pytest.approx(0.5)


def test_study_two_rungs_records_batch_and_server(tmp_path):
    mixtures = load_mixtures(_write_set(tmp_path / "syn"))
    run_dir = tmp_path / "run"
    baseline = Candidate("base", "gain(db=0)")
    # 'muted' renders silence: SI-SDRi is confidently worse at rung 0 but (E)STOI on tones is still
    # uncertain, so it earns more evaluation; at rung 1 every quality metric is worse -> STOP.
    # 'same' equals the baseline -> tied -> CONTINUE, then HUMAN_REVIEW at the last rung.
    cands = [Candidate("muted", "silence"), Candidate("same", "passthrough")]
    cfg = Config(eta=2, n0=4, rungs=2, budget_s=100.0, n_boot=200)
    state = Study("syn", baseline, cands, mixtures, run_dir, cfg, Scorers()).run()

    assert state["plan"]["rung_sizes"] == [4, 8]
    assert [d["action"] for d in state["decisions"]["muted"]] == ["CONTINUE", "STOP"]
    assert [d["action"] for d in state["decisions"]["same"]] == ["CONTINUE", "HUMAN_REVIEW"]
    assert all(r["ok"] for r in state["baseline_reference"])
    led = state["ledger"]
    assert led["baseline_reserved_s"] == pytest.approx(25.0) and led["candidates"]["same"] > 0

    records = result.load(run_dir / "results.jsonl")
    assert {(r["candidate"]["id"], r["rung"]) for r in records} == {
        ("muted", 0), ("same", 0), ("base", 0), ("muted", 1), ("same", 1), ("base", 1)}
    same0 = next(r for r in records if r["candidate"]["id"] == "same" and r["rung"] == 0)
    assert same0["decision"]["next_rung_estimate"]["mixtures"] == 8
    assert set(same0["dimensions"]) == set(result.DIMENSIONS)
    assert same0["inputs"]["mixture_ids"] == [r["mixture_id"] for r in same0["per_mixture"]]
    # same mixtures for every candidate, interleaved across clusters
    assert {m[0] for m in same0["inputs"]["mixture_ids"][:2]} == {"0", "1"}
    assert (run_dir / same0["per_mixture"][0]["output"]).exists()

    # resume: a second Study on the same dir re-renders nothing and keeps the decisions
    again = Study("syn", baseline, cands, mixtures, run_dir, cfg, Scorers()).run()
    assert again["decisions"] == state["decisions"]

    batch_dir = arena.export_batch(run_dir, 1, trials_per_candidate=3)
    batch = json.loads((batch_dir / "batch.json").read_text())
    assert batch["schema"] == arena.BATCH_SCHEMA and len(batch["trials"]) == 6  # 2 candidates x 3
    assert "key" not in json.dumps(batch)
    srv = make_server(run_dir, 1, port=0)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        assert b"Arena" in urllib.request.urlopen(base_url + "/").read()
        got = json.loads(urllib.request.urlopen(base_url + "/batch.json").read())
        trial = got["trials"][0]
        assert urllib.request.urlopen(f"{base_url}/{trial['a']}").read()[:4] == b"RIFF"
        assert urllib.request.urlopen(f"{base_url}/{trial['reference']}").read()[:4] == b"RIFF"
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base_url + "/arena/rung1/key.json")
        req = urllib.request.Request(base_url + "/verdict", method="POST",
                                     data=json.dumps({"trial_id": trial["trial_id"], "choice": "a",
                                                      "listener": "t"}).encode())
        assert json.loads(urllib.request.urlopen(req).read())["choice"] == "a"
    finally:
        srv.shutdown()
        srv.server_close()
    tally = arena.tally(batch_dir)
    assert sum(t["n"] for t in tally.values()) == 1
    assert sum(t["wins"] + t["losses"] for t in tally.values()) == 1


def test_validate_rejects_aggregate_and_winner():
    rec = {k: None for k in result.REQUIRED}
    rec.update(schema=result.SCHEMA, candidate={k: "" for k in result.CANDIDATE_REQUIRED},
               decision={"action": "CONTINUE"}, dimensions={d: {} for d in result.DIMENSIONS},
               inputs={"mixture_ids": []}, per_mixture=[], compute={"consumed_s": 0.0, "audio_s": 0.0})
    result.validate(rec)
    with pytest.raises(ValueError, match="winner"):
        result.validate({**rec, "winner": "x"})
    with pytest.raises(ValueError, match="action"):
        result.validate({**rec, "decision": {"action": "WIN"}})
