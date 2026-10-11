import numpy as np

from eval.selector_gate import gate, run


def _case(cid, target, split="dev", seed=0):
    rng = np.random.default_rng(seed)
    c = gate.unit(rng.normal(size=(1, 16)) + 0)[0]
    other = gate.unit(rng.normal(size=(1, 16)))[0]
    mk = lambda v, n: gate.unit(v + 0.2 * rng.normal(size=(n, 16)))
    emb = np.stack([mk(c, 10), mk(other, 10)])
    return gate.Case(cid, split, cid, emb, emb.copy(), target, mk(c, 6), 1.0, gallery_head=mk(c, 6))


def test_identity_matches_raw_exactly():
    cases = [_case("a", 0), _case("abs", None, seed=3)]
    raw, ident = gate.RawArm(), gate.IdentityArm()
    thr = gate.calibrate_threshold(raw, cases)
    p = gate.parity([gate.eval_case(raw, c, thr) for c in cases], [gate.eval_case(ident, c, thr) for c in cases])
    assert p["ok"]


def test_metrics_null_when_undefined_and_summary():
    cases = [_case("a", 0), _case("abs", None, seed=3)]
    arm = gate.RawArm()
    thr = gate.calibrate_threshold(arm, cases)
    rows = [gate.eval_case(arm, c, thr) for c in cases]
    assert rows[1]["top1"] is None and rows[1]["retention"] is None
    s = gate.summarize(rows)
    assert s["recovery_ratio"] is None and s["unknown_false_accept"] is None
    assert s["top1"] == 1.0


def test_one_group_gives_null_ci():
    arm = gate.RawArm()
    c = _case("a", 0)
    s = gate.summarize([gate.eval_case(arm, c, 0.0)])
    assert s["top1_ci95_by_group"] is None


def test_delivered_audio_follows_selection():
    c = _case("a", 0)
    c.audio, c.start_s = ["x", "y"], [np.zeros(3), np.zeros(3)]
    out = gate.delivered_audio(c, np.array([0, 1, -1]), lambda p, s, n: np.full(n, 1.0 if p == "x" else 2.0, np.float32))
    n = gate.SR
    assert out[0] == 1.0 and out[n] == 2.0 and out[2 * n] == 0.0


def test_blocked_arms_are_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv("HEARING_DISTILL_CKPT", raising=False)
    arms = run.make_arms([_case("a", 0)])
    blocked = {a.name: a.reason for a in arms if isinstance(a, gate.BlockedArm)}
    assert "distillation" in blocked and "lda" in blocked
