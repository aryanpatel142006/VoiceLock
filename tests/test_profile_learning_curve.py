import numpy as np

from eval import profile_learning_curve as p


def test_spread_order_prefixes_are_spread_and_complete():
    o = p.spread_order(10)
    assert sorted(o) == list(range(10))
    assert o[:2] == [0, 9]
    assert 4 in o[:3]
    assert p.spread_order(1) == [0] and p.spread_order(0) == []


def test_check_disjoint_detects_leak_and_accepts_separate_audio():
    a, b = p.fake_voice(110, 5, 1), p.fake_voice(110, 5, 2)
    assert p.check_disjoint(a, b)
    assert not p.check_disjoint(a, np.concatenate([b[:8000], a[16000:40000]]))


def test_dev_test_split_is_time_ordered_and_disjoint():
    dev, test = p.split_dev_test(list(range(10)))
    assert dev == [0, 1, 2, 3, 4] and test == [6, 7, 8, 9]
    assert not set(dev) & set(test)


def test_quality_gate_rejects_silence_clipping_and_nan():
    v = p.fake_voice(110, 1.5, 1)
    assert p.is_clean(v)
    assert not p.is_clean(np.zeros_like(v))
    assert not p.is_clean(np.clip(v * 40, -1, 1))
    bad = v.copy()
    bad[5] = np.nan
    assert not p.is_clean(bad)
    assert len(p.clean_examples(np.concatenate([v, np.zeros(24000, np.float32), v]))) == 2


def test_threshold_rejects_wrong_speaker_and_noise_at_target_far():
    rng = np.random.default_rng(0)
    neg = list(rng.uniform(0.0, 0.5, 200))
    thr = p.calibrate_threshold(neg)
    assert np.mean(np.asarray(neg) >= thr) <= p.TARGET_FAR + 1e-9
    assert p.calibrate_threshold([]) == float("inf")


def test_verdict_rules():
    base = {"tpr_clean": .5, "tpr_quiet": .5, "far_wrong": .05, "far_absent": 0.0, "wrong_speaker_s": 1.0, "closed_set_acc": .9}
    better = {**base, "tpr_clean": .8, "tpr_quiet": .7}
    assert p.promotion_verdict(base, better)["verdict"] == "embedding_gain_only"
    assert p.promotion_verdict(base, {**better, "far_wrong": .10})["verdict"] == "no_win"
    assert p.promotion_verdict(base, {**better, "wrong_speaker_s": 2.0})["verdict"] == "no_win"
    assert p.promotion_verdict(base, {**base, "tpr_clean": .52})["verdict"] == "no_win"
    assert p.promotion_verdict(base, {**better, "closed_set_acc": .8})["verdict"] == "no_win"


def test_run_target_on_synthetic_voices_blocks_and_keeps_split_disjoint():
    embed = p.Embedder(p.FakeBackend())
    a, b = p.fake_voice(110, 30, 1), p.fake_voice(160, 30, 3)
    short = p.run_target("a", a[: 5 * p.SR], a, b, b, embed, crops=(2.0,))
    assert "blocked" in short  # fewer than 8 clean examples must block, not invent
    r = p.run_target("a", a[: 20 * p.SR], a[20 * p.SR :], b[: 20 * p.SR], b[20 * p.SR :], embed, crops=(2.0,))
    rows = r["crops"]["2"]["rows"]
    assert {"reference_full", "first@1", "mean@8", "multi@8", "mean@2"} <= set(rows)
    assert rows["first@1"]["n_test"]["present_clean"] > 0
    assert rows["mean@8"]["far_absent"] is not None
    assert r["negative_rejected_at_same_voice"] in (True, False)
    assert all(v["vs_first_1"]["verdict"] in ("no_win", "embedding_gain_only") for k, v in rows.items() if k != "first@1")
