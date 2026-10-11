import numpy as np

from eval import pair_compare as pc

SR = pc.SR


def _tone(f, s, amp=0.3):
    t = np.arange(round(s * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_compose_layout_and_sir():
    tp = [_tone(200, 1.5) for _ in range(4)]
    op = [_tone(330, 1.5) for _ in range(4)]
    st = pc.compose(tp, op, -6.0, None)
    n = round(pc.DUR_S * SR)
    assert all(len(st[k]) == n for k in ("mix", "target", "other", "noise"))
    s, e = (round(x * SR) for x in pc.OVERLAP)
    # the other talks only inside the overlap section, the target in all three
    assert np.abs(st["other"][:s]).max() == 0 and np.abs(st["other"][e:]).max() == 0
    assert np.abs(st["target"][: SR]).max() > 0 and np.abs(st["target"][-2 * SR:]).max() > 0
    # SIR -6: the other's active level 6 dB above the target's
    lvl = lambda x: 20 * np.log10(pc.ks._active_rms(x))
    assert abs((lvl(st["other"]) - lvl(st["target"])) - 6.0) < 0.1
    np.testing.assert_allclose(st["mix"], st["target"] + st["other"] + st["noise"], atol=1e-6)
    assert np.abs(st["mix"]).max() <= pc.ks.PEAK + 1e-6


def test_compose_noise_snr():
    tp = [_tone(200, 1.5) for _ in range(4)]
    rng = np.random.default_rng(0)
    st = pc.compose(tp, tp, 0.0, rng.standard_normal(SR * 3).astype(np.float32))
    snr = 20 * np.log10(pc.ks._active_rms(st["target"]) / np.sqrt(np.mean(st["noise"].astype(np.float64) ** 2)))
    assert abs(snr - pc.ks.CAFE_SNR_DB) < 0.2


def test_half_shift():
    assert pc.half_shift([1, 2, 3, 4, 5]) == [3, 4, 5, 1, 2]
    assert pc.half_shift([]) == []


def test_crops_drop_silence():
    x = np.concatenate([_tone(200, 3), np.zeros(3 * SR, np.float32)])
    cs = pc.crops(x)
    assert cs and all(len(c) == SR for c in cs)
    assert len(cs) <= 6  # only crops touching the 3 s of tone (starts 0..2.5 s)


def test_confusion_and_cosines():
    e = {"a": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0]), "c": np.array([0, 0, 1.0])}
    crop = {"a": np.array([[0.9, 0.1, 0], [0.2, 0.8, 0]]), "b": np.array([[0, 1.0, 0]]), "c": np.array([[0, 0.1, 0.9]])}
    res = pc.confusion(crop, e)
    assert res["matrix"]["a"] == {"a": 1, "b": 1, "c": 0}
    assert res["pairs"]["a-b"]["errors"] == 1 and res["pairs"]["a-b"]["crops"] == 3
    assert res["pairs"]["a-c"]["errors"] == 0
    assert pc.cosine_table(e) == {"a-b": 0.0, "a-c": 0.0, "b-c": 0.0}


def test_rank_orders_by_wrong_then_leak():
    def row(pair, target, wrong, leak):
        return {"pair": pair, "target": target, "wrong_pct": wrong, "leak_db": leak, "si_sdri": 5.0,
                "si_sdri_overlap": 6.0, "atten_s": 1.0, "gated_pct": 2.0}

    rows = [row("x-y", "x", 10, -10), row("x-y", "y", 2, -10), row("x-z", "x", 4, -12), row("x-z", "z", 4, -12),
            row("y-z", "y", 4, -9), row("y-z", "z", 4, -9)]
    r = pc.rank(rows)
    assert [p["pair"] for p in r] == ["x-z", "y-z", "x-y"]
    assert r[2]["worst_role"] == "x" and r[2]["worst_role_wrong_pct"] == 10


def test_pair_key_is_order_free():
    assert pc.pair_key("ahmad", "aryan") == pc.pair_key("aryan", "ahmad") == "aryan-ahmad"
