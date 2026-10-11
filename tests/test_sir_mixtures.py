"""T-413: SIR-binned mixtures from long per-speaker takes (eval.sir_mixtures)."""

import json
from itertools import pairwise

import numpy as np
import pytest

from eval import sir_mixtures as sm
from hearing.audio import io as aio

SR = 16000


def babble(freq, seconds, seed):
    """Syllable-like bursts (0.2 s on, 0.1 s off) of a pitched tone: active-level != RMS."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    env = ((t % 0.3) < 0.2).astype(np.float64)
    x = env * np.sin(2 * np.pi * freq * t) * (0.2 + 0.05 * rng.random())
    return (x + 1e-4 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def root(tmp_path):
    for k, (name, f) in enumerate((("ann", 200.0), ("bob", 330.0), ("cy", 510.0))):
        d = tmp_path / name
        d.mkdir()
        aio.save(d / "profile-01.wav", babble(f, 70.0, k), SR)
        aio.save(d / "profile-02.wav", babble(f, 65.0, k + 10), SR)
        aio.save(d / "short-01.wav", babble(f, 5.0, k), SR)  # not a profile take
    (tmp_path / "_bad").mkdir()
    return tmp_path


def test_active_level_ignores_pauses():
    x = babble(300.0, 3.0, 0)
    padded = np.concatenate([x, np.zeros(3 * SR, np.float32)])
    assert abs(sm.active_level_db(x) - sm.active_level_db(padded)) < 0.5
    assert sm.active_level_db(np.zeros(SR)) == -np.inf


def test_regions_are_disjoint_guarded_and_enrollment_first(root):
    c = sm.load_corpus(root, holdout=("cy",))
    assert set(c.ledger()) == {"ann", "bob", "cy"}
    assert c.speakers("test_unseen") == ["cy"] and "cy" not in c.speakers("train")
    by_take = {}
    for r in c.regions:
        by_take.setdefault(r.take, []).append(r)
    for regions in by_take.values():
        regions.sort(key=lambda r: r.start)
        for a, b in pairwise(regions):
            assert b.start - a.stop >= int(sm.GUARD_S * SR)
    for s in ("ann", "bob"):
        (e,) = c.of(s, "enroll")
        assert e.start == 0 and e.seconds == sm.ENROLL_S and e.take.endswith("profile-01.wav")
        assert c.of(s, "train") and c.of(s, "val") and c.of(s, "test")


@pytest.mark.parametrize("sir", [0.0, -5.0, -15.0])
def test_render_item_hits_exact_sir_and_sums(root, sir):
    c = sm.load_corpus(root)
    it = sm.render_item(c, "test", sir, np.random.default_rng(1))
    assert it["target_id"] != it["interferer_id"]
    assert abs(it["measured_sir_db"] - sir) < 0.05
    np.testing.assert_allclose(it["mix"], it["target"] + it["interferer"] + it["noise"], atol=1e-6)
    assert np.abs(it["mix"]).max() <= sm.PEAK + 1e-6
    assert len(it["enroll"]) == int(sm.ENROLL_CROP_S * SR)
    src = it["source"]
    (e,) = c.of(it["target_id"], "enroll")
    assert src["enroll"]["take"] == e.take and src["enroll"]["start_s"] + sm.ENROLL_CROP_S <= e.seconds


def test_noisy_enrollment_and_train_sir_distribution(root):
    c = sm.load_corpus(root)
    it = sm.render_item(c, "train", 0.0, np.random.default_rng(2), enroll_snr_db=10.0)
    assert it["source"]["enroll"]["noise"].startswith("speaker:")
    draws = np.array([sm.draw_train_sir(np.random.default_rng(k)) for k in range(400)])
    assert np.mean(draws <= 0.0) > 0.7  # asymmetric: the target is usually the quieter one
    assert draws.min() >= -15 - sm.TRAIN_SIR_JITTER_DB and draws.max() <= 5 + sm.TRAIN_SIR_JITTER_DB


def test_render_set_writes_fixture_layout(root, tmp_path):
    c = sm.load_corpus(root)
    out = tmp_path / "out" / "sir_mix_test"
    ms = sm.render_set(c, out, bins=(0.0, -5.0), per_bin=2)
    assert len(ms) == 4
    index = json.loads((out / "index.json").read_text())
    assert index["count"] == 4 and {f["sir_db"] for f in index["fixtures"]} == {0.0, -5.0}
    d = out / ms[0].id
    assert {p.name for p in d.iterdir()} >= {"mix.wav", "target.wav", "interferer.wav", "noise.wav", "enroll.wav"}
    again = sm.render_set(c, tmp_path / "again" / "sir_mix_test", bins=(0.0, -5.0), per_bin=2)
    assert [m.source for m in again] == [m.source for m in ms]  # deterministic from the seed
