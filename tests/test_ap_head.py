import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_PATH = Path(__file__).resolve().parents[1] / "eval" / "heads" / "T-307" / "ap_head.py"
_spec = importlib.util.spec_from_file_location("ap_head_t307", _PATH)
ap = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ap
_spec.loader.exec_module(ap)


def test_head_is_192_64_32_and_unit_norm():
    head = ap.MLPHead().eval()
    out = head(torch.randn(5, 192))
    assert out.shape == (5, 32)
    assert torch.allclose(out.norm(dim=1), torch.ones(5), atol=1e-5)
    linears = [m for m in head.net if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(192, 64), (64, 32)]


def test_ap_loss_prefers_query_near_its_own_prototype():
    loss = ap.AngularPrototypicalLoss(init_w=10.0, init_b=-5.0)
    eye = torch.eye(3)
    aligned = eye[:, None, :].repeat(1, 3, 1)  # every crop of speaker i is e_i
    swapped = aligned.clone()
    swapped[:, 0] = eye[[1, 2, 0]]  # queries point at the wrong prototype
    logits = loss.logits(aligned)
    assert logits.shape == (3, 3)
    assert torch.allclose(logits.diag(), torch.full((3,), 5.0))  # w * cos(=1) + b
    assert loss(aligned) < loss(swapped)
    assert loss(aligned).item() == pytest.approx(-np.log(np.exp(5) / (np.exp(5) + 2 * np.exp(-5))), abs=1e-5)


def test_ap_loss_gradients_reach_scale_and_bias():
    loss = ap.AngularPrototypicalLoss()
    episode = torch.randn(4, 3, 8, requires_grad=True)
    loss(episode).backward()
    assert loss.w.grad is not None and loss.b.grad is not None
    assert torch.isfinite(episode.grad).all()


def test_ap_loss_rejects_bad_episode_shapes():
    loss = ap.AngularPrototypicalLoss()
    with pytest.raises(ValueError, match="episode"):
        loss(torch.randn(1, 3, 8))
    with pytest.raises(ValueError, match="episode"):
        loss(torch.randn(4, 1, 8))


def test_evaluate_prototypes_counts_flips_and_hard_partner():
    enroll = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1])
    queries = np.array([[0.9, 0.1], [0.4, 0.6], [0.1, 0.9]])  # the second is closer to speaker 1
    qlabels = np.array([0, 0, 1])
    m = ap.evaluate_prototypes(enroll, labels, queries, qlabels, {0: 1, 1: 0})
    assert m["flip_rate"] == pytest.approx(1 / 3)
    assert m["hard_flip_rate"] == pytest.approx(1 / 3)
    assert m["n_queries"] == 3 and m["hard_n"] == 3


def test_hardest_partner_stays_within_group():
    x = np.array([[1.0, 0.0, 0.0], [0.95, 0.3, 0.0], [0.9, 0.0, 0.4], [0.0, 0.0, 1.0]])
    y = np.array([0, 1, 2, 3])
    assert ap.hardest_partners(x, y)[0] == 1
    groups = {0: "M", 1: "F", 2: "M", 3: "M"}
    assert ap.hardest_partners(x, y, groups)[0] == 2


def test_synthetic_pairs_have_requested_centre_cosine_and_partners():
    x, y, partner = ap.make_synthetic_speakers(3, 50, noise=0.0, nuisance_scale=0.0, pair_cos=0.8, seed=1)
    assert x.shape == (300, 192) and len(np.unique(y)) == 6
    a, b = x[y == 0][0], x[y == 1][0]
    assert float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)) == pytest.approx(0.8, abs=1e-5)
    assert partner[0] == 1 and partner[1] == 0


def test_both_heads_train_and_ap_beats_raw_on_held_out_synthetic():
    cfg = ap.TrainConfig(steps=400, speakers_per_episode=32, seed=0)
    x, y, _ = ap.make_synthetic_speakers(100, 8, seed=0)
    tx, ty, partner = ap.make_synthetic_speakers(10, 15, seed=100, label_offset=1000)
    enroll = np.zeros(len(ty), bool)
    for s in np.unique(ty):
        enroll[np.flatnonzero(ty == s)[:5]] = True
    res = ap.compare(x, y, {"t": (tx[enroll], ty[enroll], tx[~enroll], ty[~enroll], partner)}, cfg)
    assert set(res["t"]) == {"raw", "ap", "ce"}
    assert res["t"]["ap"]["flip_rate"] < res["t"]["raw"]["flip_rate"]
    assert res["t"]["ap"]["margin"] > res["t"]["raw"]["margin"]
    assert 0.0 <= res["t"]["ce"]["flip_rate"] <= 1.0
    s = ap.summarize({"seed0": res})
    assert s["t"]["ap"]["margin"] == pytest.approx(res["t"]["ap"]["margin"], abs=1e-4)
