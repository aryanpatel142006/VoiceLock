"""Offline job behavior: loss geometry, leakage guards, and held-out calibration."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


def module(name):
    path = Path(__file__).with_name(f"{name}.py")
    assert path.exists(), f"missing deliverable: {path.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_ap_rewards_correct_prototypes_and_has_finite_gradients():
    job = module("train_head_losses")
    good = torch.tensor([[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]]], requires_grad=True)
    bad = good.detach().clone()
    bad[:, 0] = bad[:, 0].flip(0)
    loss = job.AngularPrototypicalLoss()
    assert loss(good) < loss(bad)
    loss(good).backward()
    assert torch.isfinite(good.grad).all()


def test_subcenter_uses_best_center_and_penalizes_target():
    job = module("train_head_losses")
    layer = job.SubCenterArcFace(2, 2, k=2, margin=0.5, scale=1.)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[[1., 0.], [-1., 0.]], [[0., 1.], [0., -1.]]]))
    logits = layer(torch.tensor([[.8, .6]]), torch.tensor([0]))
    assert float(logits[0, 0]) == pytest.approx(.8 * np.cos(0.5) - .6 * np.sin(0.5), abs=1e-4)
    assert float(logits[0, 1]) == pytest.approx(.6, abs=1e-4)


def test_metrics_count_wrong_prototype_choice():
    job = module("train_head_losses")
    enrollment = np.eye(2, dtype=np.float32)
    result = job.metrics(enrollment, np.array([0, 1]), enrollment[::-1], np.array([0, 1]))
    assert result['pair_margin'] == pytest.approx(-1.)
    assert result['flip_rate'] == 1.


def test_recording_cannot_cross_splits(tmp_path):
    job = module("train_head_losses")
    path = tmp_path / 'leaked.npz'
    np.savez(path, x=np.ones((4, 2)), speaker=['a']*4, duration=[.5]*4,
             split=['train', 'enroll', 'calibrate', 'test'], source_id=['same']*4)
    with pytest.raises(ValueError, match='source'):
        job.load_data(path)


def test_calibration_roundtrip_and_quality_features(tmp_path):
    cal = module('calibrate')
    x = np.array([[.8, .5, 1.], [.8, 2., 2.], [.8, .5, 1.], [.8, 2., 2.]])
    model = cal.fit(x, np.array([0, 1, 0, 1]))
    path = tmp_path / 'model.json'
    cal.save(model, path)
    probabilities = cal.predict(path, x)
    assert probabilities[1] > probabilities[0]
    assert np.isfinite(probabilities).all()
