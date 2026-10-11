"""Offline behavior checks for the T-302 borrowed projection candidate."""
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def candidate():
    path = Path(__file__).resolve().parents[1] / 'third_party/borrowed/T-302/projection.py'
    assert path.is_file(), 'T-302 borrowed module is missing'
    spec = importlib.util.spec_from_file_location('borrowed_projection', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_ecapa_projection_is_unit_length_and_trainable():
    module = candidate()
    with torch.random.fork_rng():
        torch.manual_seed(302)
        head = module.ProjectionHead()
        x = torch.randn(8, 192)
        z = head(x)
        assert z.shape == (8, 32)
        torch.testing.assert_close(z.norm(dim=-1), torch.ones(8))
        loss = F.cross_entropy(*module.info_nce_logits(z, head(x + 0.1 * torch.randn_like(x))))
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
        assert sum(p.grad.abs().sum() for p in head.parameters()) > 0
        torch.testing.assert_close(head(x[:1]), z[:1])


def test_info_nce_pairs_corresponding_views_and_removes_self_similarity():
    module = candidate()
    # Orthogonal instances: the positive cosine is 1, every negative is 0.
    view = torch.eye(3)
    logits, labels = module.info_nce_logits(view, view, temperature=0.5)
    assert logits.shape == (6, 5)
    torch.testing.assert_close(logits[:, 0], torch.full((6,), 2.0))
    torch.testing.assert_close(logits[:, 1:], torch.zeros(6, 4))
    assert labels.tolist() == [0] * 6
    good = F.cross_entropy(logits, labels)
    bad = F.cross_entropy(*module.info_nce_logits(view, view.roll(1, 0), temperature=0.5))
    assert good < bad


@pytest.mark.parametrize('temperature', [0.0, -1.0, float('nan'), float('inf')])
def test_info_nce_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        candidate().info_nce_logits(torch.eye(3), torch.eye(3), temperature=temperature)


def test_invalid_view_pair_and_embedding_shape_are_rejected():
    module = candidate()
    with pytest.raises(ValueError):
        module.info_nce_logits(torch.ones(1, 32), torch.ones(1, 32))
    with pytest.raises(ValueError):
        module.info_nce_logits(torch.ones(2, 32), torch.ones(3, 32))
    with pytest.raises(ValueError):
        module.ProjectionHead()(torch.ones(2, 191))
