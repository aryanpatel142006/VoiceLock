"""Offline signal and learning checks; no weights, network, or audio devices."""
import numpy as np
import pytest
import torch

from eval.dual_band_projection import (
    DualBandProjection,
    DualBandSplitter,
    extract_band_embeddings,
    train_head,
)


def test_split_reconstructs_delayed_audio_and_is_chunk_invariant():
    x = np.random.default_rng(42).uniform(-0.5, 0.5, 3200).astype(np.float32)
    splitter = DualBandSplitter()
    low, high = splitter.process(x)
    delay = splitter.delay_samples
    np.testing.assert_allclose(low + high, np.r_[np.zeros(delay), x[:-delay]], atol=1e-7)
    splitter.reset()
    pairs = [splitter.process(c) for c in np.array_split(x, 20)]
    np.testing.assert_allclose(np.concatenate([p[0] for p in pairs]), low, atol=1e-7)
    np.testing.assert_allclose(np.concatenate([p[1] for p in pairs]), high, atol=1e-7)


def test_frequency_selectivity():
    t = np.arange(16000) / 16000
    for freq, band in [(500, 0), (4000, 1)]:
        x = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        signals = DualBandSplitter().process(x)
        wanted = np.mean(signals[band][256:] ** 2)
        leakage = np.mean(signals[1 - band][256:] ** 2)
        assert wanted > 1000 * leakage


@pytest.mark.parametrize('x', [np.zeros((2, 160)), np.array([np.nan]), np.array([1.1])])
def test_split_rejects_invalid_audio(x):
    with pytest.raises(ValueError):
        DualBandSplitter().process(x)


def test_extraction_uses_two_frozen_encoder_calls():
    calls = []
    def encoder(x):
        calls.append(x.copy())
        return np.array([1, np.mean(x), np.std(x)], dtype=np.float32)
    pair = extract_band_embeddings(np.zeros(1600, dtype=np.float32), encoder)
    assert pair.shape == (2, 3)
    assert len(calls) == 2
    np.testing.assert_allclose(np.linalg.norm(pair, axis=1), 1)


def test_head_gate_and_embedding_are_normalized_and_trainable():
    torch.manual_seed(42)
    head = DualBandProjection(in_dim=8, out_dim=4)
    low, high = torch.randn(6, 8), torch.randn(6, 8)
    z, gate = head(low, high)
    assert z.shape == (6, 4)
    torch.testing.assert_close(z.norm(dim=1), torch.ones(6))
    torch.testing.assert_close(gate.sum(dim=1), torch.ones(6))
    assert torch.all((gate > 0) & (gate < 1))
    z[:, 0].sum().backward()
    for branch in [head.low, head.high, head.gate]:
        assert branch.weight.grad.abs().sum() > 0


@pytest.mark.slow
def test_training_reduces_loss_and_updates_gate():
    torch.manual_seed(42)
    labels = torch.arange(3).repeat_interleave(8)
    centers = torch.eye(3, 8)
    low = centers[labels] + 0.05 * torch.randn(24, 8)
    high = torch.randn(24, 8)
    head = DualBandProjection(8, 4)
    before = head.gate.weight.detach().clone()
    history = train_head(head, low, high, labels, epochs=80, lr=0.02)
    assert history[-1] < history[0] * 0.5
    assert not torch.equal(before, head.gate.weight)
    clone = DualBandProjection(8, 4)
    clone.load_state_dict(head.state_dict())
    torch.testing.assert_close(head(low, high)[0], clone(low, high)[0])


def test_training_rejects_single_speaker():
    with pytest.raises(ValueError, match='two speakers'):
        train_head(DualBandProjection(8, 4), torch.ones(4, 8), torch.ones(4, 8),
                   torch.zeros(4, dtype=torch.long), epochs=1)


def test_empty_split_and_reset():
    splitter = DualBandSplitter()
    assert splitter.process(np.zeros(0))[0].size == 0
    splitter.process(np.ones(160, dtype=np.float32))
    splitter.reset()
    low, high = splitter.process(np.zeros(160, dtype=np.float32))
    assert not low.any() and not high.any()


@pytest.mark.parametrize('embedding', [np.zeros(3), np.array([np.nan, 1])])
def test_rejects_invalid_encoder_output(embedding):
    with pytest.raises(ValueError, match='embedding'):
        extract_band_embeddings(np.zeros(160), lambda _: embedding)


def test_encoder_crop_is_aligned_and_same_length():
    x = np.zeros(1600, dtype=np.float32)
    x[800] = 1
    calls = []
    def encoder(band):
        calls.append(band)
        return np.ones(3)
    extract_band_embeddings(x, encoder)
    assert all(len(band) == len(x) for band in calls)
    np.testing.assert_allclose(calls[0] + calls[1], x, atol=1e-7)


def test_head_rejects_nonfinite_or_mismatched_bands():
    head = DualBandProjection(8, 4)
    with pytest.raises(ValueError):
        head(torch.zeros(1, 8), torch.zeros(2, 8))
    with pytest.raises(ValueError):
        head(torch.full((1, 8), float('nan')), torch.zeros(1, 8))
    z, _ = head(torch.zeros(1, 8), torch.zeros(1, 8))
    assert torch.isfinite(z).all() and not z.any()
