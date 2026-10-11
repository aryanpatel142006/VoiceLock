import pytest
import torch

from eval.triplet_margin_head import (
    DynamicMarginTripletLoss,
    ProjectionHead,
    evaluate_separation,
    train_cross_entropy_baseline,
    train_projection_head,
)


def test_hard_negative_mining_uses_closest_other_speaker():
    loss = DynamicMarginTripletLoss(base_margin=0.2, margin_scale=0.4)
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.99, 0.1], [0.8, 0.6], [0.0, 1.0]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 1, 1])

    result = loss(embeddings, labels)

    assert result.hard_negative_indices.tolist() == [2, 2, 1, 1]
    assert result.margin[0] > 0.2
    assert result.loss.item() >= 0


def test_projection_head_normalizes_compact_embedding():
    head = ProjectionHead(input_dim=8, projection_dim=3)
    output = head(torch.randn(4, 8))

    assert output.shape == (4, 3)
    assert torch.allclose(output.norm(dim=1), torch.ones(4), atol=1e-5)
    assert sum(p.numel() for p in head.parameters()) == 8 * 3 + 3


def test_loss_requires_two_speakers_and_two_examples_per_speaker():
    loss = DynamicMarginTripletLoss()
    with pytest.raises(ValueError, match="two examples"):
        loss(torch.randn(3, 4), torch.tensor([0, 0, 1]))
    with pytest.raises(ValueError, match="two speakers"):
        loss(torch.randn(4, 4), torch.tensor([0, 0, 0, 0]))


def test_separation_metrics_report_margin_and_false_accept_rate():
    enrolled = torch.tensor([[1.0, 0.0], [0.99, 0.1]])
    positives = torch.tensor([[0.98, 0.2], [0.95, 0.3]])
    negatives = torch.tensor([[0.0, 1.0], [0.7, 0.7]])

    metrics = evaluate_separation(enrolled, positives, negatives, threshold=0.75)

    assert metrics["separation_margin"] > 0
    assert metrics["false_accept_rate"] == 0.0
    assert 0.0 <= metrics["true_accept_rate"] <= 1.0


def test_training_helpers_return_eval_ready_heads():
    torch.manual_seed(4)
    features = torch.randn(6, 5)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    metric_head = train_projection_head(features, labels, projection_dim=3, epochs=2)
    ce_model = train_cross_entropy_baseline(features, labels, projection_dim=3, epochs=2)

    assert not metric_head.training and not ce_model.training
    assert ce_model(features)[0].shape == metric_head(features).shape == (6, 3)
