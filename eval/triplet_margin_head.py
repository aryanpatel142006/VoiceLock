"""Research implementation of a compact speaker projection head (T-281).

The ECAPA encoder is deliberately outside this module: callers provide frozen
embeddings, normally one embedding per 0.5 s crop.  The head mines the closest
example from another speaker in the current batch and increases the margin for
especially confusing negatives.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """A small linear metric head with unit-length output embeddings."""

    def __init__(self, input_dim: int, projection_dim: int = 64) -> None:
        super().__init__()
        if input_dim < 1 or projection_dim < 1:
            raise ValueError("input_dim and projection_dim must be positive")
        self.projection = nn.Linear(input_dim, projection_dim)

    def forward(self, embeddings: Tensor) -> Tensor:
        return F.normalize(self.projection(embeddings), dim=-1)


class CrossEntropyBaseline(nn.Module):
    """The matched static cross-entropy baseline for the research comparison."""

    def __init__(self, input_dim: int, projection_dim: int, classes: int) -> None:
        super().__init__()
        self.head = ProjectionHead(input_dim, projection_dim)
        self.classifier = nn.Linear(projection_dim, classes)

    def forward(self, embeddings: Tensor) -> tuple[Tensor, Tensor]:
        projected = self.head(embeddings)
        return projected, self.classifier(projected)


@dataclass(frozen=True)
class TripletLossResult:
    loss: Tensor
    margin: Tensor
    hard_negative_indices: Tensor
    positive_distances: Tensor
    negative_distances: Tensor


class DynamicMarginTripletLoss(nn.Module):
    """Batch-hard triplet loss with a similarity-dependent negative margin.

    ``margin = base_margin + margin_scale * max(cos(anchor, hard_negative), 0)``.
    Thus a negative close to the anchor receives a stricter separation target,
    while unrelated negatives do not inflate the loss.
    """

    def __init__(
        self,
        base_margin: float = 0.2,
        margin_scale: float = 0.4,
        max_margin: float = 0.8,
    ) -> None:
        super().__init__()
        if not 0 <= base_margin <= max_margin:
            raise ValueError("base_margin must be between zero and max_margin")
        if margin_scale < 0 or max_margin <= 0:
            raise ValueError("margin_scale must be non-negative and max_margin positive")
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.max_margin = max_margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> TripletLossResult:
        if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
            raise ValueError("embeddings must be 2-D and labels must match its batch length")
        if len(embeddings) < 4 or labels.unique().numel() < 2:
            raise ValueError("batch needs two speakers and two examples per speaker")
        same = labels[:, None].eq(labels[None, :])
        if bool((same.sum(dim=1) < 2).any()):
            raise ValueError("batch needs two examples for every speaker")

        cosine = F.normalize(embeddings, dim=-1) @ F.normalize(embeddings, dim=-1).T
        positive_distances = (1 - cosine.masked_fill(~same | torch.eye(len(labels), dtype=torch.bool, device=labels.device), -1)).min(dim=1).values
        negative_similarity = cosine.masked_fill(same, -torch.inf)
        hard_similarity, hard_indices = negative_similarity.max(dim=1)
        negative_distances = 1 - hard_similarity
        margin = (self.base_margin + self.margin_scale * hard_similarity.clamp_min(0)).clamp_max(
            self.max_margin
        )
        losses = F.relu(positive_distances - negative_distances + margin)
        return TripletLossResult(
            loss=losses.mean(),
            margin=margin,
            hard_negative_indices=hard_indices,
            positive_distances=positive_distances,
            negative_distances=negative_distances,
        )


def evaluate_separation(
    enrolled: Tensor,
    positives: Tensor,
    negatives: Tensor,
    *,
    threshold: float = 0.7,
) -> dict[str, float]:
    """Return cosine acceptance metrics and the positive/negative score margin."""
    if enrolled.ndim != 2 or positives.ndim != 2 or negatives.ndim != 2:
        raise ValueError("all embeddings must be 2-D")
    if enrolled.shape[1] != positives.shape[1] or enrolled.shape[1] != negatives.shape[1]:
        raise ValueError("embedding dimensions must match")
    prototype = F.normalize(enrolled.mean(dim=0, keepdim=True), dim=-1)
    positive_scores = (F.normalize(positives, dim=-1) @ prototype.T).squeeze(1)
    negative_scores = (F.normalize(negatives, dim=-1) @ prototype.T).squeeze(1)
    return {
        "separation_margin": float(positive_scores.mean() - negative_scores.mean()),
        "false_accept_rate": float((negative_scores >= threshold).float().mean()),
        "true_accept_rate": float((positive_scores >= threshold).float().mean()),
        "positive_score_mean": float(positive_scores.mean()),
        "negative_score_mean": float(negative_scores.mean()),
    }


def train_projection_head(
    features: Tensor,
    labels: Tensor,
    *,
    projection_dim: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-2,
    loss_fn: DynamicMarginTripletLoss | None = None,
) -> ProjectionHead:
    """Fit the head on frozen ECAPA features and return it in evaluation mode."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    head = ProjectionHead(features.shape[1], projection_dim)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    criterion = loss_fn or DynamicMarginTripletLoss()
    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        result = criterion(head(features), labels)
        result.loss.backward()
        optimizer.step()
    return head.eval()


def train_cross_entropy_baseline(
    features: Tensor,
    labels: Tensor,
    *,
    projection_dim: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-2,
) -> CrossEntropyBaseline:
    """Fit the matched static-CE baseline; its ``.head`` can use the same metrics."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    classes = int(labels.unique().numel())
    model = CrossEntropyBaseline(features.shape[1], projection_dim, classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        _, logits = model(features)
        F.cross_entropy(logits, labels).backward()
        optimizer.step()
    return model.eval()
