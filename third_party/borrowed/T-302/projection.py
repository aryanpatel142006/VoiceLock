# Copyright (c) 2023, Zhengyang Chen (chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapted WeSpeaker SimCLR MLP and InfoNCE assembly; see ATTRIBUTION.md.

T-302 modifications: detached from encoder/audio, rectangular 192->64->32 MLP,
unit output, exactly two views, input validation and device-local labels.
Random initialization is an offline candidate, not a trained speaker verifier.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    """Project cached ECAPA vectors; caller moves module/tensors to its device."""

    def __init__(self, in_dim=192, hidden_dim=64, out_dim=32):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, embeddings):
        if embeddings.ndim != 2 or embeddings.shape[1] != self.in_dim:
            raise ValueError(f"expected (batch, {self.in_dim}) embeddings")
        # A zero projected vector stays zero (PyTorch epsilon guard).
        return F.normalize(self.net(embeddings), dim=-1)


def info_nce_logits(view_a, view_b, temperature=0.07):
    """Return positive-first logits and CE targets for two corresponding views.

    Each row i in view_a must match row i in view_b. Other instances are
    negatives: do not use this objective with repeated same-speaker instances
    without a speaker-aware sampler/loss. At least two instances are required.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if view_a.ndim != 2 or view_a.shape != view_b.shape or view_a.shape[0] < 2:
        raise ValueError("expected matching (batch >= 2, dimension) views")
    features = torch.cat((view_a, view_b), dim=0)
    bs = view_a.shape[0]
    labels = torch.cat([torch.arange(bs, device=features.device) for _ in range(2)])
    labels = labels.unsqueeze(0) == labels.unsqueeze(1)
    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # Borrowed positive/negative assembly: discard self comparisons first.
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=features.device)
    return logits, targets
