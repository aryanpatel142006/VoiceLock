"""Projection heads, metric transforms, and contrastive loss objectives (T-246).

Implements:
1. LinearProjectionHead: Linear mapping D_in -> D_proj with unit L2 normalization.
2. MLPProjectionHead: 2-layer MLP with LayerNorm, GELU, optional residual, and unit L2 normalization.
3. WhiteningTransform: Empirical/Ledoit-Wolf shrinkage whitening (Mahalanobis metric).
4. fit_closed_set_lda: Reference 2-class LDA matching hearing.tse.enrolled.fit_lda.
5. Contrastive/triplet objectives with anti-collapse covariance regularization.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class LinearProjectionHead(nn.Module):
    """Linear metric projection head with unit L2 hypersphere normalization."""

    def __init__(self, in_dim: int = 192, proj_dim: int = 128, bias: bool = False) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        self.proj = nn.Linear(in_dim, proj_dim, bias=bias)
        nn.init.orthogonal_(self.proj.weight)
        if bias and self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project input vectors and normalize to unit Euclidean length."""
        z = self.proj(x)
        return F.normalize(z, p=2, dim=-1, eps=1e-12)

    def project_numpy(self, x: np.ndarray) -> np.ndarray:
        """Convenience projection for NumPy evaluation arrays."""
        self.eval()
        with torch.no_grad():
            t_x = torch.from_numpy(np.asarray(x, dtype=np.float32))
            out = self.forward(t_x)
            return out.cpu().numpy()


class MLPProjectionHead(nn.Module):
    """Non-linear 2-layer MLP projection head with LayerNorm, GELU, and unit normalization."""

    def __init__(
        self,
        in_dim: int = 192,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.residual = residual and (in_dim == proj_dim)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, proj_dim)

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual shortcut and unit normalization."""
        h = self.fc1(x)
        h = self.ln(h)
        h = self.act(h)
        z = self.fc2(h)
        if self.residual:
            z = z + x
        return F.normalize(z, p=2, dim=-1, eps=1e-12)

    def project_numpy(self, x: np.ndarray) -> np.ndarray:
        """Convenience projection for NumPy evaluation arrays."""
        self.eval()
        with torch.no_grad():
            t_x = torch.from_numpy(np.asarray(x, dtype=np.float32))
            out = self.forward(t_x)
            return out.cpu().numpy()


class WhiteningTransform:
    """Shrinkage whitening transform on background speaker/channel covariance."""

    def __init__(self, shrinkage: float = 0.5) -> None:
        self.shrinkage = float(shrinkage)
        self.mean: np.ndarray | None = None
        self.transform_mat: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> WhiteningTransform:
        """Estimate centroid and regularized inverse square root covariance."""
        arr = np.asarray(x, dtype=np.float64)
        n, d = arr.shape
        self.mean = arr.mean(axis=0, keepdims=True)
        centred = arr - self.mean
        cov = (centred.T @ centred) / max(n - 1, 1)

        trace_var = float(np.trace(cov)) / d if np.trace(cov) > 1e-12 else 1e-3
        reg_cov = (1.0 - self.shrinkage) * cov + self.shrinkage * trace_var * np.eye(d)

        # Eigen-decomposition of symmetric positive definite covariance: Sigma = U S U^T
        eigenvalues, eigenvectors = np.linalg.eigh(reg_cov)
        eigenvalues = np.maximum(eigenvalues, 1e-6)
        inv_sqrt_s = 1.0 / np.sqrt(eigenvalues)

        # W = U S^{-1/2} U^T
        self.transform_mat = eigenvectors @ np.diag(inv_sqrt_s) @ eigenvectors.T
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Centres, whitens, and unit-normalizes features."""
        if self.mean is None or self.transform_mat is None:
            raise RuntimeError("WhiteningTransform must be fitted before transform")
        arr = np.asarray(x, dtype=np.float64)
        whitened = (arr - self.mean) @ self.transform_mat
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return whitened / norms


def fit_closed_set_lda(
    target_embs: np.ndarray,
    other_embs: np.ndarray,
    shrink: float = 0.5,
) -> tuple[np.ndarray, float] | None:
    """2-class LDA matching hearing.tse.enrolled.fit_lda contract.

    Returns (w, b) such that w @ target - b = +1 and w @ other - b = -1.
    """
    t, o = np.asarray(target_embs, np.float64), np.asarray(other_embs, np.float64)
    if len(t) == 0 or len(o) == 0:
        return None
    mt, mo = t.mean(axis=0), o.mean(axis=0)
    centred = np.concatenate([t - mt, o - mo], axis=0)
    cov = centred.T @ centred / max(len(centred) - 2, 1)
    d = cov.shape[0]
    var = float(np.trace(cov)) / d if np.trace(cov) > 1e-12 else 1e-3
    cov = (1.0 - shrink) * cov + shrink * var * np.eye(d)
    try:
        w = np.linalg.solve(cov, mt - mo)
    except np.linalg.LinAlgError:
        return None
    half = float(w @ (mt - mo)) / 2.0
    if half <= 1e-12:
        return None
    w = w / half
    b = float(w @ (mt + mo)) / 2.0
    return w, b


def triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Cosine margin triplet loss on unit-normalized representations.

    Loss = max(0, cos(a, n) - cos(a, p) + margin)
    """
    sim_pos = torch.sum(anchor * positive, dim=-1)
    sim_neg = torch.sum(anchor * negative, dim=-1)
    loss = F.relu(sim_neg - sim_pos + margin)
    return loss.mean()


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) for unit-norm embeddings.

    features: [N, D] unit normalized embeddings
    labels: [N] class / speaker labels
    """
    device = features.device
    n = features.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # Cosine similarity matrix scaled by temperature
    sim_matrix = torch.div(torch.matmul(features, features.T), temperature)

    # Exclude self-similarity diagonal
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(n, device=device).view(-1, 1),
        0,
    )
    mask = mask * logits_mask

    # Numerical stability via max subtraction
    logits_max, _ = torch.max(sim_matrix * logits_mask, dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()

    # Denominator: exp of all negatives and distinct positives
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    # Mean over all positive pairs
    mask_sum = mask.sum(1)
    has_positives = mask_sum > 0
    if not has_positives.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    mean_log_prob_pos = (mask * log_prob).sum(1)[has_positives] / mask_sum[has_positives]
    return -mean_log_prob_pos.mean()


def covariance_regularization_loss(features: torch.Tensor, factor: float = 0.01) -> torch.Tensor:
    """Decorrelation / anti-collapse penalty on off-diagonal feature covariances.

    Penalizes dimensional collapse and feature redundancy (VICReg / Barlow Twins formulation).
    """
    n, d = features.shape
    if n <= 1:
        return torch.tensor(0.0, device=features.device)
    centred = features - features.mean(dim=0, keepdim=True)
    cov = (centred.T @ centred) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return factor * (off_diag**2).sum() / d
