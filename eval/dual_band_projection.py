"""T-283 offline dual-band metric learning; never changes live matching defaults."""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from scipy.signal import firwin, lfilter
from torch import nn
from torch.nn import functional as F


class DualBandSplitter:
    """Complementary causal FIR bands at 16 kHz with a shared 4 ms delay.

    Low: approximately 0–2 kHz; high: delayed input minus low, up to Nyquist.
    The transition around 2 kHz is finite, not an ideal brick-wall filter.
    """
    sample_rate = 16000

    def __init__(self) -> None:
        self.taps = firwin(129, 2000, fs=self.sample_rate)
        self.delay_samples = (len(self.taps) - 1) // 2
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(len(self.taps) - 1)
        self._delay = np.zeros(self.delay_samples, dtype=np.float32)

    def process(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(audio, dtype=np.float32)
        if x.ndim != 1 or not np.isfinite(x).all() or np.any(np.abs(x) > 1):
            raise ValueError('audio must be finite mono samples in [-1, 1] at 16 kHz')
        if not x.size:
            return x.copy(), x.copy()
        low, self._state = lfilter(self.taps, [1.0], x, zi=self._state)
        delayed = np.concatenate((self._delay, x))
        self._delay = delayed[-self.delay_samples:].copy()
        low = low.astype(np.float32)
        return low, (delayed[:len(x)] - low).astype(np.float32)


def extract_band_embeddings(
    audio: np.ndarray, encoder: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Call a frozen, externally loaded encoder twice; no downloads or gradients.

    Caller must explicitly resample to 16 kHz first. Each utterance starts with
    fresh FIR state. Flush and trim delay so both encoder crops align to input.
    No independent band gain normalization is performed.
    """
    if np.asarray(audio).size == 0:
        raise ValueError('empty utterance')
    splitter = DualBandSplitter()
    low, high = splitter.process(audio)
    tail = splitter.process(np.zeros(splitter.delay_samples, dtype=np.float32))
    pair = []
    for band, end in zip((low, high), tail, strict=True):
        aligned = np.concatenate((band, end))[splitter.delay_samples:]
        embedding = np.asarray(encoder(aligned), dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(embedding)
        if not np.isfinite(embedding).all() or norm <= 1e-12:
            raise ValueError('encoder returned an invalid or zero embedding')
        pair.append(embedding / norm)
    return np.stack(pair)


class DualBandProjection(nn.Module):
    """Two linear branches and a per-crop learned softmax mixture gate.

    Gate sees normalized frozen embeddings. Output is a unit metric vector;
    an all-zero fused vector remains zero and must be rejected by consumers.
    """
    def __init__(self, in_dim: int = 192, out_dim: int = 32) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 2:
            raise ValueError('positive input dimension and output dimension >= 2 required')
        self.low = nn.Linear(in_dim, out_dim, bias=False)
        self.high = nn.Linear(in_dim, out_dim, bias=False)
        self.gate = nn.Linear(in_dim * 2, 2)

    def forward(self, low: torch.Tensor, high: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if low.ndim != 2 or low.shape != high.shape or low.shape[1] != self.low.in_features:
            raise ValueError('bands must have matching [batch, in_dim] shapes')
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise ValueError('embeddings must be finite')
        low, high = F.normalize(low, dim=-1), F.normalize(high, dim=-1)
        gate = self.gate(torch.cat((low, high), dim=-1)).softmax(dim=-1)
        z = gate[:, :1] * self.low(low) + gate[:, 1:] * self.high(high)
        return F.normalize(z, dim=-1), gate


def train_head(
    head: DualBandProjection, low: torch.Tensor, high: torch.Tensor,
    labels: torch.Tensor, *, epochs: int = 100, lr: float = 0.001,
) -> list[float]:
    """Full-batch cosine pair loss: same-speaker -> 1, others -> <= 0.2.

    Cached encoder inputs are detached; only branches and gate are optimized.
    Returned history includes the initial and final evaluated training loss.
    """
    if epochs < 1 or not np.isfinite(lr) or lr <= 0:
        raise ValueError('positive epochs and finite learning rate required')
    if labels.ndim != 1 or len(labels) != len(low) or len(low) < 4:
        raise ValueError('at least four labeled rows required')
    _, counts = labels.unique(return_counts=True)
    if len(counts) < 2 or (counts < 2).any():
        raise ValueError('at least two speakers with two crops each required')
    device = next(head.parameters()).device
    low, high = low.detach().to(device), high.detach().to(device)
    labels = labels.to(device)
    same = labels[:, None] == labels[None, :]
    positive = same & ~torch.eye(len(labels), dtype=torch.bool, device=device)
    negative = ~same
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    history = []
    head.train()
    for step in range(epochs + 1):
        z, _ = head(low, high)
        sim = z @ z.T
        loss = (1 - sim[positive]).square().mean() + F.relu(sim[negative] - 0.2).square().mean()
        history.append(float(loss.detach()))
        if step < epochs:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    head.eval()
    return history


def main() -> None:
    """Train explicit local NPZ cache (low/high [N,D], labels [N]); save outside git."""
    from hearing.device import pick

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = pick()
    with np.load(args.cache, allow_pickle=False) as cache:
        low = torch.tensor(cache['low'], dtype=torch.float32)
        high = torch.tensor(cache['high'], dtype=torch.float32)
        labels = torch.tensor(cache['labels'], dtype=torch.long)
    head = DualBandProjection(in_dim=low.shape[1]).to(device)
    history = train_head(head, low, high, labels, epochs=args.epochs)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': head.state_dict(), 'in_dim': low.shape[1], 'out_dim': 32,
                'sample_rate': 16000, 'cutoff_hz': 2000, 'seed': args.seed}, args.checkpoint)
    print(json.dumps({'device': str(device), 'rows': len(low), 'initial_loss': history[0],
                      'final_loss': history[-1], 'evaluation': 'training_only'}))


if __name__ == '__main__':
    main()
