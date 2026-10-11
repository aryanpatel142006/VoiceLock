"""Small causal streaming separator for distillation (T-424): causal Conv-TasNet, fixed slot order.

Slot 0 = Aryan, slot 1 = Joseph (as the T-374/T-387 fixed-order teacher), slot 2 = the residual
(noise, strangers): softmax masks over the three, so the slots always add up to the mixture's
encoding and the net cannot go silent everywhere (two from-scratch losses collapsed that way,
see distill.py). ``net(x)`` returns slots 0-1, ``net(x, residual=True)`` all three. Encoder: a learned
1-D conv with window ``win`` samples and hop ``win // 2``; separator: ``repeats`` x ``blocks``
dilated depthwise-separable conv blocks, every conv left-padded (no future frames), every norm
per frame (channel-wise layer norm, no statistics over time); decoder: transposed conv
(overlap-add). Output sample n depends only on input samples <= n + win - 1 rounded to the
frame grid, so the algorithmic latency is one window (``latency_ms``): 16 ms for win=256.

Run on a whole signal it equals running frame by frame with carried state (causal convs), so
``forward`` on a full scene *is* the streaming output at the model's own latency; the unit test
checks causality by comparing prefixes. Self-contained: torch only.
"""

from __future__ import annotations

import torch
from torch import nn

SR = 16000

# name -> kwargs; params measured with ``count_params`` (see docs/research/DISTILL_STUDENT.md)
VARIANTS = {
    "tcn16": {"win": 256, "enc": 256, "bottleneck": 128, "hidden": 256, "blocks": 8, "repeats": 3},
    "tcn32": {"win": 512, "enc": 512, "bottleneck": 128, "hidden": 256, "blocks": 6, "repeats": 3},
    "tcn8s": {"win": 128, "enc": 256, "bottleneck": 96, "hidden": 192, "blocks": 8, "repeats": 2},
}


class FrameNorm(nn.Module):
    """Layer norm over channels of each frame (causal: no time statistics)."""

    def __init__(self, ch):
        super().__init__()
        self.norm = nn.LayerNorm(ch)

    def forward(self, x):  # [B, C, T]
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, b, h, dilation, kernel=3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.inp = nn.Sequential(nn.Conv1d(b, h, 1), nn.PReLU(), FrameNorm(h))
        self.dw = nn.Conv1d(h, h, kernel, dilation=dilation, groups=h)
        self.post = nn.Sequential(nn.PReLU(), FrameNorm(h))
        self.res = nn.Conv1d(h, b, 1)
        self.skip = nn.Conv1d(h, b, 1)

    def forward(self, x):
        y = self.inp(x)
        y = self.post(self.dw(nn.functional.pad(y, (self.pad, 0))))
        return x + self.res(y), self.skip(y)


class CausalTasNet(nn.Module):
    def __init__(self, win=256, enc=256, bottleneck=128, hidden=256, blocks=8, repeats=3, spk=3):
        super().__init__()
        self.win, self.hop, self.spk = win, win // 2, spk
        self.encoder = nn.Conv1d(1, enc, win, stride=self.hop, bias=False)
        self.inorm = FrameNorm(enc)
        self.inp = nn.Conv1d(enc, bottleneck, 1)
        self.blocks = nn.ModuleList(
            Block(bottleneck, hidden, 2**i) for _ in range(repeats) for i in range(blocks)
        )
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(bottleneck, spk * enc, 1))
        self.decoder = nn.ConvTranspose1d(enc, 1, win, stride=self.hop, bias=False)

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.win / SR

    @property
    def receptive_s(self) -> float:
        frames = 1 + sum(blk.pad for blk in self.blocks)
        return frames * self.hop / SR

    def forward(self, mix, residual=False):  # [B, T] -> [B, 2 (or spk), T]
        n = mix.shape[-1]
        # left pad one hop so frame k covers samples [k*hop - hop, k*hop + hop); right pad to the grid
        right = (-(n + self.hop - self.win)) % self.hop
        x = nn.functional.pad(mix[:, None], (self.hop, right + self.hop))
        w = torch.relu(self.encoder(x))  # [B, N, F]
        y = self.inp(self.inorm(w))
        skip = 0
        for blk in self.blocks:
            y, s = blk(y)
            skip = skip + s
        m = torch.softmax(self.mask(skip).view(w.shape[0], self.spk, w.shape[1], w.shape[2]), dim=1)
        est = self.decoder((m * w[:, None]).flatten(0, 1))  # [B*spk, 1, T']
        est = est.view(mix.shape[0], self.spk, -1)[..., self.hop : self.hop + n]
        est = est + (mix[:, None] - est.sum(dim=1, keepdim=True)) / self.spk
        return est if residual else est[:, :2]


def build(variant: str) -> CausalTasNet:
    return CausalTasNet(**VARIANTS[variant])


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())


def load(path, map_location="cpu") -> CausalTasNet:
    state = torch.load(path, map_location=map_location, weights_only=True)
    net = build(state["variant"])
    net.load_state_dict(state["model"])
    return net.eval()


def _depthwise(conv, ctx, n):
    """``conv(ctx)`` for a dilated depthwise conv as a sum of shifted slices: torch's CPU path
    (``slow_conv_dilated2d``, one call per channel) took 100-200 ms per 10 ms chunk."""
    d, k = conv.dilation[0], conv.kernel_size[0]
    w = conv.weight[:, 0]  # [H, K]
    out = conv.bias[:, None] + w[:, 0, None] * ctx[..., :n]
    for j in range(1, k):
        out = out + w[:, j, None] * ctx[..., j * d : j * d + n]
    return out


class Streamer:
    """Frame-by-frame inference with carried state; output equals ``net(x)`` (tests check it).

    ``process(chunk)`` takes any number of samples and returns ``[spk, k]`` samples (slot 2 =
    residual), the k the
    model can emit so far; in total the output trails the input by one window (``latency_ms``).
    """

    def __init__(self, net: CausalTasNet, device="cpu"):
        self.net, self.dev = net.eval(), torch.device(device)
        self.reset()

    def reset(self):
        net = self.net
        self.inbuf = torch.zeros(net.hop, device=self.dev)  # the forward's left pad
        self.pending = torch.zeros(0, device=self.dev)  # input not yet matched by output
        self.caches = [None] * len(net.blocks)
        self.tail = None
        self.drop = net.hop  # the forward trims the first hop of the decoder output

    @torch.no_grad()
    def process(self, chunk) -> torch.Tensor:
        net = self.net
        x = torch.as_tensor(chunk, dtype=torch.float32, device=self.dev).reshape(-1)
        self.inbuf = torch.cat([self.inbuf, x])
        self.pending = torch.cat([self.pending, x])
        n = (len(self.inbuf) - net.win) // net.hop + 1
        if n <= 0:
            return torch.zeros(net.spk, 0)
        used = self.inbuf[: (n - 1) * net.hop + net.win]
        self.inbuf = self.inbuf[n * net.hop :]
        w = torch.relu(net.encoder(used[None, None]))  # [1, N, n]
        y = net.inp(net.inorm(w))
        skip = 0
        for i, blk in enumerate(net.blocks):
            h = blk.inp(y)
            if self.caches[i] is None:
                self.caches[i] = h.new_zeros(h.shape[0], h.shape[1], blk.pad)
            ctx = torch.cat([self.caches[i], h], dim=-1)
            self.caches[i] = ctx[..., ctx.shape[-1] - blk.pad :]
            h = blk.post(_depthwise(blk.dw, ctx, h.shape[-1]))
            y, skip = y + blk.res(h), skip + blk.skip(h)
        m = torch.softmax(net.mask(skip).view(1, net.spk, w.shape[1], n), dim=1)
        out = net.decoder((m * w[:, None]).flatten(0, 1))[:, 0]  # [spk, (n-1)*hop + win]
        if self.tail is not None:
            out[:, : self.tail.shape[1]] += self.tail
        ready, self.tail = out[:, : n * net.hop], out[:, n * net.hop :]
        if self.drop:
            k = min(self.drop, ready.shape[1])
            ready, self.drop = ready[:, k:], self.drop - k
        mix, self.pending = self.pending[: ready.shape[1]], self.pending[ready.shape[1] :]
        ready = ready + (mix - ready.sum(dim=0)) / net.spk  # mixture consistency, as forward
        return ready.cpu()
