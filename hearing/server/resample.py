"""Low-latency streaming resampler for client audio at rates other than 16 kHz.

soxr's stream API works in internal blocks: fed 10 ms chunks it returns nothing for
three calls, then ~30 ms at once. That jitter is unacceptable on the live path, so this
is a plain causal polyphase FIR (Kaiser-windowed sinc) with explicit history: every
call returns exactly the samples its input makes computable, and chunked output equals
whole-signal output.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import firwin


class StreamResampler:
    """Rational resampler ``sr_in -> sr_out`` with ``taps`` filter taps per output sample.

    Output ``n`` sits at upsampled time ``n * M`` (``L/M = sr_out/sr_in``) and uses the
    ``taps`` most recent inputs, so it is computable once input ``n * M // L`` arrives.
    ``delay_ms`` is the filter's group delay.
    """

    def __init__(self, sr_in: int, sr_out: int, taps: int = 32, beta: float = 8.0):
        if sr_in <= 0 or sr_out <= 0:
            raise ValueError(f"sample rates must be positive, got {sr_in} -> {sr_out}")
        g = gcd(sr_in, sr_out)
        self.sr_in, self.sr_out = sr_in, sr_out
        self.up, self.down = sr_out // g, sr_in // g
        self.taps = taps
        n = taps * self.up
        if self.up == self.down == 1:
            self.delay_ms = 0.0
        else:
            # Cut off slightly below the lower Nyquist so the transition band does not alias.
            cutoff = 0.9 / max(self.up, self.down)
            h = firwin(n, cutoff, window=("kaiser", beta)) * self.up
            self._phases = h.reshape(taps, self.up).T.astype(np.float64)  # [phase, k] = h[p + k*L]
            self.delay_ms = 1000.0 * (n - 1) / 2 / (self.up * sr_in)
        self.reset()

    def reset(self) -> None:
        self._hist = np.zeros(self.taps - 1, dtype=np.float64)  # inputs before the block
        self._consumed = 0  # absolute index of the first input after the history
        self._next = 0  # absolute index of the next output

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.up == self.down == 1:
            return x
        buf = np.concatenate([self._hist, x.astype(np.float64)])
        start = self._consumed - (self.taps - 1)  # absolute index of buf[0]
        available = self._consumed + len(x)
        last = -(-available * self.up // self.down)  # first output not yet computable
        n = np.arange(self._next, last, dtype=np.int64)
        y = np.zeros(len(n), dtype=np.float64)
        if len(n):
            u = n * self.down
            newest = u // self.up - start  # buf index of each output's newest input
            window = newest[:, None] - np.arange(self.taps)[None, :]
            y = np.einsum("ij,ij->i", self._phases[u % self.up], buf[window])
        self._next = last
        self._consumed = available
        self._hist = buf[len(buf) - (self.taps - 1) :]
        return y.astype(np.float32)
