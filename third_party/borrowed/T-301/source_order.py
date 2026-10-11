"""Two-source NumPy adaptation of Asteroid's overlap correlation reordering.

Borrowed from asteroid/dsp/overlap_add.py::_reorder_sources at
https://github.com/asteroid-team/asteroid/tree/c15708a04d3d28e9a1cd50553c456299dbc6d236
Copyright (c) 2019 Pariente Manuel. SPDX-License-Identifier: MIT
The full upstream permission notice is in LICENSE alongside this module.

Modifications (T-301): unbatched NumPy input instead of flattened Torch tensors;
explicit two-permutation search instead of PITReorder; geometry/finite checks;
float64 correlation accumulation; stable identity ordering on ties. The upstream
mean-centering and negative cross-correlation objective are retained.
"""

import numpy as np


def reorder_sources(current, previous, window_size, hop_size):
    """Return a copy of current (2, window_size), ordered like previous.

    Both windows must be unweighted and exactly hop_size samples apart.
    Correlation is continuity evidence, never proof of speaker identity.
    """
    if not 0 < hop_size < window_size:
        raise ValueError('need 0 < hop_size < window_size')
    current = np.asarray(current, dtype=np.float32)
    previous = np.asarray(previous, dtype=np.float32)
    expected = (2, window_size)
    if current.shape != expected or previous.shape != expected:
        raise ValueError(f'expected two windows of shape {expected}')
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError('source windows must be finite')

    overlap_f = window_size - hop_size
    x = current[:, :overlap_f].astype(np.float64)
    y = previous[:, -overlap_f:].astype(np.float64)
    # Upstream reorder_func: mean normalization, then negative correlation.
    x = x - x.mean(-1, keepdims=True)
    y = y - y.mean(-1, keepdims=True)
    costs = -np.sum(y[:, None, :] * x[None, :, :], axis=-1)
    identity_cost = costs[0, 0] + costs[1, 1]
    swapped_cost = costs[0, 1] + costs[1, 0]
    return current[::-1].copy() if swapped_cost < identity_cost else current.copy()
