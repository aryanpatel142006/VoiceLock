"""Opt-in source-window adapter for T-301 research, used from a source checkout.

No model, Stage registration, target selection or production defaults are added.
The vendored MIT utility only keeps two already-separated channels continuous.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


class SourceContinuityAdapter:
    """Order successive unweighted (2, window_size) float32 windows.

    Call once per separator window, advancing exactly hop_size samples each time;
    reset on session/model/geometry changes. This is a window-level helper, not a
    10 ms waveform Stage. Feed its result to the existing identity matcher and OLA.
    """

    def __init__(self, window_size: int, hop_size: int):
        if not isinstance(window_size, int) or not isinstance(hop_size, int):
            raise TypeError('window and hop must be integer sample counts')
        if not 0 < hop_size < window_size:
            raise ValueError('need 0 < hop_size < window_size')
        path = Path(__file__).resolve().parents[2] / 'third_party/borrowed/T-301/source_order.py'
        spec = spec_from_file_location('_t301_source_order', path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load borrowed module at {path}')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        self._reorder = module.reorder_sources
        self.window_size = window_size
        self.hop_size = hop_size
        self.reset()

    def reset(self) -> None:
        self._previous = None

    def process_window(self, sources: np.ndarray) -> np.ndarray:
        current = np.asarray(sources, dtype=np.float32)
        if current.shape != (2, self.window_size) or not np.isfinite(current).all():
            raise ValueError(f'expected finite sources of shape (2, {self.window_size})')
        if self._previous is None:
            ordered = current.copy()
        else:
            ordered = self._reorder(current, self._previous, self.window_size, self.hop_size)
        self._previous = ordered.copy()
        return ordered
