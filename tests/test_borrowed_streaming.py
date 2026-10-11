"""Offline regression tests for the borrowed continuous-source ordering utility."""

import numpy as np
import pytest

from hearing.tse.borrowed_streaming import SourceContinuityAdapter


def test_swapped_windows_retain_source_order_and_do_not_mutate():
    rng = np.random.default_rng(301)
    sources = rng.uniform(-0.5, 0.5, (2, 960)).astype(np.float32)
    adapter = SourceContinuityAdapter(window_size=640, hop_size=320)
    first = sources[:, :640].copy()
    np.testing.assert_array_equal(adapter.process_window(first), first)
    # Separator swaps output channels and adds DC offsets on the next window.
    second = sources[::-1, 320:960].copy() + np.array([[0.1], [-0.1]], np.float32)
    untouched = second.copy()
    result = adapter.process_window(second)
    np.testing.assert_array_equal(result, second[::-1])
    np.testing.assert_array_equal(second, untouched)
    assert result.dtype == np.float32
    # Neither caller mutation nor a reset should retain previous-session state.
    result[:] = 0
    adapter.reset()
    np.testing.assert_array_equal(adapter.process_window(second), second)


def test_silent_overlap_keeps_current_order():
    adapter = SourceContinuityAdapter(320, 160)
    adapter.process_window(np.zeros((2, 320), np.float32))
    current = np.stack([np.linspace(0, 0.5, 320), np.linspace(-0.5, 0, 320)])
    np.testing.assert_array_equal(adapter.process_window(current), current.astype(np.float32))


@pytest.mark.parametrize('window,hop', [(0, 0), (320, 0), (320, 320), (320, 400)])
def test_invalid_geometry(window, hop):
    with pytest.raises(ValueError):
        SourceContinuityAdapter(window, hop)


def test_invalid_windows_do_not_poison_state():
    adapter = SourceContinuityAdapter(320, 160)
    for invalid in [np.zeros(320), np.zeros((3, 320)), np.full((2, 320), np.nan)]:
        with pytest.raises(ValueError):
            adapter.process_window(invalid)
    valid = np.ones((2, 320), np.float32)
    np.testing.assert_array_equal(adapter.process_window(valid), valid)
