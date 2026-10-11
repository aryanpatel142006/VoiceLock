"""Streaming buffers: RingBuffer, Chunker, chunk_stream."""

import numpy as np
import pytest

from hearing.audio.stream import Chunker, RingBuffer, chunk_stream
from hearing.types import AudioChunk


def test_ring_buffer_push_pull_basic():
    rb = RingBuffer(8)
    assert rb.available == 0
    assert rb.capacity == 8
    rb.push(np.arange(5, dtype=np.float32))
    assert rb.available == 5
    out = rb.pull(3)
    np.testing.assert_array_equal(out, [0, 1, 2])
    assert out.dtype == np.float32
    assert rb.available == 2


def test_ring_buffer_wraps_around_past_capacity():
    rb = RingBuffer(8)
    rb.push(np.arange(6, dtype=np.float32))  # write head at 6
    rb.pull(5)  # read head at 5, 1 sample left
    rb.push(np.arange(10, 16, dtype=np.float32))  # crosses the end: 6,7 then 0..3
    assert rb.available == 7
    np.testing.assert_array_equal(rb.pull(7), [5, 10, 11, 12, 13, 14, 15])
    assert rb.available == 0
    # Many laps.
    for lap in range(20):
        rb.push(np.full(5, lap, dtype=np.float32))
        np.testing.assert_array_equal(rb.pull(5), np.full(5, lap))


def test_ring_buffer_overflow_raises_by_default():
    rb = RingBuffer(4)
    rb.push(np.zeros(3, dtype=np.float32))
    with pytest.raises(OverflowError):
        rb.push(np.zeros(2, dtype=np.float32))
    # The buffer is unchanged after a rejected push.
    assert rb.available == 3


def test_ring_buffer_overflow_drops_oldest_when_requested():
    rb = RingBuffer(4, overflow="drop_oldest")
    rb.push(np.arange(3, dtype=np.float32))
    rb.push(np.arange(10, 13, dtype=np.float32))  # 6 samples into 4: keep newest 4
    assert rb.available == 4
    np.testing.assert_array_equal(rb.pull(4), [2, 10, 11, 12])
    # A push larger than capacity keeps only the newest ``capacity`` samples.
    rb.push(np.arange(100, 110, dtype=np.float32))
    np.testing.assert_array_equal(rb.pull(4), [106, 107, 108, 109])


def test_ring_buffer_pull_more_than_available_raises():
    rb = RingBuffer(4)
    rb.push(np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError):
        rb.pull(3)


def test_ring_buffer_clear():
    rb = RingBuffer(4)
    rb.push(np.ones(3, dtype=np.float32))
    rb.clear()
    assert rb.available == 0
    rb.push(np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(rb.pull(4), [0, 1, 2, 3])


def test_ring_buffer_rejects_bad_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)


def test_chunker_yields_exact_hops_and_keeps_remainder():
    ch = Chunker(160)
    x = np.arange(480, dtype=np.float32)
    out = []
    out += list(ch.push(x[:100]))
    assert out == []
    assert ch.remainder == 100
    out += list(ch.push(x[100:350]))
    assert len(out) == 2
    assert ch.remainder == 30
    out += list(ch.push(x[350:480]))
    assert len(out) == 3
    assert ch.remainder == 0
    assert all(c.shape == (160,) and c.dtype == np.float32 for c in out)
    np.testing.assert_array_equal(np.concatenate(out), x)


def test_chunker_flush_pads_and_resets():
    ch = Chunker(160)
    assert list(ch.push(np.ones(50, dtype=np.float32))) == []
    tail = ch.flush()
    assert tail is not None and tail.shape == (160,)
    assert np.all(tail[:50] == 1) and np.all(tail[50:] == 0)
    assert ch.remainder == 0
    assert ch.flush() is None
    ch.push(np.ones(10, dtype=np.float32))
    ch.reset()
    assert ch.remainder == 0


def test_chunker_handles_pushes_larger_than_several_hops():
    ch = Chunker(160)
    out = list(ch.push(np.zeros(1000, dtype=np.float32)))
    assert len(out) == 6
    assert ch.remainder == 40


def test_chunk_stream_seq_and_timestamps():
    x = np.arange(16000, dtype=np.float32) / 16000
    chunks = list(chunk_stream(x, 16000, hop_ms=10, t0=1.5))
    assert len(chunks) == 100
    assert all(isinstance(c, AudioChunk) for c in chunks)
    assert [c.seq for c in chunks] == list(range(100))
    ts = np.array([c.capture_ts for c in chunks])
    np.testing.assert_allclose(np.diff(ts), 0.010, atol=1e-9)
    assert ts[0] == pytest.approx(1.5)
    assert all(c.sample_rate == 16000 and c.samples.shape == (160,) for c in chunks)
    np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), x)


def test_chunk_stream_pads_last_chunk_and_supports_other_hops():
    x = np.ones(1000, dtype=np.float32)
    chunks = list(chunk_stream(x, 16000, hop_ms=20))
    assert len(chunks) == 4  # 320-sample hops, last one padded
    assert chunks[-1].samples.shape == (320,)
    assert np.all(chunks[-1].samples[40:] == 0)
    assert chunks[1].capture_ts == pytest.approx(0.020)


def test_chunk_stream_rejects_non_integer_hop():
    with pytest.raises(ValueError):
        list(chunk_stream(np.zeros(100, dtype=np.float32), 16000, hop_ms=10.3))
