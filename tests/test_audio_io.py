"""Audio file/format primitives: load, save, resample, frame, PCM16 conversion."""

import math

import numpy as np
import pytest

from hearing.audio import io as aio
from hearing.testing import make_test_signal


def tone(freq, seconds, sr, amp=0.5):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pcm16_round_trip_is_exact_for_int16_grid_values():
    ints = np.array([-32768, -1, 0, 1, 12345, 32767], dtype=np.int16)
    x = aio.pcm16_to_float(ints)
    assert x.dtype == np.float32
    assert x.min() >= -1.0 and x.max() <= 1.0
    back = np.frombuffer(aio.float_to_pcm16(x), dtype="<i2")
    np.testing.assert_array_equal(back, ints)


def test_pcm16_accepts_bytes_and_is_little_endian():
    raw = np.array([1, -2, 32767], dtype="<i2").tobytes()
    x = aio.pcm16_to_float(raw)
    assert x.shape == (3,)
    assert aio.float_to_pcm16(x) == raw


def test_float_to_pcm16_clips_out_of_range():
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    ints = np.frombuffer(aio.float_to_pcm16(x), dtype="<i2")
    assert ints[0] == -32768
    assert ints[-1] == 32767
    assert ints[2] == 0
    assert -32768 <= ints[1] <= -32767
    assert 32766 <= ints[3] <= 32767


def test_pcm16_round_trip_on_speechlike_signal_is_within_one_lsb():
    x = make_test_signal(0.2, kind="speechlike")
    back = aio.pcm16_to_float(aio.float_to_pcm16(x))
    assert np.max(np.abs(back - x)) <= 1.0 / 32768 + 1e-7


def test_resample_identity_when_rates_match():
    x = tone(1000, 0.1, 16000)
    y = aio.resample(x, 16000, 16000)
    assert y is x or np.array_equal(y, x)


def test_resample_48k_to_16k_keeps_1khz_peak_and_length():
    seconds = 0.5
    x = tone(1000, seconds, 48000)
    y = aio.resample(x, 48000, 16000)
    assert y.dtype == np.float32
    assert abs(len(y) - 16000 * seconds) <= 1
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / 16000)
    assert abs(freqs[np.argmax(spectrum)] - 1000) < 5


def test_resample_scipy_fallback_matches_soxr_closely(monkeypatch):
    x = tone(1000, 0.5, 48000)
    reference = aio.resample(x, 48000, 16000)
    monkeypatch.setattr(aio, "soxr", None)
    y = aio.resample(x, 48000, 16000)
    assert abs(len(y) - len(reference)) <= 1
    n = min(len(y), len(reference))
    # Compare the steady-state middle; filters differ only at the edges.
    assert np.max(np.abs(y[200 : n - 200] - reference[200 : n - 200])) < 0.02


def test_frame_pads_last_chunk():
    x = np.arange(400, dtype=np.float32)
    chunks = list(aio.frame(x, hop=160, pad=True))
    assert len(chunks) == math.ceil(400 / 160) == 3
    assert all(c.shape == (160,) and c.dtype == np.float32 for c in chunks)
    np.testing.assert_array_equal(chunks[0], x[:160])
    np.testing.assert_array_equal(chunks[2][:80], x[320:])
    assert np.all(chunks[2][80:] == 0)


def test_frame_drops_last_chunk_without_pad():
    x = np.arange(400, dtype=np.float32)
    chunks = list(aio.frame(x, hop=160, pad=False))
    assert len(chunks) == 2
    assert all(c.shape == (160,) for c in chunks)


def test_frame_exact_multiple_has_no_extra_chunk():
    x = np.zeros(480, dtype=np.float32)
    assert len(list(aio.frame(x, hop=160, pad=True))) == 3
    assert len(list(aio.frame(np.zeros(0, dtype=np.float32)))) == 0


def test_duration_s():
    assert aio.duration_s(np.zeros(16000, dtype=np.float32), 16000) == pytest.approx(1.0)
    assert aio.duration_s(np.zeros(8000, dtype=np.float32), 16000) == pytest.approx(0.5)


def test_save_and_load_round_trip_pcm16(tmp_path):
    x = make_test_signal(0.25, kind="speechlike")
    path = tmp_path / "a.wav"
    aio.save(path, x, 16000)
    y, sr = aio.load(path)
    assert sr == 16000
    assert y.dtype == np.float32
    assert y.ndim == 1
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) <= 1.0 / 32768 + 1e-7


def test_save_clips_and_load_downmixes_and_resamples(tmp_path):
    import soundfile as sf

    # Clipping on save.
    loud = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    aio.save(tmp_path / "loud.wav", loud, 16000)
    y, _ = aio.load(tmp_path / "loud.wav")
    assert y[0] == pytest.approx(32767 / 32768, abs=1e-6)
    assert y[1] == pytest.approx(-1.0, abs=1e-6)

    # Stereo 48 kHz file -> mono 16 kHz.
    left = tone(1000, 0.5, 48000)
    right = tone(1000, 0.5, 48000, amp=0.25)
    sf.write(tmp_path / "stereo48.wav", np.stack([left, right], axis=1), 48000, subtype="PCM_16")
    y, sr = aio.load(tmp_path / "stereo48.wav", sr=16000)
    assert sr == 16000
    assert y.ndim == 1
    assert abs(len(y) - 8000) <= 1
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), 1 / 16000)
    assert abs(freqs[np.argmax(spectrum)] - 1000) < 5
    # Downmix is the mean of channels: amplitude 0.375 (allow filter ripple).
    assert 0.33 < np.max(np.abs(y[500:-500])) < 0.42


def test_load_native_rate_when_sr_none(tmp_path):
    import soundfile as sf

    sf.write(tmp_path / "n.wav", tone(440, 0.1, 22050), 22050, subtype="PCM_16")
    y, sr = aio.load(tmp_path / "n.wav", sr=None)
    assert sr == 22050
    assert len(y) == 2205
