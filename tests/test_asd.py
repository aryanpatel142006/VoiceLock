"""Active-speaker detection: interface, lip-motion fallback and mock (T-036)."""

import time

import numpy as np
import pytest

from hearing.testing import make_test_signal
from hearing.types import AudioChunk, FaceTrack
from hearing.vision import asd
from hearing.vision.asd import LipMotionASD, MockASD, SpeakerDetector, get_asd

SR = 16000
FPS = 25
SPF = SR // FPS  # Audio samples per video frame (40 ms).


def landmarks(opening: float, face_h: float = 0.3) -> np.ndarray:
    """478 MediaPipe-style points; only 10/152 (face) and 13/14 (inner lips) matter."""
    lm = np.full((478, 3), 0.5)
    lm[10, :2] = (0.5, 0.2)
    lm[152, :2] = (0.5, 0.2 + face_h)
    lm[13, :2] = (0.5, 0.4)
    lm[14, :2] = (0.5, 0.4 + opening * face_h)
    return lm


def am_env(t: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(2 * np.pi * 4.0 * t))  # 4 Hz syllable-like envelope.


def scene(speech_s: float, silence_s: float, seed: int = 0) -> np.ndarray:
    """4 Hz amplitude-modulated test signal, then near-silence (-60 dBFS noise)."""
    rng = np.random.default_rng(seed)
    n_sp, n_si = round(speech_s * SR), round(silence_s * SR)
    t = np.arange(n_sp) / SR
    speech = make_test_signal(speech_s, kind="tone") * am_env(t)
    x = np.concatenate([speech, np.zeros(n_si)]) + rng.normal(0, 1e-3, n_sp + n_si)
    return x.astype(np.float32)


def run(det: SpeakerDetector, audio: np.ndarray, mouths, seed: int = 1, jitter: float = 0.002):
    """Feed 40 ms of audio + one landmark frame per step; ``mouths[i](ts, k)`` → opening."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(1, len(audio) // SPF + 1):
        ts = k / FPS
        chunk = AudioChunk(audio[(k - 1) * SPF : k * SPF], SR, k, ts - 1 / FPS)
        tracks = [
            FaceTrack(i, (0.1 + 0.4 * i, 0.2, 0.3, 0.3), landmarks(m(ts, k) + rng.normal(0, jitter)), 0.0, ts)
            for i, m in enumerate(mouths)
        ]
        det.update(tracks, ts, chunk)
        out.append((ts, [t.p_speaking for t in tracks]))
    return out


def synced(ts, k):
    return 0.06 * float(am_env(np.array(ts)))  # Mouth opening in step with the 4 Hz envelope.


def static(ts, k):
    return 0.02


def test_get_asd_names():
    assert isinstance(get_asd("lipmotion"), LipMotionASD)
    assert isinstance(get_asd("mock"), MockASD)
    assert isinstance(get_asd("auto"), SpeakerDetector)
    assert get_asd("auto").name in {"lipmotion", "light_asd"}
    if "light_asd" not in asd._FACTORIES:
        with pytest.raises(KeyError):
            get_asd("light_asd")
    with pytest.raises(KeyError, match="lipmotion"):
        get_asd("nope")
    assert get_asd("lipmotion") is not get_asd("lipmotion")  # Fresh state per call.


def test_synced_mouth_high_static_low_then_silence_low():
    speech_s, silence_s = 2.0, 1.6
    det = get_asd("lipmotion")
    # The synced mouth keeps moving through the silence (chewing, mouthing words).
    out = run(det, scene(speech_s, silence_s), [synced, static])
    during = [(ts, p) for ts, p in out if 0.6 <= ts <= speech_s]
    assert min(p[0] for _, p in during) >= 0.8, during
    assert max(p[1] for ts, p in out if ts <= speech_s) <= 0.3
    # activity_window_s (1.0 s) is the hangover: past it, a mouth moving in silence reads low.
    late_silence = [p for ts, p in out if ts >= speech_s + det.activity_window_s + 0.3]
    assert max(max(p) for p in late_silence) <= 0.3, late_silence
    assert det.confidence == {0: 1.0, 1: 1.0}


def test_both_low_in_silence_from_start():
    det = LipMotionASD()
    out = run(det, scene(0.0, 1.5), [synced, static])
    assert max(max(p) for _, p in out) <= 0.3


def test_speechlike_signal_with_mouth_following_its_envelope():
    x = make_test_signal(3.0, kind="speechlike", seed=3)
    env = np.sqrt(np.convolve(x.astype(np.float64) ** 2, np.ones(SPF) / SPF, mode="same"))
    env /= env.max()
    det = LipMotionASD()
    out = run(det, x, [lambda ts, k: 0.06 * float(env[min(k * SPF - SPF // 2, len(env) - 1)]), static])
    talking = [p for ts, p in out if ts >= 0.6]
    assert np.mean([p[0] for p in talking]) >= 0.7
    assert max(p[1] for p in talking) <= 0.3


def test_track_without_landmarks_uses_audio_only_capped():
    det = LipMotionASD()
    audio = scene(1.0, 0.0)
    for k in range(1, 26):
        ts = k / FPS
        tr = [FaceTrack(5, (0.3, 0.3, 0.2, 0.2), None, 0.0, ts)]
        det.update(tr, ts, AudioChunk(audio[(k - 1) * SPF : k * SPF], SR, k, ts))
    assert 0.3 < tr[0].p_speaking <= det.audio_only_max
    assert det.confidence[5] == 0.0


def test_video_only_before_any_audio():
    det = LipMotionASD()
    out = run(det, np.zeros(SR, dtype=np.float32), [synced, static])
    det2 = LipMotionASD()
    for k in range(1, 26):
        ts = k / FPS
        tr = [FaceTrack(0, (0, 0, 0.3, 0.3), landmarks(synced(ts, k)), 0.0, ts)]
        det2.update(tr, ts, None)
    assert tr[0].p_speaking >= 0.8  # Moving lips with no audio stream at all.
    assert out[-1][1][1] <= 0.3


def test_reset_clears_state_and_dropped_tracks_are_forgotten():
    det = LipMotionASD()
    run(det, scene(1.0, 0.0), [synced, static])
    assert det._tracks and det._floor is not None
    det.reset()
    assert not det._tracks and det._floor is None and not det.confidence
    assert not det.audio_energies and not det._pending
    # After reset, silence behaves as if fresh.
    out = run(det, scene(0.0, 0.4), [synced])
    assert out[-1][1][0] <= 0.3
    det.update([], 10.0, None)
    assert not det._tracks


def test_chunking_audio_in_10ms_hops_matches_frame_chunks():
    audio = scene(1.0, 0.0)
    a, b = LipMotionASD(), LipMotionASD()
    for k in range(1, 26):
        ts = k / FPS
        seg = audio[(k - 1) * SPF : k * SPF]
        ta = [FaceTrack(0, (0, 0, 0.3, 0.3), landmarks(synced(ts, k)), 0.0, ts)]
        tb = [FaceTrack(0, (0, 0, 0.3, 0.3), landmarks(synced(ts, k)), 0.0, ts)]
        a.update(ta, ts, AudioChunk(seg, SR, k, ts))
        for j in range(0, SPF, 160):
            b.push_audio(AudioChunk(seg[j : j + 160], SR, k, ts))
        b.update(tb, ts, None)
        assert ta[0].p_speaking == pytest.approx(tb[0].p_speaking, abs=1e-9)


def test_mock_returns_script():
    det = get_asd("mock", script={1: 0.9, 2: 0.1}, default=0.3)
    tracks = [FaceTrack(i, (0, 0, 0.1, 0.1), None, 0.0, 0.0) for i in (1, 2, 3)]
    assert det.update(tracks, 0.0, None) is tracks
    assert [t.p_speaking for t in tracks] == [0.9, 0.1, 0.3]
    det.reset()
    assert det.calls == 0


def test_update_time_two_tracks_under_1ms():
    det = LipMotionASD()
    audio = scene(4.0, 0.0)
    rng = np.random.default_rng(0)
    n = len(audio) // SPF
    times = []
    for k in range(1, n + 1):
        ts = k / FPS
        chunk = AudioChunk(audio[(k - 1) * SPF : k * SPF], SR, k, ts)
        tracks = [
            FaceTrack(i, (0.1 + 0.4 * i, 0.2, 0.3, 0.3), landmarks(0.03 + rng.normal(0, 0.01)), 0.0, ts)
            for i in range(2)
        ]
        t0 = time.perf_counter()
        det.update(tracks, ts, chunk)
        times.append(time.perf_counter() - t0)
    p50, p95 = (1e3 * float(np.percentile(times[13:], q)) for q in (50, 95))
    print(f"\nLipMotionASD.update, 2 tracks, 40 ms audio: p50 {p50:.3f} ms, p95 {p95:.3f} ms")
    assert p50 <= 1.0
