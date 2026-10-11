"""Behavior checks for an offline locked-identity tracker (no audio devices)."""
import numpy as np
import pytest

from eval.dynamic_ema_tracker import DynamicEMATracker


def test_confidence_controls_adaptation_speed():
    # Removing the confidence factor would make these equal.
    slow = DynamicEMATracker([1, 0])
    fast = DynamicEMATracker([1, 0])
    a = slow.update([[0.8, 0.6], [0.7, np.sqrt(0.51)]], speech_probability=1)
    b = fast.update([[0.8, 0.6], [-1, 0]], speech_probability=1)
    assert 0 < a.alpha < b.alpha < 1
    assert fast.embedding[1] > slow.embedding[1] > 0
    assert np.linalg.norm(fast.embedding) == pytest.approx(1)


@pytest.mark.parametrize('kwargs, reason', [
    ({'speech_probability': 0.1}, 'silence'),
    ({'speech_probability': 1, 'laughter': True}, 'laughter'),
])
def test_non_speech_freezes_identity_without_selecting_a_stale_index(kwargs, reason):
    tracker = DynamicEMATracker([1, 0])
    tracker.update([[0.8, 0.6], [0, 1]], speech_probability=1)
    before = tracker.embedding
    for _ in range(20):
        result = tracker.update([[0, 1], [1, 0]], **kwargs)
        assert result.index is None and result.alpha == 0 and result.reason == reason
    np.testing.assert_array_equal(tracker.embedding, before)


def test_permuted_separator_stems_reacquire_same_speaker_after_pause():
    tracker = DynamicEMATracker([1, 0], negatives=[[0, 1]])
    assert tracker.update([[1, 0], [0, 1]], speech_probability=1).index == 0
    tracker.update([[0, 1], [1, 0]], speech_probability=0)
    assert tracker.update([[0, 1], [1, 0]], speech_probability=1).index == 1


@pytest.mark.parametrize('streams, negatives, reason', [
    ([[0, 1], [-1, 0]], None, 'identity_floor'),
    ([[1, 0], [1, 0]], None, 'stream_margin'),
    ([[0.8, 0.6], [-1, 0]], [[0.8, 0.6]], 'negative_margin'),
])
def test_ambiguous_or_wrong_identity_cannot_contaminate_memory(streams, negatives, reason):
    tracker = DynamicEMATracker([1, 0], negatives=negatives)
    result = tracker.update(streams, speech_probability=1)
    assert result.index is None and result.reason == reason
    np.testing.assert_array_equal(tracker.embedding, [1, 0])


def test_long_pause_does_not_increase_first_resumed_update_weight():
    a, b = DynamicEMATracker([1, 0]), DynamicEMATracker([1, 0])
    for _ in range(100):
        a.update(None, speech_probability=0, hop_s=1)
    ra = a.update([[0.8, 0.6], [-1, 0]], speech_probability=1, hop_s=0.5)
    rb = b.update([[0.8, 0.6], [-1, 0]], speech_probability=1, hop_s=0.5)
    assert ra.alpha == rb.alpha
    np.testing.assert_array_equal(a.embedding, b.embedding)


def test_hop_weight_scales_with_elapsed_speech_time():
    a = DynamicEMATracker([1, 0]).update([[1, 0], [0, 1]], speech_probability=1)
    b = DynamicEMATracker([1, 0]).update(
        [[1, 0], [0, 1]], speech_probability=1, hop_s=0.5)
    assert 1 - a.alpha == pytest.approx((1 - b.alpha) ** 2)


def test_state_is_defensive_and_reset_restores_enrollment():
    enrollment = np.array([1., 0.])
    tracker = DynamicEMATracker(enrollment)
    enrollment[:] = 0
    tracker.update([[0.8, 0.6], [-1, 0]], speech_probability=1)
    leaked = tracker.embedding
    leaked[:] = 0
    assert np.linalg.norm(tracker.embedding) == pytest.approx(1)
    tracker.reset()
    np.testing.assert_array_equal(tracker.embedding, [1, 0])


@pytest.mark.parametrize('streams', [[], [[0, 0]], [[np.nan, 1]], [[1, 0, 0]], [1, 0]])
def test_bad_embeddings_raise_without_mutating_state(streams):
    tracker = DynamicEMATracker([1, 0])
    with pytest.raises(ValueError):
        tracker.update(streams, speech_probability=1)
    np.testing.assert_array_equal(tracker.embedding, [1, 0])


@pytest.mark.parametrize('kwargs', [
    {'speech_probability': np.nan}, {'speech_probability': 1.1},
    {'speech_probability': 1, 'hop_s': 0}, {'speech_probability': 1, 'hop_s': np.inf},
])
def test_bad_activity_or_hop_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DynamicEMATracker([1, 0]).update([[1, 0]], **kwargs)


@pytest.mark.parametrize('kwargs', [
    {'alpha_min': -1}, {'alpha_max': 1}, {'alpha_min': 0.7, 'alpha_max': 0.2},
    {'margin_min': 0.2, 'margin_full': 0.1}, {'identity_floor': np.nan},
    {'reference_hop_s': 0}, {'speech_floor': 2}, {'negative_margin': -1},
])
def test_bad_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        DynamicEMATracker([1, 0], **kwargs)


def test_time_consistency_of_identical_evidence_with_fixed_confidence():
    # Re-normalizing the accumulator on each hop would break this equality.
    full = DynamicEMATracker([1, 0], alpha_min=0.2, alpha_max=0.2)
    half = DynamicEMATracker([1, 0], alpha_min=0.2, alpha_max=0.2)
    full.update([[0.8, 0.6]], speech_probability=1, hop_s=1)
    for _ in range(2):
        half.update([[0.8, 0.6]], speech_probability=1, hop_s=0.5)
    np.testing.assert_allclose(full.embedding, half.embedding, atol=1e-12)


def test_single_candidate_still_requires_absolute_identity():
    tracker = DynamicEMATracker([1, 0])
    assert tracker.update([[0, 1]], speech_probability=1).index is None
    assert tracker.update([[1, 0]], speech_probability=1).index == 0


@pytest.mark.parametrize('target, negatives', [
    ([0, 0], None), ([np.inf, 1], None), ([1, 0], [[1, 0, 0]]),
])
def test_bad_profile_is_rejected(target, negatives):
    with pytest.raises(ValueError):
        DynamicEMATracker(target, negatives=negatives)
