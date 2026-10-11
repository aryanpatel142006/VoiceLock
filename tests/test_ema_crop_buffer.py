import numpy as np
import pytest

from eval.ema_crop_buffer import EMACropBuffer, evaluate_stream


def test_time_weighted_ema_and_copy():
    b = EMACropBuffer(half_life_s=0.5, max_gap_s=2)
    a = b.update('a', [1, 0], elapsed_s=0.5)
    a[:] = 0
    out = b.update('a', [0, 1], elapsed_s=0.5)
    np.testing.assert_allclose(out, [2**-0.5, 2**-0.5])
    np.testing.assert_allclose(b.update('b', [0, 1], elapsed_s=0.5), [0, 1])


def test_gap_reset_and_explicit_reset():
    b = EMACropBuffer()
    b.update(0, [1, 0], elapsed_s=0.5)
    np.testing.assert_allclose(b.update(0, [0, 1], elapsed_s=3), [0, 1])
    b.reset()
    np.testing.assert_allclose(b.update(0, [1, 0], elapsed_s=0.5), [1, 0])


@pytest.mark.parametrize('vector', [[0, 0], [np.nan, 1], [[1, 0]], []])
def test_invalid_embedding(vector):
    with pytest.raises(ValueError):
        EMACropBuffer().update(0, vector, elapsed_s=0.5)


@pytest.mark.parametrize('seconds', [0, -1, np.inf, np.nan])
def test_invalid_time(seconds):
    with pytest.raises(ValueError):
        EMACropBuffer().update(0, [1, 0], elapsed_s=seconds)


def test_dimension_change_rejected():
    b = EMACropBuffer()
    b.update(0, [1, 0], elapsed_s=0.5)
    with pytest.raises(ValueError):
        b.update(0, [1, 0, 0], elapsed_s=0.5)


def test_antipodal_update_stays_finite():
    b = EMACropBuffer(half_life_s=0.5)
    b.update(0, [1, 0], elapsed_s=0.5)
    np.testing.assert_allclose(b.update(0, [-1, 0], elapsed_s=0.5), [-1, 0])


def test_replay_counts_errors_and_flips_including_switch():
    x = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    result = evaluate_stream(x, np.eye(2), [0, 0, 0, 1], np.ones(4)*0.5,
                             half_life_s=0.01)
    assert result['raw']['accuracy'] == 0.75
    assert result['raw']['spurious_flips'] == 2
    assert result['ema']['accuracy'] == 0.75


def test_capacity_and_invalid_config():
    b = EMACropBuffer(max_streams=1)
    b.update(0, [1, 0], elapsed_s=0.5)
    with pytest.raises(ValueError):
        b.update(1, [0, 1], elapsed_s=0.5)
    for kwargs in ({'half_life_s': 0}, {'max_gap_s': np.nan}, {'max_streams': 0}):
        with pytest.raises(ValueError):
            EMACropBuffer(**kwargs)


def test_replay_invalid_and_no_stable_pairs():
    with pytest.raises(ValueError):
        evaluate_stream([[1, 0]], np.eye(2), [2], [0.5])
    result = evaluate_stream(np.eye(2), np.eye(2), [0, 1], [0.5, 0.5])
    assert result['ema']['mean_stable_cosine_step'] is None


def test_noisy_audio_replay_with_offline_embedder(tmp_path):
    import soundfile as sf

    from eval.ema_crop_buffer import audio_replay

    paths = []
    for hz in (200, 400):
        path = tmp_path / f'{hz}.wav'
        sf.write(path, np.sin(2*np.pi*hz*np.arange(56000)/16000)*0.1, 16000)
        paths.append(path)

    class Embedder:
        def embed(self, x):
            return np.array([np.mean(x*x), np.mean(np.abs(x))])

    x, p, y, dt = audio_replay(paths, Embedder(), 0.5)
    assert x.shape == (6, 2)
    assert p.shape == (2, 2)
    assert y.tolist() == [0, 0, 1, 1, 0, 0]
    np.testing.assert_allclose(dt, 0.5)
