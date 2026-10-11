"""Offline regression tests; tones test DSP, not speaker recognition."""
import numpy as np
import pytest
import soundfile as sf

from eval.pitch_conditioned_head import acoustic_features, validate_manifest


def test_pitch_and_centroid_are_gain_invariant():
    t = np.arange(8000) / 16000
    x = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    a = acoustic_features(x)
    b = acoustic_features(x * 0.1)
    assert a[0] == pytest.approx(150, abs=3)
    assert a[1] == pytest.approx(150, abs=3)
    np.testing.assert_allclose(a, b, rtol=1e-4)


def test_silence_and_invalid_audio():
    np.testing.assert_array_equal(acoustic_features(np.zeros(8000)), [0, 0])
    for x in (np.zeros((20, 2)), np.array([np.nan]), np.zeros(10)):
        with pytest.raises(ValueError):
            acoustic_features(x)


def test_rejects_audio_copied_between_train_and_test(tmp_path):
    records = []
    for split in ('train', 'test'):
        for label, speaker in enumerate(('aryan', 'joseph')):
            path = tmp_path / f'{split}-{speaker}.wav'
            sf.write(path, np.full(8000, 0.1 + label * 0.1), 16000)
            records.append({'split': split, 'speaker': speaker, 'path': str(path)})
    with pytest.raises(ValueError, match='overlap'):
        validate_manifest(records)


@pytest.mark.slow
def test_head_learns_acoustic_signal_and_roundtrips(tmp_path):
    import torch

    from eval.pitch_conditioned_head import fit_head, project

    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(24, 192)).astype(np.float32)
    labels = np.repeat([0, 1], 12)
    features = np.column_stack((100 + labels * 100, 500 + labels * 500)).astype(np.float32)
    model, info = fit_head(embeddings, features, labels, epochs=80, device='cpu')
    z = project(model, embeddings, features)
    assert z.shape == (24, 32)
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1, atol=1e-5)
    centers = np.stack([z[labels == i].mean(axis=0) for i in (0, 1)])
    assert np.mean(np.argmax(z @ centers.T, axis=1) == labels) == 1
    assert info['final_loss'] < info['initial_loss']
    path = tmp_path / 'head.pt'
    torch.save(model.state_dict(), path)
    from eval.pitch_conditioned_head import PitchConditionedHead
    restored = PitchConditionedHead()
    restored.load_state_dict(torch.load(path, weights_only=True))
    np.testing.assert_allclose(project(restored, embeddings, features), z, atol=1e-6)


def test_accepts_file_disjoint_manifest_and_records_hashes(tmp_path):
    records = []
    for i, (split, speaker) in enumerate((('train', 'aryan'), ('train', 'joseph'),
                                         ('test', 'aryan'), ('test', 'joseph'))):
        path = tmp_path / f'{i}.wav'
        t = np.arange(8000) / 16000
        sf.write(path, 0.2 * np.sin(2 * np.pi * (100 + 20 * i) * t), 16000)
        records.append({'split': split, 'speaker': speaker, 'path': str(path)})
    checked = validate_manifest(records)
    assert len({r['sha256_16k'] for r in checked}) == 4
    assert all(r['audio'].shape == (8000,) for r in checked)


@pytest.mark.slow
def test_rejects_malformed_training_arrays():
    from eval.pitch_conditioned_head import fit_head

    with pytest.raises(ValueError, match='Nx192'):
        fit_head(np.ones((4, 8)), np.ones((4, 2)), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match='Nx192'):
        fit_head(np.ones((4, 192)), -np.ones((4, 2)), np.array([0, 0, 1, 1]))
