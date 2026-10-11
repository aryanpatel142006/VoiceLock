"""Offline correctness checks; generated signals are not quality evidence."""
import numpy as np
import pytest

from eval.overnight_separation import audio_scores, validate_bank, word_errors


def test_word_errors_preserve_insertions_deletions_and_empty_reference():
    assert word_errors('one two three', 'one four') == {
        'substitutions': 1, 'deletions': 1, 'insertions': 0, 'reference_words': 3,
        'wer': 2 / 3,
    }
    assert word_errors('', 'false caption')['insertions'] == 2
    assert word_errors('', 'false caption')['wer'] is None


def test_zero_target_is_not_si_sdr_scored_and_leak_is_measured():
    other = np.sin(np.arange(1600) * 0.1).astype('float32')
    scores = audio_scores(other, other, np.zeros_like(other), other)
    assert scores['target_si_sdri_db'] is None
    assert scores['target_absent_output_rms'] > 0.6
    assert scores['interferer_projection_db'] == pytest.approx(0, abs=1e-6)
    assert scores['identity_accuracy'] is None


def test_identity_masks_silence_and_wrong_speaker_duration():
    t = np.tile([1., -1., 0., 0.], 400)
    i = np.tile([0., 0., 1., -1.], 400)
    scores = audio_scores(t, t + i, t, i)
    assert scores['identity_accuracy'] == 1
    assert scores['wrong_speaker_s'] == 0


def test_bank_rejects_duplicate_cases_and_global_overflow():
    with pytest.raises(ValueError, match='duplicate'):
        validate_bank({'cases': [{'id': 'a'}, {'id': 'a'}]}, 12)
    with pytest.raises(ValueError, match='300'):
        validate_bank({'cases': [{'id': str(x)} for x in range(301)]}, 12)


def test_output_requires_exact_alignment():
    with pytest.raises(ValueError, match='aligned'):
        audio_scores(np.ones(3), np.ones(4), np.ones(4), np.ones(4))


def test_nonfinite_output_is_rejected():
    with pytest.raises(ValueError, match='nonfinite'):
        audio_scores(np.array([float('nan')]), np.ones(1), np.ones(1), np.ones(1))


def test_unsafe_case_id_is_rejected():
    with pytest.raises(ValueError, match='unsafe'):
        validate_bank({'cases': [{'id': '../escape'}]}, 12)


def test_zero_crossings_are_not_target_absence():
    signal = np.tile([1., 0., -1., 0.], 400)
    assert audio_scores(signal, signal, signal, np.zeros_like(signal))['target_absent_output_rms'] is None


def test_direct_case_failure_saves_checkpoint_without_precreated_directory(tmp_path, monkeypatch):
    import json
    import sys

    from eval import overnight_separation as runner

    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'asr_model': 'small.en', 'cases': [{'id': 'missing'}]}))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    output = tmp_path / 'outputs'
    monkeypatch.setattr(runner, 'pick', lambda: 'cpu')

    def fail(*args):
        raise RuntimeError('reproducible model failure')

    monkeypatch.setattr(runner, 'score_case', fail)
    monkeypatch.setattr(sys, 'argv', ['runner', '--bank', str(bank), '--model', str(model),
                                    '--output', str(output), '--case', 'missing'])
    assert runner.main() == 2
    result = json.loads((output / 'missing' / 'result.json').read_text())
    assert result['status'] == 'error'
    assert 'reproducible model failure' in result['error']


def test_empty_bank_cannot_claim_measured_success():
    with pytest.raises(ValueError, match='empty'):
        validate_bank({'cases': []}, 12)
