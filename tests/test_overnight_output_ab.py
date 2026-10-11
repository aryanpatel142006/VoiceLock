"""Offline correctness checks for overnight mixer and AGC counterfactuals evaluation.

Signals generated in tests are correctness fixtures, not model or quality claims.
"""
import json
import sys

import numpy as np
import pytest

from eval.overnight_output_ab import (
    apply_agc_policy,
    apply_mixer_policy,
    compute_metrics,
    evaluate_gate_criteria,
    match_loudness,
    validate_bank,
)


def test_validate_bank_checks_limit_and_duplicates():
    with pytest.raises(ValueError, match="empty"):
        validate_bank({"cases": []}, 12)
    with pytest.raises(ValueError, match="duplicate"):
        validate_bank({"cases": [{"id": "a"}, {"id": "a"}]}, 12)
    with pytest.raises(ValueError, match="300"):
        validate_bank({"cases": [{"id": str(i)} for i in range(301)]}, 12)
    with pytest.raises(ValueError, match="unsafe"):
        validate_bank({"cases": [{"id": "../bad"}]}, 12)


def test_apply_mixer_policy_respects_semantics_and_semantics_slider():
    mix = np.ones(1600, dtype=np.float32)
    selected = np.zeros(1600, dtype=np.float32)
    
    # At slider 1.0 (full suppression), room_gain(1.0) = 0.0
    # confidence 0.0 -> baseline floor_db = 6.0 dB -> floor_gain = 10^(-6/20) = ~0.501187
    out_base = apply_mixer_policy(selected, mix, confidence=0.0, policy='baseline', suppression=1.0)
    assert np.allclose(out_base, 10.0 ** (-6.0 / 20.0), atol=1e-4)

    # defined24db -> min attenuation 24 dB -> floor_gain <= 10^(-24/20) = ~0.0630957
    out_24db = apply_mixer_policy(selected, mix, confidence=0.0, policy='defined24db', suppression=1.0)
    assert np.allclose(out_24db, 10.0 ** (-24.0 / 20.0), atol=1e-4)

    # t180_k2 -> gated floor 20 dB -> floor_gain = 10^(-20/20) = 0.1
    out_k2 = apply_mixer_policy(selected, mix, confidence=0.0, policy='t180_k2', suppression=1.0)
    assert np.allclose(out_k2, 0.1, atol=1e-4)


def test_apply_agc_policy_processes_audio_without_nan():
    x = (0.1 * np.sin(np.arange(16000) * 0.05)).astype(np.float32)
    out_base = apply_agc_policy(x, policy='baseline')
    out_t179 = apply_agc_policy(x, policy='t179_agc')
    assert len(out_base) == len(x)
    assert len(out_t179) == len(x)
    assert np.isfinite(out_base).all()
    assert np.isfinite(out_t179).all()


def test_match_loudness_scales_rms():
    ref = np.ones(1600, dtype=np.float32) * 0.5
    tgt = np.ones(1600, dtype=np.float32) * 0.25
    matched = match_loudness(ref, tgt)
    assert np.allclose(matched, 0.5, atol=1e-3)


def test_compute_metrics_correctness():
    # Construct synthetic stems
    t = np.sin(np.arange(16000) * 0.1).astype(np.float32)
    i = np.cos(np.arange(16000) * 0.2).astype(np.float32)
    mix = t + i
    out = t.copy()
    pre = t.copy()
    metrics = compute_metrics(out, pre, mix, t, i)
    assert metrics['wrong_speaker_s'] == 0.0
    assert metrics['true_dropout_s'] == 0.0
    assert metrics['target_si_sdri_db'] > 0


def test_evaluate_gate_criteria():
    base = {
        'interferer_projection_db': -10.0,
        'wrong_speaker_s': 0.5,
        'true_dropout_s': 0.1,
    }
    # Candidate improving leak by 4 dB, same wrong speaker, 0.1s additional dropout -> win
    cand_win = {
        'interferer_projection_db': -14.0,
        'wrong_speaker_s': 0.5,
        'true_dropout_s': 0.2,
    }
    res = evaluate_gate_criteria(base, cand_win)
    assert res['passed'] is True
    assert res['leak_win'] is True
    assert res['wrong_win'] is True
    assert res['drop_win'] is True

    # Candidate improving leak by 1 dB only (< 3 dB) -> fail
    cand_fail_leak = {
        'interferer_projection_db': -11.0,
        'wrong_speaker_s': 0.5,
        'true_dropout_s': 0.1,
    }
    assert evaluate_gate_criteria(base, cand_fail_leak)['passed'] is False

    # Candidate with more wrong speaker -> fail
    cand_fail_wrong = {
        'interferer_projection_db': -15.0,
        'wrong_speaker_s': 0.8,
        'true_dropout_s': 0.1,
    }
    assert evaluate_gate_criteria(base, cand_fail_wrong)['passed'] is False

    # Candidate with > 0.2s extra dropouts -> fail
    cand_fail_drop = {
        'interferer_projection_db': -15.0,
        'wrong_speaker_s': 0.5,
        'true_dropout_s': 0.35,  # +0.25s extra
    }
    assert evaluate_gate_criteria(base, cand_fail_drop)['passed'] is False


def test_main_checkpoint_error_on_direct_failure(tmp_path, monkeypatch):
    from eval import overnight_output_ab as runner

    bank = tmp_path / 'bank.json'
    bank.write_text(json.dumps({'asr_model': 'small.en', 'cases': [{'id': 'err_case'}]}))
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    output = tmp_path / 'out'
    monkeypatch.setattr(runner, 'pick', lambda: 'cpu')

    def fail(*args):
        raise RuntimeError('simulated scoring error')

    monkeypatch.setattr(runner, 'score_case', fail)
    monkeypatch.setattr(
        sys,
        'argv',
        ['runner', '--bank', str(bank), '--model', str(model), '--output', str(output), '--case', 'err_case'],
    )
    assert runner.main() == 2
    res = json.loads((output / 'err_case' / 'result.json').read_text())
    assert res['status'] == 'error'
    assert 'simulated scoring error' in res['error']
