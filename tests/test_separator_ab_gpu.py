"""Offline bank audit checks; generated signals are not quality evidence."""
import json

import numpy as np
import pytest
import soundfile as sf

from eval.separator_ab_gpu import audit_bank, sha256


def bank(tmp_path, silent=False):
    t = np.zeros(1600) if silent else np.sin(np.arange(1600) * 0.1) * 0.1
    row = {'id': 'case'}
    for key in ('mix', 'target', 'interferer', 'stream0', 'stream1', 'selected', 'delivered'):
        row[key] = f'{key}.wav'
        sf.write(tmp_path / row[key], t, 16000, subtype='FLOAT')
    row['selection_json'] = 'selection.json'
    (tmp_path / 'selection.json').write_text(json.dumps({'windows': [
        {'pick': 0, 'scores': [1, 0], 'start_s': 0, 'compute_ms': 10,
         'emit_start_sample': 0, 'emit_end_sample': 1600}]}))
    data = {'cases': [row], 'sha256': {name: sha256(tmp_path / name)
            for key, name in row.items() if key != 'id'}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    return path


def test_audit_silence_is_not_si_sdr(tmp_path):
    result = audit_bank(bank(tmp_path, silent=True))
    assert result['cases'][0]['best_saved_stream_si_sdr_db'] is None
    assert result['cases'][0]['selected_si_sdri_db'] is None
    assert result['cases'][0]['delivered_absent_far'] == 0


def test_tamper_rejected(tmp_path):
    path = bank(tmp_path)
    (tmp_path / 'selected.wav').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit_bank(path)


def test_missing_hash_rejected(tmp_path):
    path = bank(tmp_path)
    data = json.loads(path.read_text())
    del data['sha256']['target.wav']
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='missing hash'):
        audit_bank(path)


def test_unaligned_rejected(tmp_path):
    path = bank(tmp_path)
    sf.write(tmp_path / 'selected.wav', np.zeros(10), 16000)
    data = json.loads(path.read_text())
    data['sha256']['selected.wav'] = sha256(tmp_path / 'selected.wav')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='aligned'):
        audit_bank(path)
