"""Read-only audit of T-289's historical GPU output bank; never launches GPU work.

The frozen exporter remains with its source on the cluster. This auditor checks
saved artifacts and computes a whole-clip best-saved-stream diagnostic, distinct
from the historical window-oracle splice. It never recreates delivered audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.metrics import si_sdr
from eval.overnight_separation import sha256, validate_bank

AUDIO = ('mix', 'target', 'interferer', 'stream0', 'stream1', 'selected', 'delivered')


def audit_bank(path: Path) -> dict:
    bank = json.loads(path.read_text())
    cases = validate_bank(bank, 300)
    root = path.parent.resolve()
    hashes = bank['sha256']
    for relative, expected in hashes.items():
        artifact = (root / relative).resolve()
        if not artifact.is_relative_to(root):
            raise ValueError('artifact escapes bank')
        if sha256(artifact) != expected:
            raise ValueError(f'hash mismatch: {relative}')
    rows = []
    for case in cases:
        for key in (*AUDIO, 'selection_json'):
            if case[key] not in hashes:
                raise ValueError(f'missing hash: {case[key]}')
        audio = {}
        for key in AUDIO:
            x, rate = sf.read(root / case[key], dtype='float32')
            if rate != 16000 or x.ndim != 1 or not len(x) or not np.isfinite(x).all():
                raise ValueError('require finite nonempty mono 16 kHz audio')
            audio[key] = x
        if len({len(x) for x in audio.values()}) != 1:
            raise ValueError('audio must be exactly aligned')
        evidence = json.loads((root / case['selection_json']).read_text())
        windows = evidence['windows']
        cursor = 0
        for window in windows:
            if (window['pick'] not in (0, 1) or len(window['scores']) != 2
                    or window['emit_start_sample'] != cursor
                    or window['emit_end_sample'] <= cursor
                    or window['start_s'] < 0
                    or not np.isfinite([*window['scores'], window['compute_ms']]).all()
                    or window['compute_ms'] < 0):
                raise ValueError('invalid selection evidence or emission coverage')
            cursor = window['emit_end_sample']
        if not windows or cursor != len(audio['mix']):
            raise ValueError('selection evidence does not cover audio')
        target = audio['target']
        present = np.mean(target.astype(float) ** 2) > 1e-10
        baseline = si_sdr(audio['mix'], target) if present else None
        row = {'id': case['id'], 'input_si_sdr_db': baseline,
               'best_saved_stream_si_sdr_db': max(si_sdr(audio[k], target)
                  for k in ('stream0', 'stream1')) if present else None,
               'compute_ms_p50': float(np.percentile([w['compute_ms'] for w in windows], 50)),
               'compute_ms_p95': float(np.percentile([w['compute_ms'] for w in windows], 95))}
        for key in ('selected', 'delivered'):
            score = si_sdr(audio[key], target) if present else None
            row[f'{key}_si_sdr_db'] = score
            row[f'{key}_si_sdri_db'] = score - baseline if present else None
        # Evaluation-only acoustic false activation: 100 ms blocks, RMS >1e-4.
        # This is neither ASR false captions nor a calibrated selector FAR.
        absent = false = 0
        for start in range(0, len(target), 1600):
            sl = slice(start, start + 1600)
            size = len(target[sl])
            if np.mean(target[sl].astype(float) ** 2) <= 1e-8:
                absent += size
                if np.mean(audio['delivered'][sl].astype(float) ** 2) > 1e-8:
                    false += size
        row.update(target_absent_s=absent / 16000, delivered_absent_active_s=false / 16000,
                   delivered_absent_far=false / absent if absent else None,
                   wer=None, false_caption_words=None)
        rows.append(row)
    return {'manifest_sha256': sha256(path), 'verified_artifacts': len(hashes),
            'cases': rows, 'provenance': bank.get('provenance', {}),
            'limits': 'Offline saved audio; no ASR, live latency, or historical loaded-source proof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_bank(args.bank)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
