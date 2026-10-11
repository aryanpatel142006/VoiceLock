"""Bounded, offline scoring of frozen separator/selector/delivered checkpoints.

This consumes real saved outputs; it never substitutes an oracle for the selector
or recreates delivered audio. Missing artifacts are errors, not mock results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.metrics import si_sdr
from hearing.device import pick


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def word_errors(reference: str, hypothesis: str) -> dict:
    """Levenshtein word counts; empty-reference WER is undefined."""
    r, h = (re.findall(r"[a-z0-9']+", text.lower()) for text in (reference, hypothesis))
    table = [[(0, 0, j) for j in range(len(h) + 1)]]
    for n, token in enumerate(r, 1):
        row = [(0, n, 0)]
        for m, predicted in enumerate(h, 1):
            s, d, i = table[n - 1][m - 1]
            if token == predicted:
                row.append((s, d, i))
            else:
                a, b, c = table[n - 1][m]
                x, y, z = row[m - 1]
                row.append(min(((s + 1, d, i), (a, b + 1, c), (x, y, z + 1)), key=sum))
        table.append(row)
    s, d, i = table[-1][-1]
    return {'substitutions': s, 'deletions': d, 'insertions': i, 'reference_words': len(r),
                'wer': (s + d + i) / len(r) if r else None}


def audio_scores(output, mix, target, interferer) -> dict:
    arrays = [np.asarray(a, dtype=np.float64) for a in (output, mix, target, interferer)]
    if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays) or not len(arrays[0]):
        raise ValueError('audio must be nonempty, mono, exactly aligned')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite audio')
    out, mix, target, other = arrays
    present = np.mean(target ** 2) > 1e-10
    # Joint least squares avoids attributing target-correlated energy to the interferer.
    basis = np.stack([target, other], axis=1)
    coefficients = np.linalg.lstsq(basis, out, rcond=None)[0]
    leak = float(20 * np.log10(max(abs(coefficients[1]), 1e-10))) if np.any(other) else None
    valid = correct = wrong = 0
    hop = 1600  # evaluation-only 100 ms identity attribution windows
    for start in range(0, len(out), hop):
        end = min(start + hop, len(out))
        t, i, y = target[start:end], other[start:end], out[start:end]
        if np.mean(t ** 2) <= 1e-8 or np.mean(y ** 2) <= 1e-8:
            continue  # silence is neither correct nor wrong identity
        if np.linalg.matrix_rank(np.stack([t, i], axis=1)) < 2:
            continue  # cannot distinguish collinear stems
        a, b = np.linalg.lstsq(np.stack([t, i], axis=1), y, rcond=None)[0]
        is_correct = np.linalg.norm(a * t) >= np.linalg.norm(b * i)
        valid += end - start
        correct += (end - start) * is_correct
        wrong += (end - start) * (not is_correct)
    absent = np.zeros(len(target), dtype=bool)
    for start in range(0, len(target), hop):
        end = min(start + hop, len(target))
        absent[start:end] = np.mean(target[start:end] ** 2) <= 1e-8
    return {'target_si_sdri_db': si_sdr(out, target) - si_sdr(mix, target) if present else None,
                'interferer_projection_db': leak,
                'target_absent_output_rms': float(np.sqrt(np.mean(out[absent] ** 2))) if absent.any() else None,
                'identity_accuracy': correct / valid if valid else None,
                'identity_scored_s': valid / 16000, 'wrong_speaker_s': wrong / 16000}


def validate_bank(bank: dict, limit: int) -> list[dict]:
    cases = bank['cases']
    if not cases:
        raise ValueError('empty bank cannot be measured')
    if not 1 <= limit <= 300 or len(cases) > 300:
        raise ValueError('global bank and limit must be <=300 cases')
    ids = [case['id'] for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate case ids')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', ident) for ident in ids):
        raise ValueError('unsafe case id')
    return cases[:limit]


def atomic_json(path: Path, value: dict):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def score_case(case: dict, root: Path, model: Path, output: Path) -> dict:
    from faster_whisper import WhisperModel

    paths = {name: (root / case[name]).resolve() for name in
             ('mix', 'target', 'interferer', 'stream0', 'stream1', 'selected', 'delivered')}
    # Require selector evidence rather than an invented offline selection.
    selection = root / case['selection_json']
    evidence = json.loads(selection.read_text())
    if not evidence:
        raise ValueError('empty actual selection evidence')
    hashes = {name: sha256(path) for name, path in paths.items()}
    audio = {}
    for name, path in paths.items():
        x, rate = sf.read(path, dtype='float32')
        if rate != 16000 or x.ndim != 1:
            raise ValueError(f'{name}: require mono 16 kHz, no implicit resampling')
        audio[name] = x
    if len({len(x) for x in audio.values()}) != 1:
        raise ValueError('unaligned saved artifacts')
    target = audio['target']
    oracle = max(('stream0', 'stream1'), key=lambda k: si_sdr(audio[k], target)) if np.any(target) else None
    device = 'cuda' if pick() == 'cuda' else 'cpu'  # faster-whisper does not support MPS
    compute_type = 'float16' if device == 'cuda' else 'int8'
    asr = WhisperModel(str(model), device=device, compute_type=compute_type, local_files_only=True)
    result = {'id': case['id'], 'status': 'measured', 'source_hashes': hashes,
              'selection_sha256': sha256(selection), 'selection': evidence,
              'oracle_stream': oracle, 'asr_device': device, 'asr_compute_type': compute_type, 'scores': {}, 'transcripts': {},
              'reference_status': case.get('reference_status', 'automatic_unverified')}
    for name in ('target', 'interferer', 'mix', 'selected', 'delivered', 'stream0', 'stream1'):
        segments, _ = asr.transcribe(audio[name], language='en', beam_size=5, temperature=0,
                                    vad_filter=False, condition_on_previous_text=False)
        text = ' '.join(s.text.strip() for s in segments)
        result['transcripts'][name] = text
        shutil.copyfile(paths[name], output / f'{name}.wav')
    for label, name in [('raw_mix', 'mix'), ('oracle', oracle), ('selected', 'selected'), ('delivered', 'delivered')]:
        if name is None:
            result['scores'][label] = {'status': 'not_scored_zero_target'}
            continue
        text = result['transcripts'][name]
        metrics = audio_scores(audio[name], audio['mix'], target, audio['interferer'])
        for role in ('target', 'interferer'):
            reference = case.get(f'{role}_text')
            metrics[f'{role}_word_errors'] = word_errors(reference, text) if reference is not None else None
        metrics['target_absent_false_caption_words'] = len(text.split()) if not np.any(target) else None
        result['scores'][label] = metrics
    result['clean_stem_asr_ceiling'] = {
        role: word_errors(case[f'{role}_text'], result['transcripts'][role])
        if f'{role}_text' in case else None for role in ('target', 'interferer')}
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='Frozen local faster-whisper small.en directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--seconds', type=float, default=14400)
    p.add_argument('--case', help=argparse.SUPPRESS)
    args = p.parse_args()
    bank = json.loads(args.bank.read_text())
    cases = validate_bank(bank, args.limit)
    if not 0 < args.seconds <= 14400:
        p.error('seconds must be in (0, 14400]')
    if bank.get('asr_model') != 'small.en' or not args.model.is_dir():
        p.error('require frozen real local small.en model and bank declaration')
    model_hashes = {str(f.relative_to(args.model)): sha256(f) for f in sorted(args.model.rglob('*')) if f.is_file()}
    if not model_hashes:
        p.error('empty model directory')
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_hashes = {}
    for case in cases:
        for name in ('mix', 'target', 'interferer', 'stream0', 'stream1', 'selected', 'delivered', 'selection_json'):
            path = args.bank.parent / case.get(name, '__missing__')
            artifact_hashes[f"{case['id']}/{name}"] = sha256(path) if path.is_file() else None
    provenance = {'artifact_hashes': artifact_hashes, 'bank_sha256': sha256(args.bank), 'runner_sha256': sha256(Path(__file__)),
                  'asr_model': 'small.en', 'model_hashes': model_hashes,
                  'config': {'beam_size': 5, 'temperature': 0, 'vad_filter': False,
                             'condition_on_previous_text': False, 'device': pick(),
                             'compute_type': 'float16' if pick() == 'cuda' else 'int8'}}
    if args.case:
        case = next(c for c in cases if c['id'] == args.case)
        dest = args.output / case['id']
        dest.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        try:
            result = score_case(case, args.bank.parent, args.model, dest)
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            result = {'id': case['id'], 'status': 'error', 'error': f'{type(exc).__name__}: {exc}'}
        result.update(provenance=provenance, elapsed_s=time.monotonic() - start)
        atomic_json(dest / 'result.json', result)
        return 0 if result['status'] == 'measured' else 2
    start = time.monotonic()
    statuses = {}
    for case in cases:
        remaining = args.seconds - (time.monotonic() - start)
        if remaining <= 0:
            break
        dest = args.output / case['id']
        dest.mkdir(exist_ok=True)
        checkpoint = dest / 'result.json'
        if checkpoint.exists():
            old = json.loads(checkpoint.read_text())
            if old.get('provenance') != provenance:
                raise ValueError('checkpoint provenance mismatch; use a new output directory')
            statuses[case['id']] = old['status']
            continue  # errors are checkpointed too; no retry loop
        command = [sys.executable, '-m', 'eval.overnight_separation', '--bank', str(args.bank),
                   '--model', str(args.model), '--output', str(args.output), '--limit', str(args.limit),
                   '--case', case['id']]
        try:
            with (dest / 'process.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=log, timeout=remaining, check=False)
            if not checkpoint.exists():
                atomic_json(checkpoint, {'id': case['id'], 'status': 'error',
                            'error': f'worker exit {completed.returncode}', 'provenance': provenance})
        except subprocess.TimeoutExpired:
            atomic_json(checkpoint, {'id': case['id'], 'status': 'error', 'error': 'hard time cap',
                                    'provenance': provenance})
        statuses[case['id']] = json.loads(checkpoint.read_text())['status']
    atomic_json(args.output / 'summary.json', {'provenance': provenance, 'statuses': statuses,
                'unmeasured': [c['id'] for c in bank['cases'] if statuses.get(c['id']) != 'measured'],
                'elapsed_s': time.monotonic() - start})
    return 0 if len(statuses) == len(cases) and all(s == 'measured' for s in statuses.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
