"""Bounded, offline counterfactual evaluation of mixer and AGC policies on fixed separator outputs.

Evaluates up to 4 configurations on frozen saved separator outputs without rerunning
upstream separation models:
  1. baseline: current frozen baseline mixer (confidence_floor_db standard) + baseline AGC (T-169)
  2. defined24db: 24 dB minimum attenuation respecting slider semantics + baseline AGC
  3. t180_k2: T-180 K2 candidate (HEARING_GATED_FLOOR_DB=20) + baseline AGC
  4. t179_agc: current frozen baseline mixer + opt-in T-179 noise-floor-aware AGC (FLOOR_RULE)

Criterion for candidate win over baseline:
  - Improves leak >= 3 dB (lower leak_db by >= 3 dB)
  - Without increased wrong-speaker rate (wrong_speaker_s <= baseline wrong_speaker_s)
  - With additional true dropouts <= 0.2s per matched scene (true_dropouts_s <= baseline + 0.2s)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.metrics import si_sdr
from eval.overnight_separation import sha256, word_errors
from hearing.audio.agc import FLOOR_RULE, Agc
from hearing.device import pick
from hearing.fusion.mixer import confidence_floor_db, room_gain

CONFIGURATIONS = ('baseline', 'defined24db', 't180_k2', 't179_agc')
SR = 16000
FRAME_LEN = 160  # 10 ms at 16 kHz
DROP_DB = -10.0  # target gain in output < -10 dB is a dropout


def validate_bank(bank: dict, limit: int = 12) -> list[dict]:
    cases = bank.get('cases', [])
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


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def apply_mixer_policy(
    selected: np.ndarray,
    mix: np.ndarray,
    confidence: float,
    policy: str,
    suppression: float = 1.0,
) -> np.ndarray:
    """Applies a mixer policy per chunk or whole clip respecting slider semantics.
    
    selected: model selected stream (target candidate)
    mix: original input mixture
    confidence: TSE pick confidence in [0, 1]
    policy: 'baseline', 'defined24db', 't180_k2', 't179_agc'
    suppression: slider s in [0, 1], default 1.0
    """
    s = min(max(float(suppression), 0.0), 1.0)
    slider_room_gain = room_gain(s)
    
    c = float(confidence)
    c = min(max(c, 0.0), 1.0) if math.isfinite(c) else 0.0
    
    if policy == 'defined24db':
        # Explicit 24 dB minimum attenuation floor respecting slider semantics:
        # floor cannot duck more than 24 dB (i.e. floor attenuation is min 24 dB -> floor gain max 10^(-24/20)),
        # but also cannot suppress more than slider room_gain.
        # Specifically, floor_db is at least 24.0 dB across confidences.
        std_floor_db = confidence_floor_db(c)
        floor_db = max(24.0, std_floor_db)
    elif policy == 't180_k2':
        # T-180 K2 candidate: gated windows (confidence exactly 0) have floor 20 dB
        old_gated = os.environ.get('HEARING_GATED_FLOOR_DB')
        try:
            os.environ['HEARING_GATED_FLOOR_DB'] = '20'
            floor_db = confidence_floor_db(c)
        finally:
            if old_gated is not None:
                os.environ['HEARING_GATED_FLOOR_DB'] = old_gated
            else:
                os.environ.pop('HEARING_GATED_FLOOR_DB', None)
    else:
        # 'baseline' or 't179_agc' uses standard confidence floor
        floor_db = confidence_floor_db(c)
    
    floor_gain = 10.0 ** (-floor_db / 20.0)
    keep = max(slider_room_gain, floor_gain)
    residual = mix - selected
    out = selected + keep * residual
    return out.astype(np.float32)


def apply_agc_policy(
    audio: np.ndarray,
    policy: str,
    sample_rate: int = SR,
) -> np.ndarray:
    """Applies the AGC policy (baseline T-169 or opt-in T-179 FLOOR_RULE)."""
    kw = dict(FLOOR_RULE) if policy == 't179_agc' else {}
    agc = Agc(sample_rate=sample_rate, **kw)
    out = np.concatenate([
        agc.process_array(audio[i : i + FRAME_LEN])
        for i in range(0, len(audio), FRAME_LEN)
    ])
    if len(out) < len(audio):
        rem = audio[len(out):]
        if len(rem) > 0:
            out = np.concatenate([out, agc.process_array(rem)])
    return out[:len(audio)].astype(np.float32)


def match_loudness(ref: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Scales target to match the RMS loudness of ref for human review clips."""
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    tgt_rms = float(np.sqrt(np.mean(target.astype(np.float64) ** 2)))
    if tgt_rms <= eps or ref_rms <= eps:
        return target.copy()
    gain = ref_rms / tgt_rms
    # soft clip or scale
    scaled = target * gain
    peak = np.max(np.abs(scaled))
    if peak > 1.0:
        scaled = scaled / peak
    return scaled.astype(np.float32)


def compute_metrics(
    out: np.ndarray,
    pre_agc: np.ndarray,
    mix: np.ndarray,
    target: np.ndarray,
    interferer: np.ndarray,
    sample_rate: int = SR,
) -> dict:
    arrays = [np.asarray(a, dtype=np.float64) for a in (out, pre_agc, mix, target, interferer)]
    if any(a.ndim != 1 or len(a) != len(arrays[0]) for a in arrays) or not len(arrays[0]):
        raise ValueError('audio must be nonempty, mono, exactly aligned')
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite audio')
    
    out_a, pre_a, mix_a, tgt_a, oth_a = arrays
    present = np.mean(tgt_a ** 2) > 1e-10
    
    # Joint least squares interferer projection gain
    basis = np.stack([tgt_a, oth_a], axis=1)
    if np.linalg.matrix_rank(basis) == 2:
        coeffs = np.linalg.lstsq(basis, out_a, rcond=None)[0]
        leak = float(20 * np.log10(max(abs(coeffs[1]), 1e-10)))
        coeffs_pre = np.linalg.lstsq(basis, pre_a, rcond=None)[0]
        leak_pre = float(20 * np.log10(max(abs(coeffs_pre[1]), 1e-10)))
    else:
        leak = None
        leak_pre = None
        
    # Wrong speaker attribution in 100 ms (1600 samples) windows
    hop = 1600
    valid = correct = wrong = 0
    for start in range(0, len(out_a), hop):
        end = min(start + hop, len(out_a))
        t, i, y = tgt_a[start:end], oth_a[start:end], out_a[start:end]
        if np.mean(t ** 2) <= 1e-8 or np.mean(y ** 2) <= 1e-8:
            continue
        if np.linalg.matrix_rank(np.stack([t, i], axis=1)) < 2:
            continue
        a_c, b_c = np.linalg.lstsq(np.stack([t, i], axis=1), y, rcond=None)[0]
        is_corr = np.linalg.norm(a_c * t) >= np.linalg.norm(b_c * i)
        valid += end - start
        correct += (end - start) * is_corr
        wrong += (end - start) * (not is_corr)
        
    wrong_speaker_s = wrong / sample_rate
    
    # True dropouts: 0.5s (8000 samples) hops where target is active (RMS > -30 dBFS approx) but gain in output < DROP_DB (-10 dB)
    drop_hop = 8000
    tgt_rms_full = np.sqrt(np.mean(tgt_a ** 2))
    tgt_thresh = max(1e-4, tgt_rms_full * 0.0316)  # -30 dB of target RMS
    dropouts_count = 0
    active_target_hops = 0
    for start in range(0, len(out_a), drop_hop):
        end = min(start + drop_hop, len(out_a))
        if end - start < drop_hop:
            continue
        t_seg = tgt_a[start:end]
        if np.sqrt(np.mean(t_seg ** 2)) > tgt_thresh:
            active_target_hops += 1
            # Gain of target in output
            if np.linalg.matrix_rank(np.stack([t_seg, oth_a[start:end]], axis=1)) == 2:
                g_t = np.linalg.lstsq(np.stack([t_seg, oth_a[start:end]], axis=1), out_a[start:end], rcond=None)[0][0]
                if 20 * np.log10(max(abs(g_t), 1e-5)) < DROP_DB:
                    dropouts_count += 1
    true_dropout_s = dropouts_count * (drop_hop / sample_rate)
    
    # Target absent energy / leakage
    absent = np.zeros(len(tgt_a), dtype=bool)
    for start in range(0, len(tgt_a), hop):
        end = min(start + hop, len(tgt_a))
        absent[start:end] = np.mean(tgt_a[start:end] ** 2) <= 1e-8
    absent_rms = float(np.sqrt(np.mean(out_a[absent] ** 2))) if absent.any() else None
    
    # Pre/Post AGC quality (SI-SDRi)
    si_sdri_post = (si_sdr(out_a, tgt_a) - si_sdr(mix_a, tgt_a)) if present else None
    si_sdri_pre = (si_sdr(pre_a, tgt_a) - si_sdr(mix_a, tgt_a)) if present else None
    
    # Pause lift (AGC energy change during target-absent windows)
    if absent.any() and np.mean(pre_a[absent] ** 2) > 1e-12:
        pre_abs_rms = np.sqrt(np.mean(pre_a[absent] ** 2))
        post_abs_rms = np.sqrt(np.mean(out_a[absent] ** 2))
        pause_lift_db = float(20 * np.log10(max(post_abs_rms / pre_abs_rms, 1e-5)))
    else:
        pause_lift_db = None

    return {
        'target_si_sdri_db': si_sdri_post,
        'target_si_sdri_pre_agc_db': si_sdri_pre,
        'interferer_projection_db': leak,
        'interferer_projection_pre_agc_db': leak_pre,
        'target_absent_output_rms': absent_rms,
        'wrong_speaker_s': wrong_speaker_s,
        'true_dropout_s': true_dropout_s,
        'active_target_s': active_target_hops * (drop_hop / sample_rate),
        'pause_lift_db': pause_lift_db,
    }


def score_case(case: dict, root: Path, model_dir: Path, output: Path) -> dict:
    from faster_whisper import WhisperModel

    paths = {
        name: (root / case[name]).resolve()
        for name in ('mix', 'target', 'interferer', 'selected')
    }
    selection = root / case['selection_json']
    evidence = json.loads(selection.read_text())
    if not evidence:
        raise ValueError('empty actual selection evidence')
        
    confidence = float(evidence.get('confidence', evidence.get('pick_confidence', 0.5)))
    
    audio = {}
    for name, path in paths.items():
        x, rate = sf.read(path, dtype='float32')
        if rate != SR or x.ndim != 1:
            raise ValueError(f'{name}: require mono 16 kHz')
        audio[name] = x
        
    if len({len(x) for x in audio.values()}) != 1:
        raise ValueError('unaligned saved artifacts')

    device = 'cuda' if pick() == 'cuda' else 'cpu'
    compute_type = 'float16' if device == 'cuda' else 'int8'
    asr = WhisperModel(str(model_dir), device=device, compute_type=compute_type, local_files_only=True)
    
    case_res = {
        'id': case['id'],
        'status': 'measured',
        'confidence': confidence,
        'configs': {},
    }
    
    ref_target = case.get('target_text')
    ref_interferer = case.get('interferer_text')
    
    baseline_audio = None
    
    for cfg in CONFIGURATIONS:
        cfg_out_dir = output / cfg
        cfg_out_dir.mkdir(parents=True, exist_ok=True)
        
        pre_agc = apply_mixer_policy(audio['selected'], audio['mix'], confidence, cfg)
        post_agc = apply_agc_policy(pre_agc, cfg)
        
        # Save waveforms
        sf.write(cfg_out_dir / 'pre_agc.wav', pre_agc, SR)
        sf.write(cfg_out_dir / 'out.wav', post_agc, SR)
        
        # Human review clip: matched loudness to baseline
        if cfg == 'baseline':
            baseline_audio = post_agc
            matched = post_agc
        else:
            matched = match_loudness(baseline_audio if baseline_audio is not None else post_agc, post_agc)
        sf.write(cfg_out_dir / 'out_matched.wav', matched, SR)
        
        # Transcribe
        segments, _ = asr.transcribe(post_agc, language='en', beam_size=5, temperature=0,
                                     vad_filter=False, condition_on_previous_text=False)
        hyp_text = ' '.join(s.text.strip() for s in segments)
        
        metrics = compute_metrics(post_agc, pre_agc, audio['mix'], audio['target'], audio['interferer'])
        
        target_we = word_errors(ref_target, hyp_text) if ref_target is not None else None
        interferer_we = word_errors(ref_interferer, hyp_text) if ref_interferer is not None else None
        
        # Target word retention
        if target_we and target_we.get('reference_words', 0) > 0:
            ref_words = target_we['reference_words']
            del_words = target_we.get('deletions', 0)
            target_retention_rate = max(0.0, (ref_words - del_words) / ref_words)
        else:
            target_retention_rate = None
            
        case_res['configs'][cfg] = {
            'transcript': hyp_text,
            'target_word_errors': target_we,
            'target_retention_rate': target_retention_rate,
            'interferer_word_errors': interferer_we,
            'metrics': metrics,
        }
        
    return case_res


def evaluate_gate_criteria(
    baseline_metrics: dict,
    candidate_metrics: dict,
) -> dict:
    """Evaluates the win criteria:
    - leak improvement >= 3 dB (candidate leak_db <= baseline leak_db - 3 dB)
    - wrong speaker rate not increased (candidate wrong_speaker_s <= baseline wrong_speaker_s)
    - additional true dropouts <= 0.2s (candidate true_dropout_s <= baseline true_dropout_s + 0.2)
    """
    b_leak = baseline_metrics.get('interferer_projection_db')
    c_leak = candidate_metrics.get('interferer_projection_db')
    leak_win = (b_leak is not None and c_leak is not None and c_leak <= b_leak - 3.0)
    leak_delta = (c_leak - b_leak) if (b_leak is not None and c_leak is not None) else None
    
    b_wrong = baseline_metrics.get('wrong_speaker_s', 0.0)
    c_wrong = candidate_metrics.get('wrong_speaker_s', 0.0)
    wrong_win = (c_wrong <= b_wrong + 1e-4)
    
    b_drop = baseline_metrics.get('true_dropout_s', 0.0)
    c_drop = candidate_metrics.get('true_dropout_s', 0.0)
    drop_win = (c_drop <= b_drop + 0.2 + 1e-4)
    
    passed = bool(leak_win and wrong_win and drop_win)
    return {
        'passed': passed,
        'leak_win': bool(leak_win),
        'leak_delta_db': leak_delta,
        'wrong_win': bool(wrong_win),
        'wrong_delta_s': c_wrong - b_wrong,
        'drop_win': bool(drop_win),
        'drop_delta_s': c_drop - b_drop,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='Frozen local faster-whisper small.en directory')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=12)
    p.add_argument('--seconds', type=float, default=6000)
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
        for name in ('mix', 'target', 'interferer', 'selected', 'selection_json'):
            path = args.bank.parent / case.get(name, '__missing__')
            artifact_hashes[f"{case['id']}/{name}"] = sha256(path) if path.is_file() else None

    provenance = {
        'artifact_hashes': artifact_hashes,
        'bank_sha256': sha256(args.bank),
        'runner_sha256': sha256(Path(__file__)),
        'asr_model': 'small.en',
        'model_hashes': model_hashes,
        'configurations': list(CONFIGURATIONS),
        'config': {
            'beam_size': 5,
            'temperature': 0,
            'vad_filter': False,
            'condition_on_previous_text': False,
            'device': pick(),
            'compute_type': 'float16' if pick() == 'cuda' else 'int8',
        },
    }

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
    case_results = []
    
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
            if old['status'] == 'measured':
                case_results.append(old)
            continue
            
        command = [
            sys.executable, '-m', 'eval.overnight_output_ab',
            '--bank', str(args.bank),
            '--model', str(args.model),
            '--output', str(args.output),
            '--limit', str(args.limit),
            '--case', case['id'],
        ]
        try:
            with (dest / 'process.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=log, timeout=remaining, check=False)
            if not checkpoint.exists():
                atomic_json(checkpoint, {
                    'id': case['id'],
                    'status': 'error',
                    'error': f'worker exit {completed.returncode}',
                    'provenance': provenance,
                })
        except subprocess.TimeoutExpired:
            atomic_json(checkpoint, {
                'id': case['id'],
                'status': 'error',
                'error': 'hard time cap',
                'provenance': provenance,
            })
        old = json.loads(checkpoint.read_text())
        statuses[case['id']] = old['status']
        if old['status'] == 'measured':
            case_results.append(old)

    # Aggregate criteria evaluation across measured cases
    candidate_evaluations = {cfg: {'pass_count': 0, 'total': len(case_results), 'cases': {}} for cfg in CONFIGURATIONS if cfg != 'baseline'}
    for cr in case_results:
        cid = cr['id']
        b_met = cr['configs']['baseline']['metrics']
        for cfg, cand_eval in candidate_evaluations.items():
            c_met = cr['configs'][cfg]['metrics']
            res = evaluate_gate_criteria(b_met, c_met)
            cand_eval['cases'][cid] = res
            if res['passed']:
                cand_eval['pass_count'] += 1

    summary = {
        'provenance': provenance,
        'statuses': statuses,
        'unmeasured': [c['id'] for c in bank['cases'] if statuses.get(c['id']) != 'measured'],
        'elapsed_s': time.monotonic() - start,
        'candidate_evaluations': candidate_evaluations,
    }
    atomic_json(args.output / 'summary.json', summary)
    return 0 if len(statuses) == len(cases) and all(s == 'measured' for s in statuses.values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
