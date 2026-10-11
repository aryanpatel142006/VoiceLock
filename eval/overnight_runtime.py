"""Bounded, hardware-free FocusHQ runtime evaluation; never promotes defaults."""
import argparse
import hashlib
import json
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

AUDIO_KEYS = ('mix', 'target_enroll', 'other_enroll')


def build_plan(configs, *, repeats=3, clips=12, cases_used=0):
    if not 1 <= len(configs) <= 4:
        raise ValueError('cap: at most 4 configs')
    if not isinstance(repeats, int) or not 1 <= repeats <= 3:
        raise ValueError('cap: at most 3 repeats')
    if not any(c[0] == 2 and c[1] == 1 for c in configs):
        raise ValueError('must include the 2s/1s baseline')
    if clips != 12:
        raise ValueError('Replay12 requires exactly 12 clips')
    for w, h in configs:
        if not math.isfinite(w) or not math.isfinite(h) or not 0 < h <= w:
            raise ValueError('require finite 0 < hop <= window')
    total = (len(configs) * clips + 3) * repeats
    if not isinstance(cases_used, int) or cases_used < 0 or cases_used + total > 60:
        raise ValueError(f'global budget exceeded: {cases_used} + {total} > 60')
    plan = [{'window_s': float(w), 'hop_s': float(h), 'clip': c, 'repeat': r, 'arm': 'serial'}
            for w, h in configs for r in range(repeats) for c in range(clips)]
    for r in range(repeats):
        for arm in ('load_control', 'synthetic_load', 'target_switch'):
            plan.append({'window_s': 2., 'hop_s': 1., 'clip': 0, 'repeat': r, 'arm': arm})
    return plan


def execution_seconds(now, gpu_minutes):
    if not math.isfinite(gpu_minutes) or not 0 < gpu_minutes <= 90:
        raise ValueError('GPU cap must be finite, >0 and <=90 minutes')
    remaining = (datetime(2026, 10, 10, 14, tzinfo=UTC) - now).total_seconds()
    if remaining <= 0:
        raise ValueError('absolute stop 14:00 UTC Oct10 reached')
    return min(gpu_minutes * 60, remaining)


def summarize_windows(windows):
    times = [w['ms'] for w in windows
             if not w['cold'] and w['ms'] is not None and math.isfinite(w['ms'])]
    return {
        'warm_compute_p50_ms': float(np.percentile(times, 50)) if times else None,
        'warm_compute_p95_ms': float(np.percentile(times, 95)) if times else None,
        'warm_compute_max_ms': max(times) if times else None,
        'missed_hops': sum(w['reason'] in ('late', 'overrun') for w in windows),
        'windows': len(windows),
    }


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_audio(path):
    import soundfile as sf
    audio, sr = sf.read(path, dtype='float32')
    if sr != 16000 or audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'{path}: require nonempty finite mono 16kHz audio')
    if np.max(np.abs(audio)) > 1:
        raise ValueError(f'{path}: audio outside [-1, 1]')
    return audio


def validate_manifest(manifest, *, execute=False):
    """Validate every byte before starting a server, model or child process."""
    clips = manifest['clips']
    if len(clips) != 12 or len({c['id'] for c in clips}) != 12:
        raise ValueError('Replay12 IDs must be unique, exactly 12 clips')
    inputs = {}
    mixes = set()
    for clip in clips:
        for key in AUDIO_KEYS + tuple(k for k in ('target_reference', 'other_reference') if k in clip):
            path = clip[key]
            actual = sha(path)
            if clip.get('sha256', {}).get(key) != actual:
                raise ValueError(f'{clip["id"]} {key}: missing or mismatched hash')
            read_audio(path)
            inputs[path] = actual
            if key == 'mix':
                mixes.add(actual)
    if len(mixes) != 12:
        raise ValueError('Replay12 requires 12 distinct mix hashes')
    model_hashes = {}
    for path, expected in manifest.get('model_hashes', {}).items():
        actual = sha(path)
        if actual != expected:
            raise ValueError(f'model hash mismatch: {path}')
        model_hashes[path] = actual
    if execute:
        if 'cases_used' not in manifest or not model_hashes:
            raise ValueError('execute requires cases_used and verified model_hashes')
        from hearing import models
        from hearing.tse.blind import MODEL as SEPARATOR_MODEL
        from hearing.tse.blind import _checkpoint
        from hearing.tse.enrolled import FILES
        from hearing.tse.enrolled import MODEL as EMBEDDER_MODEL
        try:
            expected_paths = {_checkpoint().resolve(),
                              (models.cache_dir() / SEPARATOR_MODEL / 'last_best_checkpoint').resolve()}
        except RuntimeError as error:
            raise ValueError('loaded model paths unavailable') from error
        expected_paths.update((models.cache_dir() / EMBEDDER_MODEL / name).resolve() for name in FILES)
        if not expected_paths <= {Path(p).resolve() for p in model_hashes}:
            raise ValueError('hashes must cover every loaded model file under HEARING_MODELS_DIR')
        clip = clips[0]
        if not all(k in clip for k in ('target_reference', 'other_reference')):
            raise ValueError('switch arm requires aligned target_reference and other_reference on clip 0')
        length = len(read_audio(clip['mix']))
        if length < 16000 * 12 or any(len(read_audio(clip[k])) != length for k in
                                     ('target_reference', 'other_reference')):
            raise ValueError('switch references must align with mix of at least 12 seconds')
    return {'clips': clips, 'input_hashes': inputs, 'model_hashes': model_hashes}


def synthetic_gpu_load(stop, ready, deadline):
    """Separate process, fixed small matmul and 50ms pause; no unbounded job fanout."""
    import torch
    if not torch.cuda.is_available():
        return
    a = torch.ones((1024, 1024), device='cuda')
    while not stop.is_set() and time.monotonic() < deadline:
        torch.mm(a, a)
        torch.cuda.synchronize()
        ready.set()
        stop.wait(.05)


def switch_delay(output, old, new, switch_sample, delay):
    """Acoustic proxy: 3 consecutive 100ms blocks prefer new reference by cosine .1.

    Requires old-reference dominance in the final 300ms before the command.
    Reports end-of-confirmation time, including algorithmic output delay. This
    is reference similarity, not a semantic speaker-identification guarantee.
    """
    def preference(y, a, b):
        def corr(x):
            den = np.linalg.norm(y) * np.linalg.norm(x)
            return abs(float(np.dot(y, x) / den)) if den > 1e-8 else 0.
        return corr(b) - corr(a)
    n = min(len(old), len(new), len(output) - delay)
    if switch_sample < 4800 or n <= switch_sample:
        return None
    # Audible output before command corresponds to earlier source samples.
    pre = switch_sample - delay
    if pre < 4800:
        return None
    if any(preference(output[i + delay:i + delay + 1600], old[i:i + 1600],
                      new[i:i + 1600]) >= -.1 for i in range(pre - 4800, pre, 1600)):
        return None
    streak = 0
    for i in range(max(0, pre), n - 1599, 1600):
        score = preference(output[i + delay:i + delay + 1600], old[i:i + 1600], new[i:i + 1600])
        streak = streak + 1 if score > .1 else 0
        if streak == 3:
            return (i + 1600 + delay - switch_sample) / 16
    return None


def replay(mix, target, other, config, deadline, *, references=None, stage=None):
    """Real threaded wrapper, paced at 10ms; no hardware or quality scoring."""
    from hearing.tse.enroll import ENROLLER_KEY, AutoEnroller
    from hearing.tse.stage import FocusHQ
    from hearing.types import AudioChunk, StageContext

    stage = stage or FocusHQ(candidate='enrolled', window_s=config['window_s'],
                            hop_s=config['hop_s'], threaded=True, adapt_hop=False, require_gpu=True)
    seen, output_audio = {}, []
    raw_samples = frames = 0
    ctx = StageContext.default()
    enr = AutoEnroller()
    enr.enroll_clip(target, other)
    ctx.telemetry.extras[ENROLLER_KEY] = enr
    switch_sample = (len(mix) // 320) * 160
    switched = False
    load = stop = None
    try:
        stage.load()
        if stage.mock or stage.error or stage.device != 'cuda':
            raise RuntimeError(f'real CUDA required: {stage.health()}')
        if config['arm'] == 'synthetic_load':
            mp = multiprocessing.get_context('spawn')
            stop, ready = mp.Event(), mp.Event()
            load = mp.Process(target=synthetic_gpu_load, args=(stop, ready, deadline))
            load.start()
            if not ready.wait(min(15, max(0, deadline - time.monotonic()))):
                raise RuntimeError('synthetic GPU process did not become ready')
        start = time.monotonic()
        padded = np.pad(mix, (0, stage.delay + 160))
        for i in range(0, len(padded) - 159, 160):
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError('GPU wall cap reached')
            time.sleep(max(0, min(start + i / 16000 - now, deadline - now)))
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU wall cap reached')
            if load is not None and not load.is_alive():
                raise RuntimeError('synthetic GPU process exited during paired arm')
            if config['arm'] == 'target_switch' and i >= switch_sample and not switched:
                enr.enroll_clip(other, target)
                switched = True
            chunk = AudioChunk(padded[i:i + 160], 16000, i // 160, i / 16000)
            output_audio.append(stage.process(chunk, ctx).samples)
            for w in stage._recent:
                if w.status != 'pending':
                    seen[w.k] = {'k': w.k, 'cold': w.cold,
                                     'ms': w.ms if math.isfinite(w.ms) else None, 'reason': w.reason}
            if stage.delay <= i < stage.delay + len(mix):
                count = min(160, stage.delay + len(mix) - i)
                raw_samples += count * stage._mix_gain
                frames += count
        delay = None
        if switched and references is not None:
            delay = switch_delay(np.concatenate(output_audio), *references, switch_sample, stage.delay)
        result = summarize_windows(list(seen.values()))
        result.update(raw_fallback_weighted_seconds=raw_samples / 16000,
                      raw_fallback_note='approximation: end-of-10ms-frame private _mix_gain weighted exposure',
                      observed_output_seconds=frames / 16000, health=stage.health(),
                      window_records=list(seen.values()), target_switch_delay_ms=delay,
                      target_switch_status=('measured_acoustic_proxy' if delay is not None else
                                            'missing: no confirmed transition' if switched else 'not_run'),
                      target_switch_note='100ms aligned reference cosine margin .1, 3 consecutive blocks',
                      ssh_rtt_ms=None, ssh_rtt_status='missing', hardware_mic_to_ear_ms=None,
                      concurrent_load_status='synthetic_process' if load else 'not_run',
                      quality_status='not_scored', wer=None)
        return result
    finally:
        if load is not None:
            stop.set()
            load.join(timeout=1)
            if load.is_alive():
                load.terminate()
                load.join(timeout=1)
        stage.close()


def run_remote_smoke(url):
    """Read-only health probe of an existing service. Explicitly health-only."""
    import urllib.request
    start = time.perf_counter()
    with urllib.request.urlopen(url.rstrip('/') + '/health', timeout=5) as response:
        health = json.loads(response.read())
    elapsed = (time.perf_counter() - start) * 1000
    return {'status': 'health_only', 'health_round_trip_ms': elapsed,
                'served_revision': health.get('git_rev'), 'served_config_hashes': health.get('tiers_config'),
                'ssh_rtt_ms': None, 'ssh_rtt_status': 'missing: HTTP round trip is not isolated SSH RTT',
                'session_smoke_status': 'missing', 'hardware_mic_to_ear_ms': None}


def run_bounded(command, seconds):
    """Kill the entire evaluator process group, including synthetic load at the cap."""
    proc = subprocess.Popen(command, start_new_session=True)
    try:
        code = proc.wait(timeout=seconds)
        if code:
            raise subprocess.CalledProcessError(code, command)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-minutes', type=float, default=90)
    parser.add_argument('--health-url', help='existing service only; optional health-only probe')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text())
    validated = validate_manifest(manifest, execute=args.execute)
    plan = build_plan(manifest.get('configs', [[2, 1]]), repeats=manifest.get('repeats', 3),
                      clips=len(validated['clips']), cases_used=manifest.get('cases_used', 0))
    if args.execute and not args.worker:
        seconds = execution_seconds(datetime.now(UTC), args.gpu_minutes)
        if args.output.exists():
            raise FileExistsError(args.output)
        try:
            run_bounded([sys.executable, '-m', 'eval.overnight_runtime',
                         *(sys.argv[1:] if argv is None else argv), '--worker'], seconds)
        except subprocess.TimeoutExpired:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('a') as output:
                output.write(json.dumps({'type': 'stop', 'status': 'capped', 'reason': 'supervisor wall cap'}) + '\n')
            raise SystemExit(2) from None
        return
    source_paths = subprocess.check_output(
        ['git', 'ls-files', 'hearing', 'config'], text=True).splitlines()
    provenance = {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      'manifest_sha256': sha(args.manifest), 'evaluator_sha256': sha(__file__),
                      'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
                      'config_sha256': hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
                      'cases_used_before_run': manifest.get('cases_used', 0),
                      'source_hashes': {p: sha(p) for p in source_paths if Path(p).is_file()},
                      'input_hashes': validated['input_hashes'], 'model_hashes': validated['model_hashes'],
                      'served_revision': None, 'served_config_hashes': None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        def checkpoint(row):
            output.write(json.dumps(row) + '\n')
            output.flush()
            os.fsync(output.fileno())
        checkpoint({'type': 'plan', 'plan': plan, 'provenance': provenance})
        if not args.execute:
            return
        deadline = time.monotonic() + execution_seconds(datetime.now(UTC), args.gpu_minutes)
        if args.health_url:
            try:
                checkpoint(dict(type='smoke', **run_remote_smoke(args.health_url)))
            except (OSError, ValueError) as error:
                checkpoint({'type': 'smoke', 'status': 'missing', 'error': str(error)})
        for config in plan:
            clip = validated['clips'][config['clip']]
            row = {'type': 'result', 'config': config, 'clip_id': clip['id'], 'source_hashes': clip['sha256'],
                       'provenance': provenance}
            try:
                if time.monotonic() >= deadline:
                    raise TimeoutError('GPU wall cap reached')
                references = None
                if config['arm'] == 'target_switch':
                    references = tuple(read_audio(clip[k]) for k in ('target_reference', 'other_reference'))
                result = replay(*(read_audio(clip[k]) for k in AUDIO_KEYS), config, deadline,
                                references=references)
                row.update(result, status='measured')
            except (RuntimeError, ValueError, TimeoutError, OSError) as error:
                checkpoint(dict(row, status='blocked', error=f'{type(error).__name__}: {error}'))
                raise SystemExit(2) from error
            checkpoint(row)
        checkpoint({'type': 'stop', 'status': 'complete', 'cases': len(plan)})


if __name__ == '__main__':
    main()
