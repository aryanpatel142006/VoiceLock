"""Offline tests for caps and complete-window accounting (no model weights)."""
import pytest

from eval.overnight_runtime import build_plan, summarize_windows


def test_plan_caps_prevent_extra_gpu_cases():
    assert len(build_plan([(2, 1), (1, .5)], repeats=2, clips=12)) == 54
    with pytest.raises(ValueError):
        build_plan([(2, 1)] * 5, repeats=3, clips=12)
    with pytest.raises(ValueError):
        build_plan([(2, 1)], repeats=4, clips=12)
    with pytest.raises(ValueError):
        build_plan([(1, 2)], repeats=3, clips=12)


def test_warm_stats_exclude_cold_and_count_missed_windows_once():
    result = summarize_windows([
        {'k': 0, 'cold': True, 'ms': 9000, 'reason': ''},
        {'k': 1, 'cold': False, 'ms': 10, 'reason': ''},
        {'k': 2, 'cold': False, 'ms': 30, 'reason': 'late'},
        {'k': 3, 'cold': False, 'ms': None, 'reason': 'overrun'},
    ])
    assert result['warm_compute_p50_ms'] == 20
    assert result['warm_compute_p95_ms'] == 29
    assert result['warm_compute_max_ms'] == 30
    assert result['missed_hops'] == 2
    assert summarize_windows([])['warm_compute_max_ms'] is None


def test_process_cap_kills_a_stalled_model_load():
    import subprocess
    import sys

    from eval.overnight_runtime import run_bounded

    with pytest.raises(subprocess.TimeoutExpired):
        run_bounded([sys.executable, '-c', 'import time; time.sleep(10)'], .05)


def test_plan_requires_baseline_and_remaining_global_budget():
    with pytest.raises(ValueError, match='baseline'):
        build_plan([(1, .5)])
    with pytest.raises(ValueError, match='global'):
        build_plan([(2, 1), (1, .5)], cases_used=16)
    assert len(build_plan([(2, 1)], cases_used=15)) == 45
    with pytest.raises(ValueError, match='global'):
        build_plan([(2, 1)], cases_used=-1)


def test_execution_budget_never_crosses_absolute_stop():
    from datetime import UTC, datetime

    from eval.overnight_runtime import execution_seconds

    assert execution_seconds(datetime(2026, 10, 10, 13, 59, tzinfo=UTC), 90) == 60
    assert execution_seconds(datetime(2026, 10, 10, 3, tzinfo=UTC), 1) == 60
    with pytest.raises(ValueError, match='stop'):
        execution_seconds(datetime(2026, 10, 10, 14, tzinfo=UTC), 90)
    with pytest.raises(ValueError, match='cap'):
        execution_seconds(datetime(2026, 10, 10, 3, tzinfo=UTC), float('nan'))


def test_default_plan_has_three_repeats_and_paired_arms_under_60():
    plan = build_plan([(2, 1)])
    assert len(plan) == 45
    assert {c['repeat'] for c in plan} == {0, 1, 2}
    assert sum(c['arm'] == 'synthetic_load' for c in plan) == 3
    assert sum(c['arm'] == 'load_control' for c in plan) == 3
    assert sum(c['arm'] == 'target_switch' for c in plan) == 3
    with pytest.raises(ValueError, match='60'):
        build_plan([(2, 1), (1, .5)])


@pytest.fixture
def manifest(tmp_path):
    import json

    import numpy as np
    import soundfile as sf

    from eval.overnight_runtime import sha
    clips = []
    for i in range(12):
        path = tmp_path / f'{i}.wav'
        sf.write(path, np.full(1600, i / 100, np.float32), 16000)
        clips.append({'id': str(i), 'mix': str(path), 'target_enroll': str(path),
                          'other_enroll': str(path), 'sha256': {k: sha(path) for k in
                          ('mix', 'target_enroll', 'other_enroll')}})
    data = {'clips': clips, 'configs': [[2, 1]], 'repeats': 3, 'cases_used': 0}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    return path, data


def test_manifest_rejects_missing_changed_and_duplicate_inputs(manifest):
    from pathlib import Path

    from eval.overnight_runtime import validate_manifest
    _, data = manifest
    assert len(validate_manifest(data)['clips']) == 12
    path = Path(data['clips'][0]['mix'])
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        validate_manifest(data)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        validate_manifest(data)


def test_dry_run_validates_and_checkpoints_without_gpu_or_smoke(manifest, tmp_path, monkeypatch):
    import json

    from eval import overnight_runtime as runtime
    path, _ = manifest
    output = tmp_path / 'plan.jsonl'
    monkeypatch.setattr(runtime, 'run_remote_smoke', lambda: pytest.fail('dry run smoke'))
    runtime.main(['--manifest', str(path), '--output', str(output)])
    row = json.loads(output.read_text())
    assert row['type'] == 'plan'
    assert len(row['plan']) == 45
    assert row['provenance']['input_hashes']
    with pytest.raises(FileExistsError):
        runtime.main(['--manifest', str(path), '--output', str(output)])


def test_real_wrapper_instrumentation_and_non_cuda_replay_rejected():
    import numpy as np

    from eval.overnight_runtime import replay
    from hearing.tse.stage import FocusHQ
    stage = FocusHQ(candidate='identity', threaded=False)
    assert hasattr(stage, '_recent') and hasattr(stage, '_mix_gain')
    with pytest.raises(RuntimeError, match='CUDA'):
        replay(np.zeros(160), np.zeros(160), np.zeros(160),
               {'window_s': 2, 'hop_s': 1, 'arm': 'serial'}, 0, stage=stage)


def test_switch_detector_requires_sustained_new_voice_and_prior_old_voice():
    import numpy as np

    from eval.overnight_runtime import switch_delay
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=(2, 32000))
    out = a.copy()
    out[20000:] = b[20000:]
    assert switch_delay(out, a, b, 16000, 0) == pytest.approx(600)
    assert switch_delay(a, a, b, 16000, 0) is None
    assert switch_delay(b, a, b, 16000, 0) is None


def test_health_smoke_reports_real_http_timing_without_inventing_ssh():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from eval.overnight_runtime import run_remote_smoke
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"git_rev":"revision", "tiers_config":{"test":"hash"}}')
        def log_message(self, *args):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        row = run_remote_smoke(f'http://127.0.0.1:{server.server_port}')
        assert row['status'] == 'health_only'
        assert row['health_round_trip_ms'] > 0
        assert row['ssh_rtt_ms'] is None
        assert row['session_smoke_status'] == 'missing'
        assert row['served_revision'] == 'revision'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_replay_records_actual_wrapper_output_and_approximation(monkeypatch):
    import numpy as np

    from eval.overnight_runtime import replay
    from hearing.tse.stage import FocusHQ
    class Candidate:
        # Offline boundary double only. No weights/GPU and no quality evidence.
        name = 'test-runtime'
        device = 'cuda'
        needs = frozenset()
        def load(self):
            pass
        def extract(self, mix, sample_rate, **kwargs):
            return mix
    stage = FocusHQ(_candidate=Candidate(), window_s=.02, hop_s=.01, threaded=False)
    row = replay(np.full(160, .1, np.float32), np.ones(160), np.ones(160),
                 {'window_s': .02, 'hop_s': .01, 'arm': 'serial'}, float('inf'), stage=stage)
    assert row['quality_status'] == 'not_scored'
    assert row['observed_output_seconds'] == .01
    assert 'approximation' in row['raw_fallback_note']
    assert row['windows'] > 0
    assert row['ssh_rtt_ms'] is None


def test_manifest_rejects_duplicate_mix_even_with_distinct_ids(manifest):
    from eval.overnight_runtime import validate_manifest
    _, data = manifest
    data['clips'][1] = {**data['clips'][0], 'id': '1'}
    with pytest.raises(ValueError, match='distinct mix'):
        validate_manifest(data)


def test_execute_validation_blocks_before_output_or_smoke(manifest, tmp_path, monkeypatch):
    from eval import overnight_runtime as runtime
    path, _ = manifest
    output = tmp_path / 'not-created.jsonl'
    monkeypatch.setattr(runtime, 'run_remote_smoke', lambda *a: pytest.fail('premature smoke'))
    with pytest.raises(ValueError, match='model_hashes'):
        runtime.main(['--manifest', str(path), '--output', str(output), '--execute'])
    assert not output.exists()


def test_execute_rejects_unrelated_model_hashes(manifest, tmp_path, monkeypatch):
    from eval.overnight_runtime import sha, validate_manifest
    _, data = manifest
    unrelated = tmp_path / 'unrelated.bin'
    unrelated.write_bytes(b'not model weights')
    data['model_hashes'] = {str(unrelated): sha(unrelated)}
    monkeypatch.setenv('HEARING_MODELS_DIR', str(tmp_path / 'models'))
    with pytest.raises(ValueError, match='loaded model'):
        validate_manifest(data, execute=True)
