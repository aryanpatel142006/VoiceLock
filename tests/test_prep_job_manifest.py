import hashlib
import json
import subprocess

import pytest

from eval.prep_job_manifest import build_manifest, main, verified_checkpoint


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'runner.py').write_text("raise RuntimeError('must never execute')\n")
    subprocess.run(['git', '-C', str(tmp_path), 'add', 'runner.py'], check=True)
    subprocess.run(['git', '-C', str(tmp_path), '-c', 'user.name=Test', '-c',
                    'user.email=test@example.org', 'commit', '-qm', 'fixture'], check=True)
    revision = subprocess.check_output(['git', '-C', str(tmp_path), 'rev-parse', 'HEAD'], text=True).strip()
    (tmp_path / 'input.bin').write_bytes(b'fixture')
    (tmp_path / 'config.json').write_text('{}')
    return tmp_path, revision


def job(root, **overrides):
    value = {'id': 't210-baseline', 'script': 'runner.py', 'script_sha256': sha(root / 'runner.py'),
                 'inputs': [{'path': 'input.bin', 'sha256': sha(root / 'input.bin')}],
                 'outputs': ['results/a.json'], 'cases': 2, 'timeout_s': 60, 'metrics': ['si_sdri'],
                 'metric_dependencies': {'target_reference': 'input.bin', 'mix_reference': 'input.bin'},
                 'seed': 236, 'max_output_bytes': 1024, 'config_path': 'config.json', 'config_hash': sha(root / 'config.json'),
                 'requires_cuda': False, 'thresholds': {'si_sdri': 0}, 'evidence': 'planned'}
    value.update(overrides)
    return value


def write_plan(root, revision, jobs, name='plan.json'):
    path = root / name
    path.write_text(json.dumps({'source_revision': revision, 'jobs': jobs}))
    return path


def build(prepared, jobs, **kwargs):
    root, revision = prepared
    return build_manifest([write_plan(root, revision, jobs)], repo_root=root, **kwargs)


def test_available_counts_and_replayable_cli(prepared):
    root, rev = prepared
    path = write_plan(root, rev, [job(root)])
    out = root / 'manifest.json'
    assert main(['--plan', str(path), '--repo-root', str(root), '--dry-run', '--out', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['summary']['available_inputs'] == 1
    assert result['summary']['planned_cases'] == 2
    assert result['jobs'][0]['status'] == 'ready'
    assert result['jobs'][0]['timeout_total_s'] == 120
    assert result['jobs'][0]['recipe_sha256']
    assert result['contract']['never_executes']


def test_all_duplicate_and_colliding_jobs_pending(prepared):
    root, _ = prepared
    result = build(prepared, [job(root), job(root), job(root, id='other', outputs=['results/../results/a.json'])])
    assert result['summary']['ready_jobs'] == 0
    assert any('duplicate job id' in x for x in result['errors'])
    assert all(any('output collision' in x for x in j['reasons']) for j in result['jobs'])


@pytest.mark.parametrize('overrides, reason', [
    ({'cases': 301}, 'cap'), ({'cases': 300, 'timeout_s': 97}, 'cap'),
    ({'cases': True}, 'cases'), ({'timeout_s': float('nan')}, 'timeout'),
    ({'timeout_s': float('inf')}, 'timeout'), ({'timeout_s': True}, 'timeout'),
    ({'requires_cuda': True}, 'capability'),
    ({'requires_cuda': True, 'capability': 'a100'}, 'capability'),
    ({'metrics': ['unknown']}, 'unknown metric'),
    ({'metric_dependencies': {}}, 'metric dependency'),
    ({'thresholds': {'si_sdri': float('nan')}}, 'threshold'),
    ({'thresholds': {'unknown': 1}}, 'threshold'),
    ({'inputs': [{'path': 'missing.bin', 'sha256': '0' * 64}]}, 'missing input'),
    ({'inputs': [{'path': 'input.bin', 'sha256': '0' * 64}]}, 'hash mismatch'),
    ({'model_path': 'input.bin', 'model_hash': '0' * 64}, 'model hash mismatch'),
    ({'profile_hash': '0' * 64}, 'profile'),
    ({'config_hash': '0' * 64}, 'config hash mismatch'),
    ({'script_sha256': '0' * 64}, 'script hash mismatch'),
    ({'seed': None}, 'seed'), ({'lock_path': '/tmp/other.lock'}, 'lock'),
])
def test_unsafe_jobs_never_ready(prepared, overrides, reason):
    root, _ = prepared
    result = build(prepared, [job(root, **overrides)])
    assert result['jobs'][0]['status'] == 'pending'
    assert reason in ' '.join(result['jobs'][0]['reasons'])
    assert result['errors']


def test_cuda_explicit_capability_and_8h_boundary(prepared):
    root, _ = prepared
    result = build(prepared, [job(root, cases=300, timeout_s=96, requires_cuda=True, capability='a100')],
                   capabilities={'cuda', 'a100'})
    assert result['summary']['ready_jobs'] == 1
    assert result['summary']['planned_seconds'] == 28800
    over = build(prepared, [job(root, cases=150, timeout_s=97),
                           job(root, id='second', cases=150, timeout_s=97, outputs=['results/b.json'])])
    assert over['summary']['ready_jobs'] == 0


@pytest.mark.parametrize('revision', [None, 'unknown', 'deadbeef'])
def test_missing_or_unresolvable_source_revision(prepared, revision):
    root, _ = prepared
    result = build_manifest([write_plan(root, revision, [job(root)])], repo_root=root)
    assert result['jobs'][0]['status'] == 'pending'
    assert 'source revision' in ' '.join(result['jobs'][0]['reasons'])


def test_uncommitted_script_cannot_claim_source_revision(prepared):
    root, _ = prepared
    (root / 'runner.py').write_text('changed')
    assert build(prepared, [job(root)])['summary']['ready_jobs'] == 0


def test_incremental_prepared_plans_and_missing_plan(prepared):
    root, rev = prepared
    first = write_plan(root, rev, [job(root)], 't210.json')
    missing = root / 't211.json'
    assert main(['--plan', str(first), '--plan', str(missing), '--repo-root', str(root),
                 '--out', str(root / 'before.json')]) == 2
    before = build_manifest([first, missing], repo_root=root)
    assert before['summary']['available_plans'] == 1
    assert before['summary']['pending_plans'] == 1
    assert before['summary']['ready_jobs'] == 1
    write_plan(root, rev, [job(root, id='t211-augmentation', outputs=['results/b.json'])], 't211.json')
    assert main(['--plan', str(first), '--plan', str(missing), '--repo-root', str(root),
                 '--out', str(root / 'after.json')]) == 0
    after = build_manifest([first, missing], repo_root=root)
    assert after['summary']['ready_jobs'] == 2
    assert after['summary']['available_inputs'] == 2
    assert after['summary']['planned_seconds'] == 240
    assert before['jobs'][0]['recipe_sha256'] == after['jobs'][0]['recipe_sha256']


def test_resume_verifies_identity_and_output_bytes(prepared):
    root, _ = prepared
    result = build(prepared, [job(root)], resume=True)
    recipe = result['jobs'][0]['recipe_sha256']
    output = root / 'results/a.json'
    output.parent.mkdir()
    output.write_text('{}')
    checkpoint = root / 'checkpoint.json'
    data = {'status': 'complete', 'recipe_sha256': recipe, 'case_id': 'case-1',
                'outputs': [{'path': 'results/a.json', 'sha256': sha(output)}]}
    checkpoint.write_text(json.dumps(data))
    assert verified_checkpoint(checkpoint, recipe, 'case-1', root)
    assert not verified_checkpoint(checkpoint, 'different', 'case-1', root)
    assert not verified_checkpoint(checkpoint, recipe, 'case-2', root)
    output.write_text('corrupted')
    assert not verified_checkpoint(checkpoint, recipe, 'case-1', root)
    data['status'] = 'failed'
    checkpoint.write_text(json.dumps(data))
    assert not verified_checkpoint(checkpoint, recipe, 'case-1', root)
    assert result['contract']['resume_checkpoint']['atomic']


@pytest.mark.parametrize('overrides', [
    {'capability': []}, {'model_path': 12},
    {'metric_dependencies': {'target_reference': [], 'mix_reference': 'input.bin'}},
    {'args': [float('nan')]}, {'requires_cuda': 'yes'},
])
def test_malformed_fields_produce_pending_recipes(prepared, overrides):
    root, _ = prepared
    result = build(prepared, [job(root, **overrides)])
    assert result['summary']['ready_jobs'] == 0
    json.dumps(result, allow_nan=False)


def test_cuda_cannot_use_cpu_capability(prepared):
    root, _ = prepared
    assert build(prepared, [job(root, requires_cuda=True, capability='cpu')],
                 capabilities={'cpu'})['summary']['ready_jobs'] == 0


def test_audio_reference_requires_provenance(prepared):
    root, _ = prepared
    (root / 'ref.wav').write_bytes(b'not real audio; hash-only fixture')
    item = {'path': 'ref.wav', 'sha256': sha(root / 'ref.wav')}
    assert build(prepared, [job(root, inputs=[item])])['summary']['ready_jobs'] == 0


def test_source_revision_assertion_cannot_override_git(prepared):
    root, _ = prepared
    assert build(prepared, [job(root)], git_revision='fake')['summary']['ready_jobs'] == 0


def test_missing_script_and_cli_pending_exit(prepared):
    root, rev = prepared
    path = write_plan(root, rev, [job(root, script='absent.py')])
    assert main(['--plan', str(path), '--repo-root', str(root), '--out', str(root / 'out.json')]) == 2


def test_output_byte_cap_and_input_count_cap(prepared):
    root, _ = prepared
    assert build(prepared, [job(root, max_output_bytes=2**30 + 1)])['summary']['ready_jobs'] == 0
    assert build(prepared, [job(root, inputs=job(root)['inputs'] * 501)])['summary']['ready_jobs'] == 0


def test_capability_is_in_recipe_identity(prepared):
    root, _ = prepared
    a = build(prepared, [job(root, requires_cuda=True, capability='a100')],
              capabilities={'cuda', 'a100', 'h100'})['jobs'][0]
    b = build(prepared, [job(root, requires_cuda=True, capability='h100')],
              capabilities={'cuda', 'a100', 'h100'})['jobs'][0]
    assert a['capability'] == 'a100'
    assert a['recipe_sha256'] != b['recipe_sha256']


def test_resume_rehashes_generated_media_outputs_without_source_metadata(prepared):
    root, _ = prepared
    recipe = build(prepared, [job(root)])['jobs'][0]['recipe_sha256']
    output = root / 'rendered.wav'
    output.write_bytes(b'generated checkpoint fixture')
    checkpoint = root / 'checkpoint.json'
    checkpoint.write_text(json.dumps({'status': 'complete', 'recipe_sha256': recipe,
                                     'case_id': 'c1', 'outputs': [
                                         {'path': 'rendered.wav', 'sha256': sha(output)}]}))
    assert verified_checkpoint(checkpoint, recipe, 'c1', root)
