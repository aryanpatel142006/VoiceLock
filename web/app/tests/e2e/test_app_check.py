"""Guard the installability gate against incomplete manifests."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def checker():
    path = ROOT / 'scripts/app_check.py'
    assert path.exists(), 'The app check command must exist'
    spec = importlib.util.spec_from_file_location('app_check', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANIFEST = {
    'name': 'Test app', 'short_name': 'Test', 'id': '/app/',
    'start_url': '/app/', 'scope': '/app/', 'display': 'standalone',
    'theme_color': '#07080a', 'background_color': '#07080a',
    'icons': [
        {'src': '192.png', 'sizes': '192x192', 'type': 'image/png'},
        {'src': '512.png', 'sizes': '512x512', 'type': 'image/png'},
        {'src': 'mask.png', 'sizes': '512x512', 'type': 'image/png', 'purpose': 'maskable'},
    ],
}


def test_valid_installable_manifest():
    checker().validate_manifest(MANIFEST)


@pytest.mark.parametrize('field,value', [
    ('name', ''), ('start_url', '/judge'), ('scope', '/'),
    ('display', 'browser'), ('icons', []),
    ('icons', [MANIFEST['icons'][0], MANIFEST['icons'][2]]),
    ('icons', MANIFEST['icons'][:2]),
])
def test_rejects_noninstallable_manifest(field, value):
    manifest = copy.deepcopy(MANIFEST)
    manifest[field] = value
    module = checker()
    with pytest.raises(AssertionError):
        module.validate_manifest(manifest)


@pytest.mark.slow
@pytest.mark.parametrize('fault', [None, 'manifest', 'console'])
def test_command_journey_and_failure_exit(tmp_path, monkeypatch, fault):
    """Real TLS/browser boundary: broken install metadata or console errors must fail the command."""
    import json

    from playwright.sync_api import Browser

    module = checker()
    if fault:
        # Faults enter at the browser boundary, never in the checker or application state.
        original = Browser.new_context

        def context_with_fault(browser, *args, **kwargs):
            context = original(browser, *args, **kwargs)
            if fault == 'console':
                context.add_init_script("console.error('T-354 injected console failure')")
            else:
                # A broken manifest link must fail, even when the shell renders.
                context.add_init_script('''addEventListener('DOMContentLoaded', () => {
                  document.querySelector('link[rel="manifest"]').href = '/app/index.html';
                });''')
            return context

        monkeypatch.setattr(Browser, 'new_context', context_with_fault)
    rc = module.main(['--output', str(tmp_path)])
    result = json.loads((tmp_path / 'results.json').read_text())
    assert rc == (1 if fault else 0)
    assert result['passed'] is (fault is None)
    if fault == 'console':
        assert 'T-354 injected console failure' in result['error']
    elif fault is None:
        assert len(result['checks']) == 3
        assert len(list(tmp_path.glob('*.png'))) == 18
