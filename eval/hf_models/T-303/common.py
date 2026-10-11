"""Explicit research-only downloads, scratch cleanup and honest smoke measurements."""
import argparse
import contextlib
import hashlib
import json
import os
import platform
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def scratch():
    # All weights, Hub caches and upstream source disappear even on inference failure.
    with tempfile.TemporaryDirectory(prefix='hearing-T303-') as directory:
        yield Path(directory)


class Hub:
    """Network adapter. No implicit downloads during inference; mock mode bypasses it."""

    def __init__(self, root):
        self.root = root
        self.artifacts = []

    def get(self, repo, filename, revision):
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(repo, filename, revision=revision,
                                  cache_dir=self.root / 'hf', token=False))
        self.artifacts.append({'repo': repo, 'file': filename, 'revision': revision,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                               'bytes': path.stat().st_size})
        return path

    def source(self, relative, revision):
        import httpx

        path = self.root / 'upstream' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(f'https://raw.githubusercontent.com/espnet/espnet/{revision}/{relative}',
                             timeout=30, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
        self.artifacts.append({'source': relative, 'revision': revision,
                               'sha256': hashlib.sha256(response.content).hexdigest()})
        return path


def dummy(rate, seconds=2):
    """Seeded, bounded speech-like tones plus noise, NOT real speech or a quality eval."""
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    return (0.06 * np.sin(2 * np.pi * 180 * t)
            + 0.04 * np.sin(2 * np.pi * 310 * t)
            + 0.005 * np.random.default_rng(303).standard_normal(t.shape)).astype(np.float32)


def arrays(output):
    if isinstance(output, (list, tuple)):
        return [value for item in output for value in arrays(item)]
    if hasattr(output, 'detach'):
        output = output.detach().cpu().numpy()
    return [np.asarray(output)]


def measure(call, repeats=5, synchronize=lambda: None):
    def once():
        synchronize()
        start = time.perf_counter()
        output = call()
        synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        values = arrays(output)
        if not values or any(not value.size or not np.isfinite(value).all() for value in values):
            raise ValueError('model output must be nonempty and finite')
        return elapsed, values

    first, _ = once()
    times = []
    for _ in range(repeats):
        elapsed, values = once()
        times.append(elapsed)
    return {'first_inference_ms': first, 'latencies_ms': times,
            'p50_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95)),
            'output_shapes': [list(value.shape) for value in values]}


def session(path):
    import onnxruntime as ort

    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # ORT has no MPS provider. CPU is an explicit runtime constraint, not torch override.
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def torch_runtime():
    import torch

    from hearing.device import pick

    torch.set_num_threads(1)
    device = pick()
    sync = {'cpu': lambda: None, 'cuda': torch.cuda.synchronize,
            'mps': torch.mps.synchronize}[device]
    return torch, device, sync


def run(name, smoke):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mock', action='store_true', help='offline harness check; no model metrics')
    args = parser.parse_args()
    if args.mock:
        print(json.dumps({'model': name, 'mock': True,
                          'note': 'deterministic placeholder, NOT measured model inference'}))
        return
    with scratch() as root:
        os.environ['HF_HOME'] = str(root / 'hf-home')
        os.environ['HF_XET_CACHE'] = str(root / 'xet')
        hub = Hub(root)
        with contextlib.redirect_stdout(sys.stderr):
            result = smoke(hub)
        result.update({'model': name, 'mock': False, 'artifacts': hub.artifacts,
                       'platform': platform.platform(), 'python': platform.python_version(),
                       'hostname': platform.node(), 'threads': 1,
                       'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                       / (1024 ** 2 if sys.platform == 'darwin' else 1024)})
    result['scratch_cleaned'] = not root.exists()
    print(json.dumps(result, indent=2))
