"""Offline random-embedding smoke and CPU microbenchmark; no accuracy claims."""
import importlib.util
import json
import platform
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    borrowed = load('third_party/borrowed/T-302/projection.py', 'borrowed_projection')
    pilot = load('eval/projection_head_pilot.py', 'projection_pilot')
    torch.manual_seed(302)
    torch.set_num_threads(1)
    x = torch.randn(1, 192)
    results = {'device': 'cpu', 'machine': platform.machine(), 'torch': torch.__version__,
               'seed': 302, 'batch': 1, 'threads': 1, 'iterations': 1000, 'heads': {}}
    with torch.inference_mode():
        for name, head in [('borrowed', borrowed.ProjectionHead()),
                           ('T-273', pilot.TeamProjectionHead())]:
            head.eval()
            for _ in range(100):
                head(x)
            samples = []
            for _ in range(1000):
                start = time.perf_counter_ns()
                z = head(x)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            assert z.shape == (1, 32) and torch.isfinite(z).all()
            torch.testing.assert_close(z.norm(dim=-1), torch.ones(1))
            results['heads'][name] = {
                'parameters': sum(p.numel() for p in head.parameters()),
                'p50_ms': sorted(samples)[500], 'p95_ms': sorted(samples)[950],
                'output_shape': list(z.shape),
            }
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
