"""Jenthe/ECAPA2 card's torch.jit.load -> model(audio), on 2 s dummy 16 kHz."""
import time

from common import dummy, measure, run, torch_runtime

REPO = 'Jenthe/ECAPA2'
REVISION = '207cb6d137c671a12ba820ebec3b719549b06c0f'


def smoke(hub):
    torch, device, sync = torch_runtime()
    path = hub.get(REPO, 'ecapa2.pt', REVISION)
    start = time.perf_counter()
    model = torch.jit.load(str(path), map_location=device).eval()
    sync()
    load_ms = (time.perf_counter() - start) * 1000
    audio = torch.from_numpy(dummy(16000)[None]).to(device)
    # Model card recommends disabling JIT optimization when first calls are slow.
    with torch.inference_mode(), torch.jit.optimized_execution(False):
        result = measure(lambda: model(audio), synchronize=sync)
    return dict(result, load_ms=load_ms, device=device, sample_rate=16000,
                input_shapes=[list(audio.shape)], input_seconds=2,
                params=sum(p.numel() for p in model.parameters()), license='CC-BY-NC-4.0')


if __name__ == '__main__':
    run(REPO, smoke)
