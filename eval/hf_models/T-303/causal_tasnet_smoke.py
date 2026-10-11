"""Cadenza's load_model.py example from local snapshot, 2 s dummy stereo 44.1 kHz."""
import importlib.util
import time

from common import dummy, measure, run, torch_runtime

REPO = 'cadenzachallenge/ConvTasNet_Lyrics_Causal'
REVISION = 'd2601a366fd8a3bd969852ca5bced524b6c8aae9'


def smoke(hub):
    torch, device, sync = torch_runtime()
    code = hub.get(REPO, 'tasnet.py', REVISION)
    hub.get(REPO, 'config.json', REVISION)
    hub.get(REPO, 'model.safetensors', REVISION)
    spec = importlib.util.spec_from_file_location('t303_cadenza_tasnet', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = time.perf_counter()
    model = module.ConvTasNetStereo.from_pretrained(str(code.parent), local_files_only=True)
    model = model.to(device).eval()
    sync()
    load_ms = (time.perf_counter() - start) * 1000
    audio = torch.from_numpy(dummy(44100)[None, None]).repeat(1, 2, 1).to(device)
    with torch.inference_mode():
        result = measure(lambda: model(audio), synchronize=sync)
        # Causality check: future input must not alter earlier output away from
        # the 20-sample encoder/decoder boundary. Causal flag alone is insufficient.
        modified = audio.clone()
        modified[:, :, 44100:] = 0
        first = model(audio)[..., :44000]
        second = model(modified)[..., :44000]
        delta = (first - second).abs().max().item()
        prefix = model(audio[..., :44100])[..., :44000]
        prefix_delta = (prefix - first).abs().max().item()
        torch.testing.assert_close(prefix, first, atol=1e-4, rtol=1e-4)
    return dict(result, load_ms=load_ms, device=device, sample_rate=44100, input_seconds=2,
                input_shapes=[list(audio.shape)], params=sum(p.numel() for p in model.parameters()),
                license='Apache-2.0 weights; MIT-derived inference code',
                future_change_prefix_max_delta=delta,
                prefix_recompute_max_delta=prefix_delta,
                streaming_scope='causal architecture; no explicit stateful chunk API in release')


if __name__ == '__main__':
    run(REPO, smoke)
