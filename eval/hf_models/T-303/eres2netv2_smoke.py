"""3D-Speaker ERes2NetV2 ONNX: mean-normalized 80-bin Kaldi fbank -> embedding."""
import time

from common import dummy, measure, run, session

REPO = 'csukuangfj/speaker-embedding-models'
REVISION = '0743f301363dec56491a490f6d6cbc9d67f9a3bf'
FILE = '3dspeaker_speech_eres2netv2_sv_zh-cn_16k-common.onnx'


def smoke(hub):
    import numpy as np
    import onnx
    import torch
    import torchaudio

    torch.set_num_threads(1)
    path = hub.get(REPO, FILE, REVISION)
    start = time.perf_counter()
    model = session(path)
    load_ms = (time.perf_counter() - start) * 1000
    audio = torch.from_numpy(dummy(16000)[None])
    features = torchaudio.compliance.kaldi.fbank(audio * 32768, num_mel_bins=80,
                                               sample_frequency=16000, dither=0)
    features -= features.mean(dim=0, keepdim=True)
    feed = {model.get_inputs()[0].name: features.numpy()[None]}
    result = measure(lambda: model.run(None, feed))
    graph = onnx.load(str(path))
    params = sum(int(np.prod(t.dims)) for t in graph.graph.initializer)
    return dict(result, load_ms=load_ms, device='cpu', sample_rate=16000,
                input_shapes=[list(features.numpy()[None].shape)], input_seconds=2,
                params=params, params_kind='ONNX initializer elements (includes buffers)',
                license='Apache-2.0')


if __name__ == '__main__':
    run('ERes2NetV2', smoke)
