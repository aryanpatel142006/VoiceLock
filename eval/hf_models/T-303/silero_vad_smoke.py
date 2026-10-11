"""Silero ONNX wrapper example: 512 samples + 64 context with carried recurrent state."""
import time

from common import dummy, measure, run, session

REPO = 'onnx-community/silero-vad'
REVISION = 'e71cae966052b992a7eca6b17738916ce0eca4ec'


def smoke(hub):
    import numpy as np
    import onnx

    path = hub.get(REPO, 'onnx/model.onnx', REVISION)
    start = time.perf_counter()
    model = session(path)
    load_ms = (time.perf_counter() - start) * 1000
    audio = dummy(16000, seconds=0.032)[None]
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)

    def infer():
        nonlocal state, context
        probability, state = model.run(None, {'input': np.concatenate([context, audio], axis=1),
                                             'state': state, 'sr': np.array(16000, dtype=np.int64)})
        context = audio[:, -64:]
        if not np.all((probability >= 0) & (probability <= 1)):
            raise ValueError('VAD probability outside [0, 1]')
        return probability, state

    result = measure(infer)
    graph = onnx.load(str(path))
    # Export puts parameters inside If subgraphs, not just the root graph.
    def count(graph):
        return sum(int(np.prod(t.dims)) for t in graph.initializer) + sum(
            count(a.g) for node in graph.node for a in node.attribute if a.type == onnx.AttributeProto.GRAPH)
    return dict(result, load_ms=load_ms, device='cpu', sample_rate=16000, input_seconds=0.032,
                input_shapes=[[1, 576], [2, 1, 128], []], params=count(graph.graph),
                params_kind='ONNX initializer elements (both rate branches, includes buffers)',
                license='MIT')


if __name__ == '__main__':
    run(REPO, smoke)
