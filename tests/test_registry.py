from dataclasses import replace

import numpy as np
import pytest


def setup_stages(monkeypatch):
    from hearing import registry

    monkeypatch.setattr(registry, "_factories", {})
    calls = []

    class Dummy:
        sample_rate = 16000

        def __init__(self, name, latency_ms, x=1, **kwargs):
            self.name, self.latency_ms, self.x = name, latency_ms, x
            self.kwargs = kwargs
            self.count = 0

        def process(self, chunk, ctx):
            calls.append(self.name)
            self.count += len(chunk.samples)
            return replace(chunk, samples=chunk.samples * self.x + 0.01)

        def reset(self):
            self.count = 0

    for name, latency in [("a", 1.5), ("b", 2.0), ("c", 3.5)]:

        def factory(name=name, latency=latency, **kwargs):
            return Dummy(name, latency, **kwargs)

        registry.register(name)(factory)
    return registry, calls


def test_pipeline_orders_processing_sums_latency_and_resets(monkeypatch):
    from hearing.types import AudioChunk, Stage, StageContext

    registry, calls = setup_stages(monkeypatch)
    pipeline = registry.build_pipeline("a>b(x=2)>c")
    assert isinstance(pipeline, Stage)
    assert pipeline.name == "a>b(x=2)>c"
    assert pipeline.sample_rate == 16000
    assert pipeline.latency_ms == 7.0
    chunk = AudioChunk(np.full(160, 0.1, dtype=np.float32), 16000, 9, 2.5)
    output = pipeline.process(chunk, StageContext.default())
    assert calls == ["a", "b", "c"]
    np.testing.assert_allclose(output.samples, 0.24)
    assert (output.seq, output.capture_ts, output.sample_rate) == (9, 2.5, 16000)
    assert output.samples.dtype == np.float32
    assert all(stage.count == 160 for stage in pipeline.stages)
    pipeline.reset()
    assert all(stage.count == 0 for stage in pipeline.stages)


def test_chunked_pipeline_matches_whole_file(monkeypatch):
    from hearing.types import AudioChunk, StageContext

    registry, _ = setup_stages(monkeypatch)
    samples = np.linspace(-0.2, 0.2, 480, dtype=np.float32)
    ctx = StageContext.default()
    whole = registry.build_pipeline("a>b(x=2)>c").process(
        AudioChunk(samples, 16000, 0, 0.0), ctx
    )
    pipeline = registry.build_pipeline("a>b(x=2)>c")
    parts = [
        pipeline.process(AudioChunk(part, 16000, i, i * 0.01), ctx).samples
        for i, part in enumerate(np.split(samples, 3))
    ]
    np.testing.assert_array_equal(np.concatenate(parts), whole.samples)


def test_get_returns_fresh_stages_and_names_are_sorted(monkeypatch):
    registry, _ = setup_stages(monkeypatch)
    assert registry.names() == ["a", "b", "c"]
    assert registry.get("a") is not registry.get("a")
    assert registry.get("b", x=3).x == 3


def test_unknown_stage_lists_known_names(monkeypatch):
    registry, _ = setup_stages(monkeypatch)
    with pytest.raises(KeyError, match="missing.*a.*b.*c"):
        registry.build_pipeline("a>missing")


def test_sample_rate_mismatch_rejected_at_build(monkeypatch):
    registry, _ = setup_stages(monkeypatch)

    @registry.register("other")
    def other():
        stage = registry.get("a")
        stage.sample_rate = 48000
        return stage

    with pytest.raises(ValueError, match="16000.*48000"):
        registry.build_pipeline("a>other")


def test_parser_preserves_scalar_types_and_quoted_delimiters(monkeypatch):
    registry, _ = setup_stages(monkeypatch)
    stage = registry.build_pipeline(
        " a(x=-2,ratio=1e-3,on=true,off=False,label=gate,path='a>b,c=(x)',word='2') "
    ).stages[0]
    assert stage.x == -2 and type(stage.x) is int
    assert stage.kwargs == {
        "ratio": 0.001,
        "on": True,
        "off": False,
        "label": "gate",
        "path": "a>b,c=(x)",
        "word": "2",
    }
    assert type(stage.kwargs["ratio"]) is float
    assert stage.kwargs["on"] is True
    assert stage.kwargs["off"] is False


@pytest.mark.parametrize(
    "spec",
    [
        "",
        " ",
        ">a",
        "a>",
        "a>>b",
        "a(x)",
        "a(x=)",
        "a(x=1,x=2)",
        "a(x=1,)",
        "a(x='open)",
        "a(x=1",
        "a(x=[1,2])",
        "a(x=func())",
        "a(1=2)",
        "a()junk",
        "a(x=1+2)",
    ],
)
def test_malformed_pipeline_rejected(monkeypatch, spec):
    registry, _ = setup_stages(monkeypatch)
    with pytest.raises(ValueError):
        registry.build_pipeline(spec)


def test_duplicate_registration_rejected(monkeypatch):
    registry, _ = setup_stages(monkeypatch)
    with pytest.raises(ValueError, match="a"):
        registry.register("a")(lambda: None)


def test_stage_module_run_as_main_keeps_package_factory():
    """`python -m hearing.server.remote` imports remote.py twice (package, then __main__)."""
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-m", "hearing.server.remote", "--help"],
                         capture_output=True, text=True, timeout=60, check=False)
    assert out.returncode == 0, out.stderr
    assert "Probe a remote session server" in out.stdout
