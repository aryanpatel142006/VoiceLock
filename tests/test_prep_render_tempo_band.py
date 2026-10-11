import json

import numpy as np
import soundfile as sf

from eval.prep_render_tempo_band import (
    Variant,
    enumerate_variants,
    render_source,
)


def test_variant_plan_has_baseline_and_independent_dimensions():
    variants = enumerate_variants(["aryan.wav"], seed=7, max_variants=20)

    assert variants[0] == Variant.baseline()
    assert any(v.tempo == 0.9 and v.sample_rate == 16000 for v in variants)
    assert any(v.bandlimit_hz == 4000 and v.tempo == 1.0 for v in variants)
    assert any(v.clip == 0.8 and v.gain_db == 0.0 for v in variants)
    assert len(variants) <= 20


def test_render_writes_native_audio_and_provenance(tmp_path):
    source = tmp_path / "aryan" / "take.wav"
    source.parent.mkdir()
    sf.write(source, np.sin(np.linspace(0, 20, 1600, dtype=np.float32)), 16000)
    out = tmp_path / "out"

    result = render_source(source, out, [Variant.baseline(), Variant(tempo=0.9)], seed=3)

    assert result["written"] == 2
    audio, sr = sf.read(out / "take__baseline.wav", dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    metadata = json.loads((out / "take__tempo09.json").read_text())
    assert metadata["origin_group"] == "aryan"
    assert metadata["transcript"] is None
    assert metadata["tempo_pitch_change"] is True
    assert metadata["baseline"] is False


def test_render_resume_skips_completed_variant(tmp_path):
    source = tmp_path / "joseph.wav"
    sf.write(source, np.zeros(800, dtype=np.float32), 16000)
    out = tmp_path / "out"
    render_source(source, out, [Variant.baseline()], seed=1)
    first = (out / "joseph__baseline.json").read_text()

    result = render_source(source, out, [Variant.baseline()], seed=1)

    assert result["skipped"] == 1
    assert (out / "joseph__baseline.json").read_text() == first
