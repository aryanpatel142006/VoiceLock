import json

import numpy as np

from eval import prep_render_gain_pause as p


def _source(tmp_path, name, speaker, role="target", value=0.1):
    wav = tmp_path / f"{name}.wav"
    p.audio_io.save(wav, np.full(1600, value, np.float32), p.SR)
    return p.Source(name, speaker, role, wav, "exact words", (), "final", {})


def test_sir_preserves_target_amplitude_and_renders_overrange(tmp_path):
    a = _source(tmp_path, "a", "A", value=0.8)
    b = _source(tmp_path, "b", "B", "interferer", value=0.8)
    case = p.Case("x", a, b, None, -15, "target_on", 0, True, (), 3)
    mix, meta = p.render_case(case)
    assert np.allclose(mix[:1600], 0.8 + 0.8 * 10 ** (15 / 20), rtol=1e-3)
    assert meta["declared_overrange"] is True


def test_pause_and_off_do_not_claim_transcript(tmp_path):
    a = _source(tmp_path, "a", "A")
    b = _source(tmp_path, "b", "B", "interferer")
    pause = p.Case("pause", a, b, None, 0, "target_pause", 0, True, ((0.02, 0.05),), 0)
    off = p.Case("off", a, b, None, 0, "target_off", 0, False, (), 0)
    _, pause_meta = p.render_case(pause)
    _, off_meta = p.render_case(off)
    assert pause_meta["transcript"] == off_meta["transcript"] == ""


def test_discovery_and_checkpoint_resume(tmp_path):
    root = tmp_path / "sources" / "case"
    root.mkdir(parents=True)
    a, b = _source(root, "a", "A"), _source(root, "b", "B", "interferer")
    (root / "manifest.json").write_text(json.dumps({
        "id": "case", "set": "enrol", "duration_s": 0.1,
        "target_id": a.source_id, "interferer_id": b.source_id,
        "source": {"target_speaker": "A", "interferer_speaker": "B"},
        "paths": {"target": "a.wav", "interferer": "b.wav"},
    }))
    sources = p.discover_sources(tmp_path / "sources")
    assert {s.speaker_id for s in sources} == {"A", "B"}
    cases = p.plan_cases(sources, limit=2)
    out = tmp_path / "out"
    assert p.render_batch(cases, out) == 2
    assert p.render_batch(cases, out) == 2
    assert len(list(out.glob("*/mix.wav"))) == 2
