import numpy as np
import pytest

from eval import perturb_profiles as p


def test_transform_audio_preserves_bounds_and_duration_for_pitch_and_formant():
    audio = p.fake_voice(140, 1.0)
    out = p.transform_audio(audio, pitch_semitones=3, formant_semitones=-2)
    assert out.dtype == np.float32
    assert len(out) == len(audio)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0
    assert not np.allclose(out, audio)


def test_speed_shift_changes_duration_and_rejects_invalid_values():
    audio = p.fake_voice(140, 1.0)
    faster = p.transform_audio(audio, speed=1.25)
    assert len(faster) == round(len(audio) / 1.25)
    with np.testing.assert_raises(ValueError):
        p.transform_audio(audio, speed=0)


def test_perturbation_rows_report_cosine_degradation():
    audio = p.fake_voice(140, 1.0)
    embedder = p.Embedder(p.FakeBackend())
    rows = p.measure_perturbations(
        audio,
        embedder,
        [{"id": "pitch+3", "pitch_semitones": 3}],
    )
    assert rows[0]["id"] == "pitch+3"
    assert 0.0 <= rows[0]["cosine"] <= 1.0
    assert rows[0]["cosine_degradation"] == pytest.approx(1.0 - rows[0]["cosine"])


def test_manifest_contains_hashes_but_no_audio_payload(tmp_path):
    audio = p.fake_voice(140, 1.0)
    manifest = p.write_manifest(
        tmp_path / "manifest.json",
        [{"id": "baseline", "audio": audio, "cosine": 1.0}],
        source="fixture.wav",
    )
    assert manifest["source"] == "fixture.wav"
    assert manifest["rows"][0]["sha256"]
    assert "audio" not in manifest["rows"][0]
