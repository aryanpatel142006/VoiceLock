import json
from hashlib import sha256

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import fftconvolve

from eval import prep_team_acoustic_stress as prep
from hearing.audio import io as aio


def _tone(freq: float, seconds: float = 0.4) -> np.ndarray:
    t = np.arange(round(seconds * prep.SAMPLE_RATE)) / prep.SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_prepare_writes_all_team_profile_acoustic_conditions(tmp_path):
    profiles = tmp_path / "profiles"
    for name, freq in (("aryan", 180), ("joseph", 220)):
        directory = profiles / name / "v1"
        directory.mkdir(parents=True)
        aio.save(directory / "piece-01.wav", _tone(freq), prep.SAMPLE_RATE)
    noise = _tone(70, 0.8)
    noise_path = tmp_path / "babble.wav"
    aio.save(noise_path, noise, prep.SAMPLE_RATE)
    output = tmp_path / "generated"

    manifest_path = prep.prepare(profiles, noise_path, output)

    manifest = json.loads(manifest_path.read_text())
    assert manifest["sample_rate"] == 16000
    assert manifest["rt60_s"] == [0.2, 0.5]
    assert manifest["snr_db"] == [5.0, 10.0]
    assert len(manifest["pairs"]) == 8
    assert {p["speaker"] for p in manifest["pairs"]} == {"aryan", "joseph"}
    assert {p["rt60_s"] for p in manifest["pairs"]} == {0.2, 0.5}
    assert {p["snr_db"] for p in manifest["pairs"]} == {5.0, 10.0}
    for pair in manifest["pairs"]:
        mix, sr = aio.load(output / pair["mix"], prep.SAMPLE_RATE)
        target, _ = aio.load(output / pair["target"], prep.SAMPLE_RATE)
        assert sr == prep.SAMPLE_RATE and len(mix) == len(target)
        assert np.max(np.abs(mix)) <= 1.0
        assert pair["measured_snr_db"] == pytest.approx(pair["snr_db"], abs=0.05)


@pytest.mark.parametrize("rt60", [0.2, 0.5])
def test_rir_signed_unit_energy_and_schroeder_rt60(rt60):
    rir = prep.make_rir(rt60)
    assert np.any(rir[1:] < 0) and np.any(rir[1:] > 0)
    assert np.sum(rir.astype(np.float64) ** 2) == pytest.approx(1, abs=1e-6)
    assert rir[0] > 0
    energy = np.cumsum((rir.astype(np.float64) ** 2)[::-1])[::-1]
    decay_db = 10 * np.log10(energy / energy[0])
    window = (decay_db < -5) & (decay_db > -35)
    times = np.arange(len(rir)) / 16000
    slope = np.polyfit(times[window], decay_db[window], 1)[0]
    assert -60 / slope == pytest.approx(rt60, rel=0.08)


def test_rir_seed_controls_signed_tail():
    a = prep.make_rir(0.2, seed=17)
    np.testing.assert_array_equal(a, prep.make_rir(0.2, seed=17))
    assert not np.array_equal(a, prep.make_rir(0.2, seed=18))


@pytest.mark.parametrize("rt60", [0, -1, float("nan"), float("inf")])
def test_rir_rejects_invalid_decay(rt60):
    with pytest.raises(ValueError):
        prep.make_rir(rt60)


def test_missing_inputs_leave_no_output(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        prep.prepare(tmp_path / "profiles", tmp_path / "missing.wav", out)
    aio.save(tmp_path / "noise.wav", _tone(70), 16000)
    with pytest.raises(FileNotFoundError):
        prep.prepare(tmp_path / "profiles", tmp_path / "noise.wav", out)
    assert not out.exists()


def test_short_noise_repeats_and_outputs_are_deterministic(tmp_path):
    profiles = tmp_path / "profiles"
    for speaker in ("aryan", "joseph"):
        for version in ("v1", "v2"):
            folder = profiles / speaker / version
            folder.mkdir(parents=True)
            aio.save(folder / "piece.wav", _tone(200, 0.05), 16000)
    noise_path = tmp_path / "noise.wav"
    aio.save(noise_path, _tone(70, 0.01), 16000)
    outputs = [tmp_path / "a", tmp_path / "b"]
    manifests = [json.loads(prep.prepare(profiles, noise_path, out).read_text()) for out in outputs]
    assert manifests[0] == manifests[1]
    assert manifests[0]["acoustics"] == "synthetic_not_measured_venue"
    pairs = manifests[0]["pairs"]
    assert len({p["id"] for p in pairs}) == 16
    for pair in pairs:
        a, _ = aio.load(outputs[0] / pair["mix"])
        b, _ = aio.load(outputs[1] / pair["mix"])
        np.testing.assert_array_equal(a, b)
        target, _ = aio.load(outputs[0] / pair["target"])
        measured = 20 * np.log10(np.linalg.norm(target) / np.linalg.norm(a - target))
        assert measured == pytest.approx(pair["snr_db"], abs=0.05)


def test_cancelling_noise_does_not_clip_target_reference(tmp_path):
    # A quieter mixture must not let the louder reference clip during PCM16 save.
    profiles = tmp_path / "profiles"
    source = (0.95 * np.random.default_rng(4).choice([-1, 1], 16000)).astype(np.float32)
    for speaker in ("aryan", "joseph"):
        folder = profiles / speaker
        folder.mkdir(parents=True)
        aio.save(folder / "piece.wav", source, 16000)
    saved_source, _ = aio.load(profiles / "aryan" / "piece.wav")
    seed = int(sha256(b"aryan/piece.wav").hexdigest()[:16], 16)
    cancelling_noise = -fftconvolve(saved_source, prep.make_rir(0.2, seed=seed))
    noise_path = tmp_path / "noise.wav"
    # Keep the anti-correlated waveform intact rather than clipping the input.
    sf.write(noise_path, cancelling_noise, 16000, subtype="FLOAT")
    output = tmp_path / "out"
    manifest = json.loads(prep.prepare(profiles, noise_path, output).read_text())
    pair = next(
        p for p in manifest["pairs"]
        if p["speaker"] == "aryan" and p["rt60_s"] == 0.2 and p["snr_db"] == 5
    )
    target, _ = aio.load(output / pair["target"])
    assert np.max(np.abs(target)) <= 0.9001
    assert pair["measured_snr_db"] == pytest.approx(5, abs=0.05)
