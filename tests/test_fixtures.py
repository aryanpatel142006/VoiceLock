"""Fixture bank: offline tiny set, mixing maths, manifests, determinism, committed clip limits."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval import make_fixtures
from eval.fixtures import mix as fxmix
from eval.fixtures import sources as fxsrc
from eval.fixtures.mix import Stems, fit_length, load_fixture, make_mixture, rms
from hearing.audio import io as aio

REPO = Path(__file__).resolve().parents[1]
TINY_DIR = REPO / "eval" / "fixtures" / "tiny"
TINY_WAVS = ("speech_a.wav", "speech_b.wav", "noise_babble.wav")
STEMS = ("mix", "target", "interferer", "noise")


# --- mixing maths ---------------------------------------------------------------------------


def test_fit_length_trims_and_loops():
    x = np.arange(5, dtype=np.float32)
    assert fit_length(x, 3).tolist() == [0, 1, 2]
    assert fit_length(x, 12).tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1]
    assert fit_length(x, 5) is x or fit_length(x, 5).tolist() == x.tolist()


def test_make_mixture_hits_levels_and_sums_exactly():
    rng = np.random.default_rng(1)
    t = rng.normal(0, 0.3, 16000).astype(np.float32)
    i = rng.normal(0, 0.05, 9000).astype(np.float32)  # shorter: must be looped
    z = rng.normal(0, 1.0, 30000).astype(np.float32)  # longer: must be trimmed
    stems = make_mixture(t, i, z, snr_db=3.0, sir_db=-2.0)
    for name in STEMS:
        assert stems[name].dtype == np.float32 and len(stems[name]) == 16000
    assert abs(stems.measured_snr_db() - 3.0) < 1e-6
    assert abs(stems.measured_sir_db() + 2.0) < 1e-6
    assert abs(float(np.max(np.abs(stems.mix))) - fxmix.PEAK) < 1e-6
    np.testing.assert_allclose(stems.target + stems.interferer + stems.noise, stems.mix, atol=1e-6)


def test_make_mixture_without_interferer_is_noise_only():
    rng = np.random.default_rng(2)
    t = rng.normal(0, 0.3, 16000).astype(np.float32)
    z = rng.normal(0, 1.0, 8000).astype(np.float32)
    stems = make_mixture(t, None, z, snr_db=0.0, sir_db=None)
    assert not np.any(stems.interferer) and len(stems.interferer) == 16000
    assert abs(stems.measured_snr_db()) < 1e-6 and stems.measured_sir_db() == np.inf
    np.testing.assert_allclose(stems.target + stems.noise, stems.mix, atol=1e-6)
    with pytest.raises(ValueError):
        make_mixture(t, z, z, snr_db=0.0, sir_db=None)


def test_make_mixture_rejects_silent_target():
    with pytest.raises(ValueError):
        make_mixture(np.zeros(100, np.float32), np.ones(100, np.float32), np.ones(100, np.float32), 0, 0)


def test_pink_noise_is_unit_rms_and_low_frequency_heavy():
    x = fxsrc.pink_noise(32000, np.random.default_rng(0))
    assert x.dtype == np.float32 and abs(rms(x) - 1.0) < 1e-6
    spec = np.abs(np.fft.rfft(x.astype(np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 16000)
    low = spec[(f > 100) & (f < 500)].mean()
    high = spec[(f > 2000) & (f < 6000)].mean()
    assert low > 3 * high  # roughly 1/f: ~10 dB per decade


def test_babble_sums_normalised_talkers():
    rng = np.random.default_rng(0)
    utts = [
        fxsrc.Utterance(id=str(k), speaker_id=str(k), text="", audio=rng.normal(0, 0.1 * (k + 1), 4000).astype(np.float32))
        for k in range(6)
    ]
    b = fxsrc.make_babble(utts, 8000)
    assert len(b) == 8000 and abs(rms(b) - 1.0) < 1e-6
    with pytest.raises(ValueError):
        fxsrc.make_babble(utts[:1], 100)


# --- committed tiny clips -------------------------------------------------------------------


@pytest.mark.parametrize("name", TINY_WAVS)
def test_committed_tiny_wav_is_small_3s_16k_mono(name):
    path = TINY_DIR / name
    assert path.exists(), f"{name} missing; run `python -m eval.make_fixtures --make-tiny-source`"
    assert os.path.getsize(path) < 100 * 1024
    info = sf.info(str(path))
    assert info.samplerate == 16000 and info.channels == 1
    assert abs(info.frames / info.samplerate - 3.0) < 1e-9
    x, sr = aio.load(path)
    assert sr == 16000 and x.dtype == np.float32 and len(x) == 48000
    assert np.max(np.abs(x)) <= 1.0 and rms(x) > 0.01


def test_tiny_manifest_lists_sources_and_sexes():
    m = json.loads((TINY_DIR / "manifest.json").read_text())
    assert m["dataset"] == "openslr/librispeech_asr" and m["split"] == "test-clean"
    assert "CC BY 4.0" in m["license"]
    speech = m["speech"]
    assert [e["file"] for e in speech] == ["speech_a.wav", "speech_b.wav"]
    assert {e["sex"] for e in speech} == {"F", "M"}
    assert len({e["speaker_id"] for e in speech}) == 2
    for e in speech:
        assert e["id"].startswith(e["speaker_id"] + "-") and e["text"].strip()
    noise = m["noise"]
    assert noise["type"] == "babble" and noise["file"] == "noise_babble.wav"
    assert len(noise["source_ids"]) >= 6
    assert len(set(noise["source_speakers"])) >= 6
    assert not ({e["speaker_id"] for e in speech} & set(noise["source_speakers"]))


# --- tiny set generation (offline) ----------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_set(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixtures")
    manifests = make_fixtures.make_tiny(out, seed=0)
    assert len(manifests) == 16
    return out / "tiny"


def test_tiny_set_has_16_fixtures_with_all_stems_and_index(tiny_set: Path):
    dirs = sorted(p for p in tiny_set.iterdir() if p.is_dir())
    assert len(dirs) == 16
    for d in dirs:
        for stem in STEMS:
            assert (d / f"{stem}.wav").exists(), f"{d.name} lacks {stem}.wav"
        assert (d / "manifest.json").exists()
    index = json.loads((tiny_set / "index.json").read_text())
    assert index["set"] == "tiny" and index["count"] == 16 and index["seed"] == 0
    assert [e["id"] for e in index["fixtures"]] == [d.name for d in dirs]
    for e in index["fixtures"]:
        for stem in STEMS:
            assert (tiny_set / e["paths"][stem]).exists()


def test_tiny_set_covers_grid_in_both_orderings(tiny_set: Path):
    index = json.loads((tiny_set / "index.json").read_text())
    combos = {(e["target_id"], e["snr_db"], e["sir_db"]) for e in index["fixtures"]}
    targets = {e["target_id"] for e in index["fixtures"]}
    assert len(targets) == 2
    assert len(combos) == 16
    assert {e["snr_db"] for e in index["fixtures"]} == {-5.0, 0.0, 5.0, 10.0}
    assert {e["sir_db"] for e in index["fixtures"]} == {0.0, 5.0}
    for e in index["fixtures"]:
        assert e["target_id"] != e["interferer_id"]
        assert e["noise_type"] == "babble"
        assert e["transcript"] == e["transcript"].lower().strip()


def test_tiny_manifest_fields_and_measured_levels(tiny_set: Path):
    required = {
        "id", "set", "sample_rate", "duration_s", "snr_db", "sir_db", "target_id", "interferer_id",
        "noise_type", "transcript", "paths", "seed", "source",
    }
    for d in sorted(p for p in tiny_set.iterdir() if p.is_dir()):
        manifest, stems = load_fixture(d)
        raw = json.loads((d / "manifest.json").read_text())
        assert required <= set(raw), required - set(raw)
        assert manifest.sample_rate == 16000 and abs(manifest.duration_s - 3.0) < 1e-6
        assert manifest.source["dataset"] == "openslr/librispeech_asr"
        assert isinstance(stems, Stems)
        for stem in STEMS:
            assert stems[stem].dtype == np.float32 and len(stems[stem]) == 48000
        assert abs(stems.measured_snr_db() - manifest.snr_db) < 0.1, d.name
        assert abs(stems.measured_sir_db() - manifest.sir_db) < 0.1, d.name
        total = stems.target.astype(np.float64) + stems.interferer + stems.noise
        assert np.max(np.abs(total - stems.mix)) < 1e-4, d.name
        assert float(np.max(np.abs(stems.mix))) <= fxmix.PEAK + 1e-4


def test_tiny_set_is_byte_identical_across_runs(tiny_set: Path, tmp_path: Path):
    make_fixtures.make_tiny(tmp_path, seed=0)
    other = tmp_path / "tiny"
    files = sorted(p.relative_to(tiny_set) for p in tiny_set.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(other) for p in other.rglob("*") if p.is_file())
    for rel in files:
        assert (tiny_set / rel).read_bytes() == (other / rel).read_bytes(), rel


def test_tiny_cli_runs_offline_without_datasets(tmp_path: Path):
    """`--tiny` must not import `datasets` (network) and must finish quickly."""
    code = (
        "import sys, runpy; sys.argv = ['eval.make_fixtures', '--tiny', '--out', sys.argv[1]];\n"
        "try:\n    runpy.run_module('eval.make_fixtures', run_name='__main__')\n"
        "except SystemExit as e:\n    assert (e.code or 0) == 0, e.code\n"
        "assert 'datasets' not in sys.modules, 'tiny path imported datasets'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], cwd=REPO, capture_output=True, text=True, timeout=60, check=False
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "tiny: 16 fixtures" in proc.stdout and "tiny-noise: 16 fixtures" in proc.stdout
    assert len([p for p in (tmp_path / "tiny").iterdir() if p.is_dir()]) == 16
    assert len([p for p in (tmp_path / "tiny-noise").iterdir() if p.is_dir()]) == 16


# --- noise-only sets (target + noise, no interferer) -----------------------------------------


@pytest.fixture(scope="module")
def tiny_noise_set(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixtures_noise")
    manifests = make_fixtures.make_tiny_noise(out, seed=0)
    assert len(manifests) == 16
    return out / "tiny-noise"


def test_tiny_noise_set_covers_targets_noises_and_snrs(tiny_noise_set: Path):
    index = json.loads((tiny_noise_set / "index.json").read_text())
    assert index["set"] == "tiny-noise" and index["count"] == 16
    fixtures = index["fixtures"]
    combos = {(e["target_id"], e["noise_type"], e["snr_db"]) for e in fixtures}
    assert len(combos) == 16 and len({e["target_id"] for e in fixtures}) == 2
    assert {e["noise_type"] for e in fixtures} == {"babble", "pink"}
    assert {e["snr_db"] for e in fixtures} == set(make_fixtures.SNR_GRID_DB)
    for e in fixtures:
        assert e["sir_db"] is None and e["interferer_id"] == "" and e["interferer_transcript"] == ""
        assert "_sir" not in e["id"] and e["id"].endswith(f"_{e['noise_type']}")


def test_tiny_noise_stems_have_no_interferer_and_hit_snr(tiny_noise_set: Path):
    for d in sorted(p for p in tiny_noise_set.iterdir() if p.is_dir()):
        manifest, stems = load_fixture(d)
        assert manifest.sir_db is None and len(stems.mix) == 48000
        assert not np.any(stems.interferer), d.name
        assert abs(stems.measured_snr_db() - manifest.snr_db) < 0.1, d.name
        np.testing.assert_allclose(stems.target + stems.noise, stems.mix, atol=1e-4)


def test_tiny_noise_is_deterministic(tiny_noise_set: Path, tmp_path: Path):
    make_fixtures.make_tiny_noise(tmp_path, seed=0)
    other = tmp_path / "tiny-noise"
    for f in sorted(p for p in tiny_noise_set.rglob("*") if p.is_file()):
        assert f.read_bytes() == (other / f.relative_to(tiny_noise_set)).read_bytes(), f


def test_tiny_noise_cli_alone(tmp_path: Path):
    assert make_fixtures.main(["--tiny-noise", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "tiny-noise" / "index.json").exists() and not (tmp_path / "tiny").exists()


def test_natural_mode_defaults_to_dfn_ll_and_synth_keeps_raw_input():
    from hearing.server.session import DEFAULT_PIPELINES, ServerConfig, parse_pipelines

    assert DEFAULT_PIPELINES["natural"] == "dfn_ll"  # T-103 (T-102's gate chain scored worse, Q-240)
    cfg = ServerConfig()
    assert cfg.pipelines["natural"] == "dfn_ll"
    # the gate chain stays selectable by name (HEARING_PIPELINES / --pipelines)
    assert ServerConfig(pipelines=parse_pipelines("natural=natural_gate")).pipelines["natural"] == "dfn_ll>gate>mixer"
    assert cfg.pipelines["synthetic"].startswith("passthrough>synth(")
    custom = ServerConfig(pipelines={"natural": "dfn>gate"})
    # T-079: the mixture feed stays the raw room even with a configured natural chain
    assert custom.pipelines["text"].startswith("passthrough>synth(")


# --- full set (network) ---------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.slow
def test_full_set_streams_librispeech(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "eval.make_fixtures", "--full", "--out", str(tmp_path), "--seed", "0"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    noise_index = json.loads((tmp_path / "full-noise" / "index.json").read_text())
    assert noise_index["count"] == 96 and all(e["sir_db"] is None for e in noise_index["fixtures"])
    index = json.loads((tmp_path / "full" / "index.json").read_text())
    assert index["count"] == 96
    assert {e["noise_type"] for e in index["fixtures"]} == {"babble", "pink"}
    for e in index["fixtures"][:: 8]:
        manifest, stems = load_fixture(tmp_path / "full" / e["id"])
        assert 4.0 <= manifest.duration_s <= 8.0
        assert manifest.source["target_speaker"] != manifest.source["interferer_speaker"]
        assert abs(stems.measured_snr_db() - manifest.snr_db) < 0.1
        assert abs(stems.measured_sir_db() - manifest.sir_db) < 0.1
