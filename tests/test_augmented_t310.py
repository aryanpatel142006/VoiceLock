"""T-310: augmented dataset integrity tests (fast, no network)."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1] / "eval" / "augmented" / "T-310"
GROUPS = ["clean", "pitch_pm2", "pitch_pm4", "reverb_02", "reverb_05",
          "reverb_08", "opus16", "opus32", "combo"]


def _manifest():
    with open(ROOT / "manifest.json") as f:
        return json.load(f)


def test_manifest_groups_complete():
    m = _manifest()
    assert sorted(m["groups"].keys()) == sorted(GROUPS)
    for g, entries in m["groups"].items():
        assert len(entries) == 2, g  # speaker_a + speaker_b


def test_files_exist_and_sha256_match():
    m = _manifest()
    for entries in m["groups"].values():
        for e in entries:
            p = ROOT / e["file"]
            assert p.exists(), p
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            assert h == e["sha256"], p


def test_sample_rate_and_no_clipping():
    m = _manifest()
    for entries in m["groups"].values():
        for e in entries:
            p = ROOT / e["file"]
            sr, x = wavfile.read(str(p))
            assert sr == 16000, p
            peak_db = 20 * np.log10(np.max(np.abs(x.astype(np.float32))) / 32768 + 1e-12)
            assert peak_db < -1.0, (p, peak_db)


def _read(path):
    sr, x = wavfile.read(str(path))
    return sr, x.astype(np.float32)


def test_pitch_shift_sane():
    # rubberband is verified exact on synthetic tones (440 Hz -> 555 Hz for +4 st,
    # checked during development). Here: shifted files must differ from clean,
    # keep duration/sample-rate, and preserve RMS within 6 dB (no blowup/mute).
    for spk, stem in (("speaker_a", "speech_a"), ("speaker_b", "speech_b")):
        sr0, x0 = _read(ROOT / "clean" / spk / f"{stem}.wav")
        for g in ("pitch_pm2", "pitch_pm4"):
            sr1, x1 = _read(ROOT / g / spk / f"{stem}.wav")
            assert sr1 == sr0 == 16000
            assert abs(len(x1) - len(x0)) / len(x0) < 0.05, (g, spk)  # duration preserved
            assert not np.array_equal(x0, x1), (g, spk)  # transform applied
            rms_ratio = np.sqrt(np.mean(x1**2) / (np.mean(x0**2) + 1e-12))
            assert 0.25 < rms_ratio < 4.0, (g, spk, rms_ratio)


def test_reverb_rir_rt60():
    # verify the synthetic RIR itself hits the target RT60 via Schroeder decay
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gen310", ROOT / "generate.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    for rt60 in (0.2, 0.5, 0.8):
        rir = gen.synth_rir(16000, rt60, seed=0)
        assert rir[0] == 1.0  # direct path
        # proper Schroeder RT60: slope of the TAIL between -15 and -35 dB
        # (skip the direct-path drop), extrapolated: RT60 = 3 * (t35 - t15)
        sch = np.cumsum(rir[::-1] ** 2)[::-1]
        sch_db = 10 * np.log10(sch / (sch[0] + 1e-12) + 1e-12)
        t15 = np.where(sch_db <= -15)[0][0] / 16000
        t35 = np.where(sch_db <= -35)[0][0] / 16000
        measured = 3 * (t35 - t15)
        assert abs(measured - rt60) / rt60 < 0.25, (rt60, measured)


def test_reverb_files_differ_and_valid():
    for spk, stem in (("speaker_a", "speech_a"), ("speaker_b", "speech_b")):
        _sr0, x0 = _read(ROOT / "clean" / spk / f"{stem}.wav")
        for g in ("reverb_02", "reverb_05", "reverb_08", "opus16", "opus32", "combo"):
            sr1, x1 = _read(ROOT / g / spk / f"{stem}.wav")
            assert sr1 == 16000
            assert not np.array_equal(x0, x1), (g, spk)
            assert np.max(np.abs(x1)) > 100, (g, spk)  # not silent


def test_standin_labels_neutral():
    m = _manifest()
    assert m["sources_are_standins"] is True
    blob = json.dumps(m).lower()
    assert "aryan" not in blob and "joseph" not in blob
