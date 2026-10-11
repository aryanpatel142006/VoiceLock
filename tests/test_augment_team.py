import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from eval import augment_team as at


def _voice(seconds: float, hz: float, amp: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * at.SR)) / at.SR
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(np.float64)
    x = amp * env * np.sin(2 * np.pi * hz * t) + rng.normal(0, 0.003, len(t))
    return x.astype(np.float32)


def test_cap_peak_never_raises_and_caps():
    quiet = np.full(10, 0.1, dtype=np.float32)
    y, g = at.cap_peak(quiet)
    assert g == 0.0 and np.allclose(y, quiet)
    loud = np.full(10, 0.99, dtype=np.float32)
    y, g = at.cap_peak(loud)
    assert at.peak_dbfs(y) == pytest.approx(at.PEAK_CAP_DBFS, abs=1e-3) and g < 0


def test_segment_merges_short_tail():
    x = np.zeros(round(23 * at.SR))
    assert at.segment(x) == [(0, 10 * at.SR), (10 * at.SR, 23 * at.SR)]
    assert at.segment(np.zeros(round(25 * at.SR)))[-1] == (20 * at.SR, 25 * at.SR)


@pytest.mark.parametrize("rt60", [0.2, 0.5, 0.8])
def test_synthetic_rir_decays_60db_at_rt60(rt60):
    h = at.synthetic_rir(rt60, seed=1)
    energy = np.cumsum(h[::-1] ** 2)[::-1]  # Schroeder backward integral
    edc = 10 * np.log10(energy / energy[0])
    # -60 dB point of the tail (fit -5..-25 dB, T20 style) within 20%
    i5, i25 = np.argmax(edc < -5), np.argmax(edc < -25)
    est = 3 * (i25 - i5) / at.SR
    assert est == pytest.approx(rt60, rel=0.2)


def test_render_layout_manifest_and_peaks(tmp_path):
    src = tmp_path / "team"
    for spk, hz, amp in (("aryan", 130, 0.95), ("joseph", 110, 0.3)):
        (src / spk).mkdir(parents=True)
        sf.write(src / spk / "clip.wav", _voice(5.0, hz, amp, seed=hz), at.SR)
    out = tmp_path / "aug"
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    m = at.render(src, out)
    assert set(m["counts"]) == set(at.GROUPS) and all(m["counts"][g] > 0 for g in at.GROUPS)
    assert m["counts"]["pitch_pm2"] == 2 * m["counts"]["clean"]
    assert m["counts"]["combo"] == 1  # ceil(20% of 2 segments)
    for f in m["files"]:
        assert (out / f["path"]).is_file()
        assert f["path"].split("/")[0] == f["augmentation"]
        assert f["peak_dbfs"] < at.CLIP_LIMIT_DBFS
        assert {"speaker", "augmentation", "params", "source_file", "sha256"} <= set(f)
    rows = at.spot_check(json.loads((out / "manifest.json").read_text()), out)
    assert rows and all(r["ok"] for r in rows)
