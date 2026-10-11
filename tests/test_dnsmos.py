from pathlib import Path

import numpy as np
import pytest

from eval import dnsmos
from hearing.audio import io as aio

TINY = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "tiny"


def test_segments_tile_short_clips_and_hop_one_second():
    short = np.ones(16000, np.float32)  # 1 s: doubled until >= 9.01 s (16 s), floor(16 - 9.01) + 1 = 7
    assert dnsmos.segments(short).shape == (7, 144160)
    long = np.ones(16000 * 12, np.float32)  # 12 s: floor(12 - 9.01) + 1 = 3 segments
    segs = dnsmos.segments(long)
    assert segs.shape == (3, 144160) and segs.dtype == np.float32
    with pytest.raises(ValueError):
        dnsmos.segments(np.zeros(0, np.float32))


def test_mock_is_deterministic_and_flagged():
    x = 0.1 * np.sin(np.linspace(0, 2000, 48000)).astype(np.float32)
    a, b = dnsmos.MockDnsmos().score(x, 16000), dnsmos.MockDnsmos().score(x, 16000)
    assert a == b
    assert a.scorer == "mock" and 1.0 <= a.ovrl <= 5.0
    assert a.duration_s == pytest.approx(3.0)
    louder = dnsmos.MockDnsmos().score(4 * x, 16000)
    assert louder.ovrl > a.ovrl


def test_mock_resamples_and_downmixes():
    stereo = np.zeros((48000, 2), np.float32)
    stereo[:, 0] = 0.1
    s = dnsmos.MockDnsmos().score(stereo, 48000)
    assert s.duration_s == pytest.approx(1.0)


def test_get_dnsmos_names(monkeypatch):
    assert dnsmos.get_dnsmos("mock").name == "mock"
    monkeypatch.setattr(dnsmos, "available", lambda: False)
    assert dnsmos.get_dnsmos("auto").name == "mock"
    with pytest.raises(RuntimeError):
        dnsmos.get_dnsmos("dnsmos")
    with pytest.raises(ValueError):
        dnsmos.get_dnsmos("nope")


def test_cli_scores_clips_through_pipelines(tmp_path, capsys):
    x, sr = aio.load(TINY / "speech_a.wav")
    aio.save(tmp_path / "a.wav", x, sr)
    rows = dnsmos.score_clips([tmp_path / "a.wav"], ["input", "gain(db=-6)"], dnsmos.MockDnsmos())
    assert [r["stages"] for r in rows] == ["input", "gain(db=-6)"]
    assert rows[1]["ovrl"] < rows[0]["ovrl"]  # mock tracks loudness
    assert dnsmos.main([str(tmp_path), "--scorer", "mock", "--stages", "passthrough"]) == 0
    out = capsys.readouterr().out
    assert "| a.wav | input |" in out and "| a.wav | passthrough |" in out
    assert dnsmos.main([str(tmp_path / "missing"), "--scorer", "mock"]) == 2


@pytest.mark.slow
def test_onnx_matches_the_reference_implementation():
    """Numbers from Microsoft's dnsmos_local.py (commit 591184a9) on the committed tiny clip."""
    if not dnsmos.available():
        pytest.skip("DNSMOS models not provisioned (python -m eval.dnsmos --download)")
    x, sr = aio.load(TINY / "speech_a.wav")
    s = dnsmos.OnnxDnsmos().score(x, sr)
    assert (s.sig, s.bak, s.ovrl, s.p808) == pytest.approx((3.589, 3.629, 3.081, 3.447), abs=2e-3)
    assert s.segments == 3 and s.scorer == "dnsmos"
