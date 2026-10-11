"""T-089: the pre-demo headroom check (``scripts/demo.sh --check``), with a mock tier."""

from hearing.fusion import tiers as tiers_mod
from hearing.tse import headroom


def test_check_passes_with_a_fast_tier_and_fails_below_the_minimum(monkeypatch, capsys):
    cfg = tiers_mod.LadderConfig.from_dict({"tiers": [
        {"name": "enrolled", "pipeline": 'tse(candidate="identity", window_s=1.0, hop_s=0.5)>mixer'}]})
    monkeypatch.setattr(tiers_mod.LadderConfig, "load", classmethod(lambda cls, path=None: cfg))
    r = headroom.measure(windows=3)
    assert r["built"] and r["candidate"] == "identity" and 0.9 < r["headroom"] <= 1.0
    assert headroom.main(["--windows", "2"]) == 0
    assert headroom.main(["--windows", "2", "--min", "1.01"]) == 1
    assert "headroom" in capsys.readouterr().out


def test_check_skips_a_tier_that_is_not_built_here(monkeypatch):
    cfg = tiers_mod.LadderConfig.from_dict({"tiers": [
        {"name": "enrolled", "pipeline": "tse(candidate=\"identity\")", "devices": ["cuda"]},
        {"name": "audio-only", "pipeline": "passthrough"}]})
    monkeypatch.setattr(tiers_mod.LadderConfig, "load", classmethod(lambda cls, path=None: cfg))
    monkeypatch.setattr(tiers_mod.hdevice, "pick", lambda: "mps")
    assert headroom.measure()["built"] is False
    assert headroom.main([]) == 0
