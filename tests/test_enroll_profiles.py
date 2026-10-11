"""T-172: scripts/enroll_profiles.py validates demo-mic takes, writes the profiles and fails on one voice."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from hearing.tse.enroll import load_profile
from scripts import enroll_profiles as ep


def speechy(seconds=30.0, sr=48000, f0=120.0, level=0.3, seed=0):
    """Harmonic 'syllables' (200 ms on, 100 ms off) over a quiet room: voiced like speech for the gate."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * sr)) / sr
    tone = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    on = (t % 0.3) < 0.2
    x = level * tone / np.max(np.abs(tone)) * on + 0.001 * rng.standard_normal(len(t))
    return x.astype(np.float32)


def test_good_take_passes():
    c = ep.check_take(speechy(), 48000)
    assert c.ok, c.errors
    assert c.speech_s >= ep.MIN_SPEECH_S and c.speech_db > ep.MIN_SPEECH_DB


def test_too_short():
    c = ep.check_take(speechy(seconds=8.0), 48000)
    assert any("too short" in e for e in c.errors)


def test_clipped():
    c = ep.check_take(np.clip(speechy(level=3.0), -1, 1), 48000)
    assert any("clipped" in e for e in c.errors)


def test_wrong_rate():
    c = ep.check_take(speechy(sr=8000), 8000)
    assert any("wrong rate" in e for e in c.errors)
    assert ep.check_take(speechy(sr=16000), 16000).ok


def test_too_quiet():
    c = ep.check_take(speechy(level=0.003), 48000)
    assert any("too quiet" in e for e in c.errors)


def test_stereo_is_downmixed():
    x = speechy()
    assert ep.check_take(np.stack([x, x], axis=1), 48000).ok


def test_parse_groups():
    g = ep.parse_groups(["aryan", "a1.wav", "a2.m4a", "joseph", "j.wav"])
    assert [p.name for p in g["aryan"]] == ["a1.wav", "a2.m4a"] and len(g["joseph"]) == 1
    for bad in (["a.wav"], ["aryan"], ["../x", "a.wav"]):
        with pytest.raises(ValueError):
            ep.parse_groups(bad)


class PitchEmbedder:
    """Fake speaker embedding: the dominant pitch band, so two pitches are two voices."""

    def embed(self, audio):
        spec = np.abs(np.fft.rfft(audio, n=4096))
        v = np.zeros(32)
        v[min(31, int(np.argmax(spec[1:400])) // 4)] = 1.0
        return v


def _write_pair(tmp_path, f0_b=220.0):
    a, b = tmp_path / "aryan.wav", tmp_path / "joseph.wav"
    sf.write(a, speechy(f0=120.0), 48000)
    sf.write(b, speechy(f0=f0_b, seed=1), 48000)
    return a, b


def test_enrolls_both_lists_them_in_health_and_prints_gpu_up(tmp_path, monkeypatch, capsys):
    a, b = _write_pair(tmp_path)
    out = tmp_path / "profiles"
    (out / "aryan").mkdir(parents=True)
    sf.write(out / "aryan" / "profile-01.wav", speechy(seconds=12.0, sr=16000, f0=120.0), 16000)  # the old one
    monkeypatch.setattr(ep, "load_embedder", lambda kind: PitchEmbedder())
    argv = ["aryan", str(a), "joseph", str(b), "--out", str(out), "--mic", "MacBook Pro Microphone", "--room", "CoRE 101"]
    assert ep.main(argv) == 0
    text = capsys.readouterr().out
    # the flat old profile became v1, the new takes v2 (current); nothing flat is left beside them
    assert (out / "aryan" / "current").read_text().strip() == "v2"
    assert not list((out / "aryan").glob("*.wav"))
    assert sorted(p.name for p in (out / "aryan" / "v1").glob("*.wav")) == ["profile-01.wav"]
    assert sorted(p.name for p in (out / "aryan" / "v2").glob("*.wav")) == ["aryan.wav"]
    assert json.loads((out / "aryan" / "v1" / "manifest.json").read_text())["migrated"] is True
    m = json.loads((out / "aryan" / "v2" / "manifest.json").read_text())
    assert m["mic"] == "MacBook Pro Microphone" and m["room"] == "CoRE 101" and m["previous"] == "v1"
    assert [s["file"] for s in m["sources"]] == [str(a.resolve())] and m["files"] == ["aryan.wav"] and m["date"]
    assert m["cos"] == {"joseph": 0.0}
    assert (out / "joseph" / "current").read_text().strip() == "v1"
    assert (out / "joseph" / "v1" / "joseph.wav").exists()
    assert (tmp_path / "profiles-all" / "joseph" / "joseph.wav").exists()
    assert "cos(aryan, joseph) = 0.000" in text
    assert f"ILAB_PROFILES={out.resolve()} {ep.GPU_UP}" in text
    assert len(load_profile("aryan", out)) == 12  # the 30 s v2 take in 2.5 s pieces, not the 12 s v1 one

    from hearing.server.app import create_app
    from hearing.server.session import ServerConfig

    monkeypatch.setenv("HEARING_PROFILES_DIR", str(out))
    health = TestClient(create_app(ServerConfig())).get("/health").json()
    assert health["profiles"] == ["aryan", "joseph"]
    assert health["profile_versions"] == {"aryan": "v2", "joseph": "v1"}

    # one-command rollback, then forward again by name
    assert ep.main(["--out", str(out), "--rollback", "aryan"]) == 0
    assert (out / "aryan" / "current").read_text().strip() == "v1"
    assert len(load_profile("aryan", out)) < 12
    assert TestClient(create_app(ServerConfig())).get("/health").json()["profile_versions"]["aryan"] == "v1"
    assert ep.main(["--out", str(out), "--rollback", "aryan"]) == 1  # nothing before v1
    assert ep.main(["--out", str(out), "--rollback", "aryan", "--to", "v9"]) == 1
    assert ep.main(["--out", str(out), "--rollback", "aryan", "--to", "v2"]) == 0
    capsys.readouterr()
    assert ep.main(["--out", str(out), "--versions"]) == 0
    listing = capsys.readouterr().out
    assert "* aryan/v2" in listing and "  aryan/v1" in listing and "mic=MacBook Pro Microphone" in listing


def test_multi_enrollment_aggregates_takes_into_one_version(tmp_path, monkeypatch):
    takes = []
    for i in range(3):
        takes.append(tmp_path / f"aryan-{i}.wav")
        sf.write(takes[-1], speechy(seed=i), 48000)
    out = tmp_path / "profiles"
    monkeypatch.setattr(ep, "load_embedder", lambda kind: PitchEmbedder())
    assert ep.main(["aryan", *map(str, takes), "--out", str(out)]) == 0
    assert ep.main(["aryan", str(takes[0]), "--out", str(out)]) == 0
    assert ep.versions(out / "aryan") == ["v1", "v2"]
    m1 = json.loads((out / "aryan" / "v1" / "manifest.json").read_text())
    assert len(m1["sources"]) == 3 and len(m1["files"]) == 3 and m1["previous"] is None
    assert len(load_profile("aryan", out)) == 12  # current v2: one take
    ep.rollback(out / "aryan")
    assert len(load_profile("aryan", out)) == 36  # v1: all three takes


def test_same_voice_fails_and_replaces_nothing(tmp_path, monkeypatch, capsys):
    a, b = _write_pair(tmp_path, f0_b=121.0)
    out = tmp_path / "profiles"
    monkeypatch.setattr(ep, "load_embedder", lambda kind: PitchEmbedder())
    assert ep.main(["aryan", str(a), "joseph", str(b), "--out", str(out)]) == 3
    assert "FAIL >= 0.7" in capsys.readouterr().out
    assert not out.exists()


def test_bad_take_writes_nothing(tmp_path, capsys):
    a = tmp_path / "a.wav"
    sf.write(a, speechy(seconds=5.0), 48000)
    out = tmp_path / "profiles"
    assert ep.main(["aryan", str(a), "--out", str(out), "--embedder", "off"]) == 2
    assert "too short" in capsys.readouterr().out and not out.exists()
