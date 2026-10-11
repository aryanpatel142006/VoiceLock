"""Offline tests for the overnight speech collector (no network)."""

import io
import tarfile

import numpy as np
import pytest
import soundfile as sf

from eval import overnight_speech_collect as c


def _flac(seconds=3.0, sr=16000):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr), dtype="float32"), sr, format="FLAC")
    return buf.getvalue()


def _tar_gz(speakers):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for spk in speakers:
            for chap in ("1", "2"):
                trans = []
                for i in range(5):
                    uid = f"{spk}-{chap}-{i:04d}"
                    raw = _flac()
                    ti = tarfile.TarInfo(f"LibriSpeech/dev-clean/{spk}/{chap}/{uid}.flac")
                    ti.size = len(raw)
                    tf.addfile(ti, io.BytesIO(raw))
                    trans.append(f"{uid} HELLO WORLD {i}")
                t = "\n".join(trans).encode()
                ti = tarfile.TarInfo(f"LibriSpeech/dev-clean/{spk}/{chap}/{spk}-{chap}.trans.txt")
                ti.size = len(t)
                tf.addfile(ti, io.BytesIO(t))
    return buf.getvalue()


def test_counting_reader_refuses_past_limit():
    r = c.CountingReader(io.BytesIO(b"x" * 100), 40)
    assert len(r.read(30)) == 30
    assert len(r.read(30)) == 10
    with pytest.raises(c.BudgetExceeded):
        r.read(1)


def test_stream_speakers_stops_after_n_and_yields_complete_speakers():
    data = _tar_gz(["10", "20", "30"])
    got = list(c.stream_speakers(io.BytesIO(data), 2))
    assert [g[0] for g in got] == ["10", "20"]
    assert len(got[0][1]["1"]["flac"]) == 5 and "10-1-0000" in got[0][1]["1"]["trans"]


def test_roles_are_disjoint_whole_utterances_and_cover_all_three():
    chapters = next(iter(c.stream_speakers(io.BytesIO(_tar_gz(["10", "20"])), 1)))[1]
    picked = c.pick_roles(c.candidates(chapters))
    assert len(picked) == c.PER_SPEAKER
    assert len({u["id"] for u in picked}) == c.PER_SPEAKER
    assert {u["role"] for u in picked} == {"enrollment", "development", "holdout"}
    assert sum(u["role"] == "enrollment" for u in picked) == 3


def test_candidates_drop_missing_transcript_and_bad_duration():
    flacs = {"1-1-0": _flac(3), "1-1-1": _flac(0.5), "1-1-2": _flac(3)}
    got = c.candidates({"1": {"flac": flacs, "trans": {"1-1-0": "A", "1-1-1": "B"}}})
    assert [u["id"] for u in got] == ["1-1-0"]


def test_write_clip_keeps_native_and_derives_without_resampling(tmp_path):
    u = {"id": "1-1-0", "chapter": "1", "role": "holdout", "text": "Don't STOP", **c.clip_info(_flac()),
         "raw": _flac()}
    e = c._write_clip(tmp_path, "1", "F", u)
    assert e["native"]["sample_rate"] == 16000 and e["transcript_normalized"] == "dont stop"
    assert (tmp_path / e["derived_16k_mono"]["path"]).exists()
    u48 = {**u, "raw": _flac(sr=48000)}
    u48.update(c.clip_info(u48["raw"]))
    assert c._write_clip(tmp_path, "1", "F", u48)["derived_16k_mono"] is None


def test_validate_flags_short_speaker():
    e = [{"id": "a", "speaker_id": "1", "transcript_normalized": "x", "role": "holdout"}]
    assert any("< 8" in m for m in c.validate(e))
