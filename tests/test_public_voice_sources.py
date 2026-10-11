"""Offline tests for the public-voice kit: splits, checksums, caps and the committed manifest."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from eval import public_voice_sources as pvs

MANIFEST = (
    Path(__file__).resolve().parents[1] / "docs/research/public_voice_manifest.json"
)


def _utts(n, secs=8.0):
    return [
        {"speaker": "1", "utt_id": f"1-2-{i:04d}", "seconds": secs, "sha256": str(i)}
        for i in range(n)
    ]


def test_split_is_ordered_budgeted_and_leaves_eval():
    out = pvs.split_enrollment(_utts(6), enroll_seconds=20.0)
    assert [u["role"] for u in out] == ["enroll"] * 3 + ["eval"] * 3
    one = pvs.split_enrollment(_utts(1), 20.0)
    assert one[0]["role"] == "eval"  # never all-enrollment
    short = pvs.split_enrollment(_utts(2, secs=30.0), 20.0)
    assert [u["role"] for u in short] == ["enroll", "eval"]


def test_check_disjoint_rejects_shared_recording():
    ok = pvs.split_enrollment(_utts(4))
    pvs.check_disjoint(ok)
    bad = [{**ok[0], "role": "enroll"}, {**ok[0], "role": "eval"}]
    with pytest.raises(ValueError, match="overlap"):
        pvs.check_disjoint(bad)


def _src(tmp_path, payload: bytes, md5=None):
    f = tmp_path / "remote.bin"
    f.write_bytes(payload)
    return {
        "file": "x.bin",
        "url": f.as_uri(),
        "md5": md5 or hashlib.md5(payload).hexdigest(),
        "size": len(payload),
        "license": "CC BY 4.0",
    }


def test_download_verifies_checksum_and_cap(tmp_path):
    payload = b"abc" * 1000
    ev = pvs.download(_src(tmp_path, payload), tmp_path / "out", cap_bytes=10_000)
    assert (
        ev["bytes"] == len(payload)
        and ev["sha256"] == hashlib.sha256(payload).hexdigest()
    )
    with pytest.raises(RuntimeError, match="cap"):
        pvs.download(_src(tmp_path, payload), tmp_path / "out2", cap_bytes=100)
    with pytest.raises(RuntimeError, match="md5 mismatch"):
        pvs.download(_src(tmp_path, payload, md5="0" * 32), tmp_path / "out3")


def test_librispeech_entries_from_tiny_tar(tmp_path):
    root = tmp_path / "LibriSpeech/test-clean"
    for spk in ("11", "22", "33"):
        d = root / spk / "1"
        d.mkdir(parents=True)
        lines = []
        for i in range(4):
            utt = f"{spk}-1-{i:04d}"
            sig = (
                0.1
                * np.sin(
                    np.arange(16000 * 10)
                    * (0.01 + i * spk.__len__() * 0.001 + int(spk) * 1e-4)
                )
            ).astype("float32")
            sf.write(d / f"{utt}.flac", sig, 16000)
            lines.append(f"{utt} HELLO {i}")
        (d / f"{spk}-1.trans.txt").write_text("\n".join(lines))
    tar = tmp_path / "t.tar.gz"
    with tarfile.open(tar, "w:gz") as tf:
        tf.add(tmp_path / "LibriSpeech", arcname="LibriSpeech")
    ents = pvs.librispeech_entries(tar, ("11", "22"))
    assert {e["speaker"] for e in ents} == {"11", "22"}  # speaker 33 excluded
    assert len(ents) == 8 and all(e["transcript"].startswith("HELLO") for e in ents)
    assert {e["role"] for e in ents} == {"enroll", "eval"}
    assert all(abs(e["seconds"] - 10.0) < 0.01 for e in ents)


def test_committed_manifest_is_consistent():
    m = json.loads(MANIFEST.read_text())
    for sid, src in pvs.SOURCES.items():
        rec = m["sources"][sid]
        assert rec["md5"] == src["md5"] and rec["license"] == "CC BY 4.0"
        assert (
            rec["url"].startswith("https://")
            and "token" not in rec["url"]
            and "expires" not in rec["url"]
        )
    ents = m["librispeech_test_clean"]
    assert {e["speaker"] for e in ents} == set(pvs.DEFAULT_SPEAKERS)
    pvs.check_disjoint(ents)
    assert len({e["sha256"] for e in ents}) == len(ents)
    assert all(e["transcript"] for e in ents)
    assert sum(s["size"] for s in pvs.SOURCES.values()) < pvs.CAP_BYTES
