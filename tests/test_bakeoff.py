import json
import math
from typing import ClassVar

import numpy as np
import pytest

from eval import bakeoff
from eval.bakeoff_fixtures import (
    load_fixture,
    load_set,
    mouth_box_from_face,
    oracle_p_speaking,
    synthetic_av_set,
)
from eval.transcribe import MockTranscriber
from hearing.audio import io
from hearing.tse import register_candidate

SR = 16000


@register_candidate("test-raises")
class Raises:
    name = "test-raises"
    licence = "test"
    needs: frozenset[str] = frozenset()

    def load(self) -> None:
        pass

    def extract(self, mix, sr, **kw):
        raise RuntimeError("boom")


@register_candidate("test-load-fails")
class LoadFails(Raises):
    name = "test-load-fails"

    def load(self) -> None:
        raise ImportError("no weights")


@register_candidate("test-av-recorder")
class AVRecorder:
    """Identity that asserts it receives exactly the inputs it declares."""

    name = "test-av-recorder"
    licence = "test"
    needs = frozenset({"lips", "p_speaking"})
    calls: ClassVar[list] = []

    def load(self) -> None:
        pass

    def extract(self, mix, sr, *, lips, lips_fps, p_speaking, enroll):
        AVRecorder.calls.append(
            (
                len(mix),
                None if lips is None else lips.shape,
                None if p_speaking is None else p_speaking.shape,
                enroll,
            )
        )
        return mix


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return synthetic_av_set(
        n=2, seed=0, out=tmp_path_factory.mktemp("bakeoff_tiny"), duration_s=3.0
    )


def test_synthetic_set_layout_and_lips(tiny):
    dirs = sorted(p for p in tiny.iterdir() if p.is_dir())
    assert [d.name for d in dirs] == ["syn00", "syn01"]
    for d in dirs:
        for f in (
            "mix.wav",
            "target.wav",
            "interferer.wav",
            "noise.wav",
            "manifest.json",
            "lips.npy",
        ):
            assert (d / f).exists()
    fx = load_fixture(dirs[0])
    assert fx.sr == SR and fx.mix.shape == fx.target.shape == (3 * SR,)
    assert (
        fx.lips.shape == (75, 88, 88)
        and fx.lips.dtype == np.uint8
        and fx.lips_source == "lips.npy"
    )
    assert (
        fx.p_speaking.shape == (75,)
        and 0 <= fx.p_speaking.min()
        and fx.p_speaking.max() <= 1
    )
    # Mouth opening (dark pixels) follows target energy.
    opening = (fx.lips < 100).sum(axis=(1, 2))
    assert np.corrcoef(opening, fx.p_speaking)[0, 1] > 0.5
    assert fx.enroll_source == "target[:3s]" and np.array_equal(
        fx.enroll, fx.target[: 3 * SR]
    )
    assert isinstance(fx.transcript, str) and len(fx.transcript.split()) == 6
    mix = io.load(dirs[0] / "mix.wav")[0]
    parts = sum(
        io.load(dirs[0] / f"{k}.wav")[0] for k in ("target", "interferer", "noise")
    )
    assert (
        np.max(np.abs(mix - parts)) < 1e-3
    )  # mix = target + interferer + noise (PCM16 rounding)


def test_synthetic_set_is_cached_and_regenerated_on_change(tmp_path):
    root = synthetic_av_set(n=2, out=tmp_path, duration_s=1.0)
    mtime = (root / "syn00" / "mix.wav").stat().st_mtime_ns
    assert synthetic_av_set(n=2, out=tmp_path, duration_s=1.0) == root
    assert (root / "syn00" / "mix.wav").stat().st_mtime_ns == mtime
    synthetic_av_set(n=1, out=tmp_path, duration_s=1.0)
    assert len(load_set(root)) == 1


def test_enroll_wav_overrides_fallback(tiny, tmp_path):
    src = tiny / "syn00"
    d = tmp_path / "fx"
    d.mkdir()
    for f in ("mix.wav", "target.wav"):
        (d / f).write_bytes((src / f).read_bytes())
    io.save(d / "enroll.wav", np.full(SR, 0.25, np.float32), SR)
    manifest = json.loads((src / "manifest.json").read_text()) | {
        "enroll_wav": "enroll.wav"
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    fx = load_fixture(d)
    assert fx.enroll_source == "enroll_wav" and fx.enroll.shape == (SR,)
    assert fx.lips is None and fx.lips_source is None  # no lips.npy, no video
    assert load_set(d)[0].id == fx.id


def test_lips_from_video_with_target_box(tiny, tmp_path):
    cv2 = pytest.importorskip("cv2")
    src = tiny / "syn00"
    d = tmp_path / "av"
    d.mkdir()
    for f in ("mix.wav", "target.wav"):
        (d / f).write_bytes((src / f).read_bytes())
    face = [0.25, 0.1, 0.5, 0.8]  # normalised x, y, w, h
    mx, my, mw, mh = mouth_box_from_face(face)
    w, h = 160, 120
    vw = cv2.VideoWriter(
        str(d / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h)
    )
    if not vw.isOpened():
        pytest.skip("no mp4 encoder")
    for _ in range(50):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[
            int(my * h) + 2 : int((my + mh) * h) - 2,
            int(mx * w) + 2 : int((mx + mw) * w) - 2,
        ] = 255
        vw.write(frame)
    vw.release()
    manifest = json.loads((src / "manifest.json").read_text()) | {"target_box": face}
    (d / "manifest.json").write_text(json.dumps(manifest))
    fx = load_fixture(d)
    assert fx.lips_source == "video.mp4@box"
    assert fx.lips.shape == (50, 88, 88) and fx.lips.dtype == np.uint8
    assert fx.lips[:, 20:68, 20:68].mean() > 200  # the crop is the bright mouth region


def test_oracle_p_speaking_tracks_energy():
    x = np.zeros(SR, np.float32)
    x[SR // 2 :] = 0.5 * np.sin(2 * np.pi * 200 * np.arange(SR // 2) / SR)
    p = oracle_p_speaking(x, SR, 25.0)
    assert p.shape == (25,)
    assert p[:12].max() < 0.01 and p[13:].min() > 0.99


def _run(tiny, names, windows=(0.5, 1.0)):
    fixtures = load_set(tiny)
    rows = []
    for n in names:
        rows += bakeoff.run_candidate(n, fixtures, list(windows), MockTranscriber())
    return rows


def test_raising_candidate_becomes_error_row(tiny):
    rows = _run(tiny, ["test-raises", "identity"])
    assert [r["candidate"] for r in rows] == ["test-raises"] * 2 + ["identity"] * 2
    bad = rows[0]
    assert (
        "RuntimeError: boom" in bad["error"]
        and bad["n_fixtures"] == 0
        and math.isnan(bad["si_sdri_db"])
    )
    assert rows[2]["error"] is None and rows[2]["si_sdri_db"] == pytest.approx(
        0.0, abs=1e-3
    )
    assert bakeoff.rank(rows)[-1]["error"] is not None


def test_load_failure_and_unknown_candidate_become_error_rows(tiny):
    rows = _run(tiny, ["test-load-fails", "does-not-exist"], windows=(1.0,))
    assert rows[0]["error"].startswith("load: ImportError")
    assert rows[1]["error"].startswith("unknown: KeyError")


def test_candidate_gets_only_declared_inputs_aligned(tiny):
    AVRecorder.calls.clear()
    rows = _run(tiny, ["test-av-recorder"], windows=(1.0,))
    assert rows[0]["error"] is None
    assert (
        len(AVRecorder.calls) == 2 * 5
    )  # 2 fixtures x 5 windows of 1 s over 3 s, hop 0.5 s
    for n, lips, p, enroll in AVRecorder.calls:
        assert n == SR and lips == (25, 88, 88) and p == (25,) and enroll is None


def test_cli_six_rows_json_and_summary(tiny, tmp_path, capsys):
    summary = tmp_path / "summary.md"
    rc = bakeoff.main(
        [
            "--set", str(tiny), "--candidates", "identity,oracle", "--windows", "0.5,1,2",
            "--transcriber", "mock", "--out", str(tmp_path), "--summary", str(summary),
        ]
    )  # fmt: skip
    assert rc == 0
    out = capsys.readouterr().out
    table = out.split("\n\n")[0].splitlines()
    assert table[0].split()[:3] == ["candidate", "win_s", "SI-SDRi_dB"]
    for col in ("WER", "RTF", "algo_ms", "comp_p95_ms", "gpu_mem_MB", "device"):
        assert col in table[0]
    assert len(table) == 2 + 6
    (path,) = tmp_path.glob("bakeoff_*.json")
    result = json.loads(path.read_text())
    assert len(result["rows"]) == 6 and result["transcriber"] == "mock"
    for r in result["rows"]:
        assert r["error"] is None and r["device"] and r["rtf"] is not None
        assert r["algorithmic_latency_ms"] == r["window_s"] * 1000
        assert r["compute_p50_ms"] <= r["compute_p95_ms"]
    assert [x.split("@")[0] for x in result["ranking"]] == ["oracle"] * 3 + [
        "identity"
    ] * 3
    md = summary.read_text()
    assert md.index("| oracle |") < md.index("| identity |")
    oracle_sdr = min(
        r["si_sdri_db"] for r in result["rows"] if r["candidate"] == "oracle"
    )
    identity_sdr = max(
        r["si_sdri_db"] for r in result["rows"] if r["candidate"] == "identity"
    )
    assert oracle_sdr > identity_sdr + 10


def test_gpu_memory_is_none_on_cpu():
    assert bakeoff.gpu_memory_mb("cpu") is None
