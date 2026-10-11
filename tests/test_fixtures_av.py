"""AV fixture bank: video I/O and frame maths on synthetic frames (offline), GRID end to end (network)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from eval import make_fixtures
from eval.fixtures import grid as fxgrid
from eval.fixtures import video as fxvideo
from eval.fixtures.mix import PEAK
from eval.fixtures.sources import TinySource, Utterance
from hearing.audio import io as aio

REPO = Path(__file__).resolve().parents[1]
STEMS = ("mix", "target", "interferer", "noise")
needs_ffmpeg = pytest.mark.skipif(not fxvideo.available(), reason="no ffmpeg (install the eval extra)")


def _clip(n: int, w: int, h: int, seed: int = 0) -> np.ndarray:
    """Smooth gradient frames (seed shifts the colours) with a moving bright square."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 120 // w, yy * 120 // h, np.full_like(xx, 20 * (seed % 5))], axis=-1).astype(np.uint8)
    frames = np.repeat(base[None], n, axis=0)
    for k in range(n):
        x = (k * 7) % max(1, w - 16)
        frames[k, h // 3 : h // 3 + 16, x : x + 16] = 230
    return frames


def _solid(n: int, w: int, h: int, rgb: tuple[int, int, int]) -> np.ndarray:
    return np.broadcast_to(np.array(rgb, np.uint8), (n, h, w, 3)).copy()


# --- frame maths ------------------------------------------------------------------------------


def test_fit_size_keeps_aspect_never_upscales_even_dims():
    assert fxvideo.fit_size(1920, 1080) == (640, 360)
    assert fxvideo.fit_size(1280, 960) == (480, 360)
    assert fxvideo.fit_size(360, 288) == (360, 288)  # GRID: already fits, untouched
    w, h = fxvideo.fit_size(1001, 563)
    assert w <= 640 and h <= 360 and w % 2 == 0 and h % 2 == 0


def test_resample_fps_30_to_25_count_and_order():
    frames = np.arange(60, dtype=np.uint8)[:, None, None, None] * np.ones((1, 2, 2, 3), np.uint8)
    out = fxvideo.resample_fps(frames, 30, 25)
    assert len(out) == 50
    idx = out[:, 0, 0, 0].astype(int)
    assert np.all(np.diff(idx) >= 1) and idx[0] == 0 and idx[-1] <= 59
    assert len(fxvideo.resample_fps(frames[:25], 25, 25)) == 25
    assert len(fxvideo.resample_fps(frames[:50], 50, 25)) == 25
    assert len(fxvideo.resample_fps(frames[:24], 24, 25)) == 25  # upsampling repeats frames


def test_fit_resize_and_resize_frames():
    out = fxvideo.fit_resize(_clip(3, 1280, 720))
    assert out.shape == (3, 360, 640, 3) and out.dtype == np.uint8
    solid = fxvideo.resize_frames(_solid(2, 100, 50, (10, 200, 30)), (37, 21))
    assert solid.shape == (2, 21, 37, 3) and np.all(solid == np.array([10, 200, 30], np.uint8))


def test_side_by_side_is_640x360_left_from_a_right_from_b():
    a = _solid(5, 360, 288, (255, 0, 0))
    b = _solid(5, 360, 288, (0, 0, 255))
    out = fxvideo.side_by_side(a, b)
    assert out.shape == (5, 360, 640, 3) and out.dtype == np.uint8
    assert np.all(out[:, :, :320] == [255, 0, 0])
    assert np.all(out[:, :, 320:] == [0, 0, 255])
    with pytest.raises(ValueError):
        fxvideo.side_by_side(a, b[:4])


def test_side_by_side_left_half_is_the_centre_crop_of_a():
    a = _clip(2, 360, 288, seed=1)
    b = _solid(2, 360, 288, (0, 0, 0))
    out = fxvideo.side_by_side(a, b)
    x, y, w, h = fxvideo.crop_box_to_aspect(360, 288, 320 / 360)
    assert (w, h) == (256, 288)
    ref = fxvideo.resize_frames(a[:, y : y + h, x : x + w], (320, 360))
    np.testing.assert_array_equal(out[:, :, :320], ref)


def test_fit_frames_loops_like_fit_length():
    f = np.arange(4, dtype=np.uint8)[:, None, None, None] * np.ones((1, 1, 1, 3), np.uint8)
    assert fxvideo.fit_frames(f, 10)[:, 0, 0, 0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    assert len(fxvideo.fit_frames(f, 2)) == 2


def test_side_by_side_boxes_map_into_each_half():
    full = (0.0, 0.0, 1.0, 1.0)
    lbox, rbox = fxvideo.side_by_side_boxes(full, (360, 288), full, (360, 288))
    assert lbox == [0.0, 0.0, 0.5, 1.0] and rbox == [0.5, 0.0, 0.5, 1.0]
    # A centred box: the 256-px-wide centre crop of 360 px keeps it centred in its half.
    lbox, rbox = fxvideo.side_by_side_boxes((0.4, 0.25, 0.2, 0.5), (360, 288), (0.4, 0.25, 0.2, 0.5), (360, 288))
    assert abs(lbox[0] + lbox[2] / 2 - 0.25) < 1e-3 and abs(rbox[0] + rbox[2] / 2 - 0.75) < 1e-3
    assert abs(lbox[2] - 0.2 * 360 / 256 * 320 / 640) < 1e-3 and lbox[1] == 0.25 and lbox[3] == 0.5


# --- file I/O (ffmpeg, no network) ------------------------------------------------------------


@needs_ffmpeg
def test_write_read_round_trip_keeps_frames_and_size(tmp_path: Path):
    frames = _clip(50, 640, 360)
    path = fxvideo.write_video(tmp_path / "a.mp4", frames, 25)
    out, fps = fxvideo.read_video(path)
    assert fps == 25.0
    assert abs(len(out) - 50) <= 1
    assert out.dtype == np.uint8 and out.shape[1:] == (360, 640, 3)
    assert np.abs(out[: min(len(out), 50)].astype(int) - frames[: len(out)]).mean() < 8  # lossy, but close


@needs_ffmpeg
def test_read_video_fits_large_30fps_input_to_640x360_at_25(tmp_path: Path):
    path = fxvideo.write_video(tmp_path / "big.mp4", _clip(60, 960, 540), 30)
    info = fxvideo.probe(path)
    assert (info.width, info.height, info.fps) == (960, 540, 30.0) and not info.has_audio
    out, fps = fxvideo.read_video(path, fps=25)
    assert fps == 25.0 and abs(len(out) - 50) <= 1
    assert out.shape[2] <= 640 and out.shape[1] <= 360 and out.shape[1:3] == (360, 640)


@needs_ffmpeg
def test_write_video_muxes_audio_and_read_audio_decodes_it(tmp_path: Path):
    t = np.arange(32000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = tmp_path / "tone.wav"
    aio.save(wav, tone, 16000)
    path = fxvideo.write_video(tmp_path / "v.mp4", _clip(50, 320, 180), 25, audio_wav=wav)
    assert fxvideo.probe(path).has_audio
    audio = fxvideo.read_audio(path)
    assert audio.dtype == np.float32 and abs(len(audio) - 32000) < 2048
    remux = fxvideo.mux_audio(fxvideo.write_video(tmp_path / "s.mp4", _clip(50, 320, 180), 25), wav, tmp_path / "m.mp4")
    assert fxvideo.probe(remux).has_audio and fxvideo.read_video(remux)[0].shape[0] in (49, 50, 51)


# --- GRID alignments (offline) ----------------------------------------------------------------

ALIGN = "0 23750 sil\n23750 29500 bin\n29500 34000 blue\n34000 35500 at\n35500 41000 f\n41000 47250 two\n47250 53000 now\n53000 74500 sil\n"


def test_parse_align_transcript_and_intervals():
    words = fxgrid.parse_align(ALIGN)
    assert len(words) == 8 and words[1] == fxgrid.Word(0.95, 1.18, "bin")
    assert fxgrid.transcript_of(words) == "bin blue at f two now"
    assert fxgrid.speech_intervals(words) == [[0.95, 2.12]]
    assert fxgrid.speech_intervals(words, duration_s=1.0) == [[0.95, 1.0]]
    gap = fxgrid.parse_align("0 100 sil\n100 25000 a\n25000 30000 sp\n30000 50000 b\n")
    assert fxgrid.speech_intervals(gap) == [[0.004, 1.0], [1.2, 2.0]]


def test_loop_intervals_tiles_and_clips():
    assert fxgrid.loop_intervals([[0.5, 1.5]], 2.0, 3.0) == [[0.5, 1.5], [2.5, 3.0]]
    assert fxgrid.loop_intervals([[0.5, 1.5]], 3.0, 3.0) == [[0.5, 1.5]]
    with pytest.raises(ValueError):
        fxgrid.loop_intervals([[0, 1]], 0, 1)


def test_grid_utterance_unpacks_to_path_audio_transcript():
    u = fxgrid.GridUtterance("s1", "bbaf2n", "M", Path("x.mpg"), np.zeros(8, np.float32), fxgrid.parse_align(ALIGN))
    path, audio, transcript = u
    assert path == Path("x.mpg") and audio.dtype == np.float32 and transcript == "bin blue at f two now"
    assert u.id == "grid-s1-bbaf2n"


# --- AV set assembly on synthetic clips (offline) ---------------------------------------------


def _synthetic_clip(speech: Utterance, n_frames: int, w: int, h: int, seed: int, actor: str) -> make_fixtures.AVClip:
    n = n_frames * make_fixtures.SAMPLES_PER_FRAME
    utt = Utterance(id=f"syn-{actor}", speaker_id=actor, text=speech.text, audio=make_fixtures._pad_or_trim(speech.audio, n),
                    sex=speech.sex, source={"dataset": "synthetic", "split": ""})
    return make_fixtures.AVClip(utt, _clip(n_frames, w, h, seed), [[0.2, 1.4]], (0.3, 0.2, 0.4, 0.6))


@pytest.fixture(scope="module")
def synthetic_av_set(tmp_path_factory) -> tuple[Path, list]:
    if not fxvideo.available():
        pytest.skip("no ffmpeg")
    out = tmp_path_factory.mktemp("avfx")
    src = TinySource()
    a, b = src.speech()
    # Shorter clips than the real 3 s ones keep this fast; the interferer is shorter so it loops.
    target = _synthetic_clip(a, 40, 360, 288, 1, "spk-a")
    inter = _synthetic_clip(b, 30, 360, 288, 2, "spk-b")
    manifests = make_fixtures.make_av_set(
        out, "av_syn", [(target, inter)], [src.noise()], [(0.0, 0.0), (5.0, 5.0)], seed=0
    )
    return out / "av_syn", manifests


def test_synthetic_av_set_files_manifest_and_levels(synthetic_av_set):
    root, manifests = synthetic_av_set
    assert len(manifests) == 4  # 1 pair x 2 layouts x 2 conditions
    index = json.loads((root / "index.json").read_text())
    assert index["count"] == 4 and [e["layout"] for e in index["fixtures"]] == ["single", "single", "two_face", "two_face"]
    for e in index["fixtures"]:
        d = root / e["id"]
        for f in ("video.mp4", "manifest.json", *(f"{s}.wav" for s in STEMS)):
            assert (d / f).exists(), f
        m, stems = make_fixtures.load_av_fixture(d)
        assert m.fps == 25.0 and m.n_frames == 40 and abs(m.duration_s - 40 / 25) < 1e-9
        assert len(stems.mix) == 40 * 640
        assert abs(stems.measured_snr_db() - m.snr_db) < 0.1 and abs(stems.measured_sir_db() - m.sir_db) < 0.1
        total = stems.target.astype(np.float64) + stems.interferer + stems.noise
        assert np.max(np.abs(total - stems.mix)) < 1e-4 and np.max(np.abs(stems.mix)) <= PEAK + 1e-4
        frames, fps = fxvideo.read_video(d / m.video)
        assert fps == 25.0 and abs(len(frames) - m.n_frames) <= 1
        assert list(frames.shape[2:0:-1]) == m.frame_size and m.frame_size[0] <= 640 and m.frame_size[1] <= 360
        assert fxvideo.probe(d / m.video).has_audio
        assert m.target_intervals == [[0.2, 1.4]]
        # interferer is 30 frames (1.2 s) looped over 1.6 s
        assert m.interferer_intervals == [[0.2, 1.2], [1.4, 1.6]]
        for face in m.faces:
            x, y, w, h = face["box"]
            assert 0 <= x and 0 <= y and x + w <= 1 + 1e-6 and y + h <= 1 + 1e-6 and w > 0 and h > 0


def test_synthetic_av_layouts_faces_and_sides(synthetic_av_set):
    _, manifests = synthetic_av_set
    single = [m for m in manifests if m.layout == "single"]
    two = [m for m in manifests if m.layout == "two_face"]
    for m in single:
        assert m.target_side == "center" and not m.interferer_visible and m.frame_size == [360, 288]
        assert m.faces == [{"actor": "spk-a", "role": "target", "side": "center", "box": [0.3, 0.2, 0.4, 0.6]}]
    for m in two:
        assert m.target_side == "left" and m.interferer_visible and m.frame_size == [640, 360]
        assert [(f["actor"], f["role"], f["side"]) for f in m.faces] == [
            ("spk-a", "target", "left"),
            ("spk-b", "interferer", "right"),
        ]
        assert m.faces[0]["box"][0] + m.faces[0]["box"][2] <= 0.5 and m.faces[1]["box"][0] >= 0.5


def test_two_face_side_pattern_covers_both_sides():
    sides = [make_fixtures.two_face_side(p) for p in range(4)]
    assert sides == ["left", "right", "right", "left"]


def test_render_layout_right_side_puts_target_right():
    src = TinySource()
    a, b = src.speech()
    t = _synthetic_clip(a, 10, 360, 288, 1, "t")
    i = _synthetic_clip(b, 10, 360, 288, 2, "i")
    i.frames[:] = 0
    frames, faces = make_fixtures.render_layout("two_face", t, i, "right")
    assert frames.shape == (10, 360, 640, 3)
    assert np.all(frames[:, :, :320] == 0) and frames[:, :, 320:].max() > 0
    assert faces[1]["role"] == "target" and faces[1]["side"] == "right"
    with pytest.raises(ValueError):
        make_fixtures.render_layout("two_face", t, i, "center")


# --- GRID (network) ---------------------------------------------------------------------------


@pytest.mark.network
@pytest.mark.slow
def test_grid_download_and_iter_utterances(tmp_path: Path):
    corpus = fxgrid.GridCorpus(fxgrid.SOURCES_DIR)
    utts = list(fxgrid.iter_utterances("s4", limit=1, corpus=corpus))
    assert len(utts) == 1
    path, audio, transcript = utts[0]
    assert path.exists() and audio.dtype == np.float32 and 2.5 < len(audio) / 16000 < 3.5
    assert len(transcript.split()) == 6
    frames, fps = fxvideo.read_video(path)
    assert fps == 25.0 and frames.shape[1:] == (288, 360, 3) and 70 <= len(frames) <= 76


@pytest.mark.network
@pytest.mark.slow
def test_av_tiny_cli_writes_16_fixtures(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "eval.make_fixtures", "--av", "--tiny", "--out", str(tmp_path)],
        cwd=REPO, capture_output=True, text=True, timeout=900, check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "av_tiny: 16 fixtures" in proc.stdout
    root = tmp_path / "av_tiny"
    index = json.loads((root / "index.json").read_text())
    assert index["count"] == 16
    assert {e["layout"] for e in index["fixtures"]} == {"single", "two_face"}
    assert {e["target_side"] for e in index["fixtures"] if e["layout"] == "two_face"} == {"left", "right"}
    for e in index["fixtures"]:
        m, stems = make_fixtures.load_av_fixture(root / e["id"])
        assert (root / e["id"] / "video.mp4").exists() and m.faces and m.target_side
        assert m.source["target_sex"] != m.source["interferer_sex"]
        assert abs(stems.measured_snr_db() - m.snr_db) < 0.1 and abs(stems.measured_sir_db() - m.sir_db) < 0.1
        total = stems.target.astype(np.float64) + stems.interferer + stems.noise
        assert np.max(np.abs(total - stems.mix)) < 1e-4
