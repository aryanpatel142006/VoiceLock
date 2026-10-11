"""Build reproducible two-talker-plus-noise fixtures with clean stems and manifests.

    .venv/bin/python -m eval.make_fixtures --tiny            # offline, tiny (16) + tiny-noise (16), < 10 s
    .venv/bin/python -m eval.make_fixtures --tiny-noise      # offline, only the noise-only tiny set
    .venv/bin/python -m eval.make_fixtures --full            # streams LibriSpeech, full (96) + full-noise (96)
    .venv/bin/python -m eval.make_fixtures --tiny --out /tmp/fx --seed 1
    .venv/bin/python -m eval.make_fixtures --av --tiny       # network once (GRID, ~22 MB), 16 AV fixtures
    .venv/bin/python -m eval.make_fixtures --av --full       # 10 GRID pairs x 2 layouts x 8 conditions

Output: ``<out>/<set>/<id>/{mix,target,interferer,noise}.wav + manifest.json`` and
``<out>/<set>/index.json``; AV sets (``av_tiny``, ``av_full``) add ``video.mp4``. Layout,
manifest fields and licences: ``eval/fixtures/README.md``.

Noise-only sets (``tiny-noise``, ``full-noise``) are target + noise with no competing talker
(all-zero interferer stem, ``sir_db`` null, ``interferer_id`` ""), each target under both
babble and pink noise at every SNR: the enhancer benchmark, since denoisers cannot remove a
second talker (that is the TSE / gate tiers' job).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from eval.fixtures import grid as fxgrid
from eval.fixtures import video as fxvideo
from eval.fixtures.mix import (
    STEM_NAMES,
    Manifest,
    Stems,
    make_mixture,
    write_fixture,
    write_index,
)
from eval.fixtures.sources import (
    BABBLE_TALKERS,
    TINY_DIR,
    LibriSpeechSource,
    TinySource,
    Utterance,
    make_babble,
    pick_babble_talkers,
    pick_pairs,
    pink_noise,
)
from hearing.audio import io as aio

DEFAULT_OUT = Path(os.environ["HEARING_FIXTURES_DIR"]) if os.environ.get("HEARING_FIXTURES_DIR") else Path(__file__).resolve().parent / "fixtures" / "generated"
SNR_GRID_DB = (-5.0, 0.0, 5.0, 10.0)
SIR_GRID_DB = (0.0, 5.0)
FULL_PAIRS = 12
FULL_MIN_S, FULL_MAX_S = 4.0, 8.0
FULL_NOISE_TYPES = ("babble", "pink")
TINY_CLIP_S = 3.0
STEM_FILES = {"mix": "mix.wav", "target": "target.wav", "interferer": "interferer.wav", "noise": "noise.wav"}
NOISE_ONLY_SETS = {"tiny": "tiny-noise", "full": "full-noise"}
NOISE_ONLY_TYPES = ("babble", "pink")


def _fmt_db(v: float) -> str:
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


def fixture_id(idx: int, snr_db: float, sir_db: float | None, noise_type: str) -> str:
    """``003_snr0_sir5_babble``; noise-only fixtures (``sir_db`` None) drop the sir part."""
    sir = "" if sir_db is None else f"_sir{_fmt_db(sir_db)}"
    return f"{idx:03d}_snr{_fmt_db(snr_db)}{sir}_{noise_type}"


def build_fixture(
    idx: int,
    set_name: str,
    target: Utterance,
    interferer: Utterance | None,
    noise: np.ndarray,
    noise_type: str,
    noise_source: dict,
    snr_db: float,
    sir_db: float | None,
    seed: int,
) -> tuple[Manifest, Stems]:
    """One fixture; ``interferer=None`` builds a noise-only one (``sir_db`` forced to None)."""
    if interferer is None:
        sir_db = None
    stems = make_mixture(target.audio, None if interferer is None else interferer.audio, noise, snr_db, sir_db)
    fid = fixture_id(idx, snr_db, sir_db, noise_type)
    manifest = Manifest(
        id=fid,
        set=set_name,
        sample_rate=aio.PIPELINE_SR,
        duration_s=round(len(stems.mix) / aio.PIPELINE_SR, 4),
        snr_db=float(snr_db),
        sir_db=None if sir_db is None else float(sir_db),
        target_id=target.id,
        interferer_id="" if interferer is None else interferer.id,
        noise_type=noise_type,
        transcript=target.transcript,
        paths=dict(STEM_FILES),
        seed=seed,
        source={
            "dataset": target.source.get("dataset", ""),
            "split": target.source.get("split", ""),
            "target": target.id,
            "target_speaker": target.speaker_id,
            "target_sex": target.sex,
            "interferer": "" if interferer is None else interferer.id,
            "interferer_speaker": "" if interferer is None else interferer.speaker_id,
            "interferer_sex": "" if interferer is None else interferer.sex,
            "noise": noise_source,
        },
        interferer_transcript="" if interferer is None else interferer.transcript,
    )
    return manifest, stems


def _grid():
    for snr in SNR_GRID_DB:
        for sir in SIR_GRID_DB:
            yield snr, sir


def _progress(i: int, n: int, fid: str) -> None:
    print(f"\r  [{i}/{n}] {fid:<40}", end="", file=sys.stderr, flush=True)


# --- sets -----------------------------------------------------------------------------------


def make_tiny(out_root: Path, seed: int, tiny_dir: Path = TINY_DIR) -> list[Manifest]:
    """Both orderings of the two committed talkers x SNR grid x SIR grid = 16 fixtures (offline)."""
    set_name = "tiny"
    set_root = out_root / set_name
    src = TinySource(tiny_dir)
    speech = src.speech()
    if len(speech) != 2:
        raise ValueError(f"tiny manifest must list exactly two speech clips, found {len(speech)}")
    noise_type, noise, noise_source = src.noise()
    manifests: list[Manifest] = []
    orderings = [(speech[0], speech[1]), (speech[1], speech[0])]
    total = len(orderings) * len(SNR_GRID_DB) * len(SIR_GRID_DB)
    idx = 0
    for target, interferer in orderings:
        for snr, sir in _grid():
            manifest, stems = build_fixture(
                idx, set_name, target, interferer, noise, noise_type, noise_source, snr, sir, seed
            )
            write_fixture(set_root, manifest, stems)
            manifests.append(manifest)
            idx += 1
            _progress(idx, total, manifest.id)
    print(file=sys.stderr)
    write_index(set_root, set_name, seed, manifests)
    return manifests


def make_noise_only_set(
    out_root: Path,
    set_name: str,
    targets: list[Utterance],
    noises_for: Callable[[Utterance], list[tuple[str, np.ndarray, dict]]],
    seed: int,
) -> list[Manifest]:
    """Every target x each ``(noise_type, noise, source)`` of ``noises_for(target)`` x SNR grid."""
    set_root = out_root / set_name
    manifests: list[Manifest] = []
    plan = [(t, noises_for(t)) for t in targets]
    total = sum(len(noises) for _, noises in plan) * len(SNR_GRID_DB)
    idx = 0
    for target, noises in plan:
        for noise_type, noise, noise_source in noises:
            for snr in SNR_GRID_DB:
                manifest, stems = build_fixture(
                    idx, set_name, target, None, noise, noise_type, noise_source, snr, None, seed
                )
                write_fixture(set_root, manifest, stems)
                manifests.append(manifest)
                idx += 1
                _progress(idx, total, manifest.id)
    print(file=sys.stderr)
    write_index(set_root, set_name, seed, manifests)
    return manifests


def _pink(n: int, rng: np.random.Generator, seed: int) -> tuple[str, np.ndarray, dict]:
    return "pink", pink_noise(n, rng), {"type": "pink", "generator": "1/f-shaped gaussian", "seed": seed}


def make_tiny_noise(out_root: Path, seed: int, tiny_dir: Path = TINY_DIR) -> list[Manifest]:
    """``tiny-noise``: the two committed talkers x {committed babble, pink} x SNR grid = 16 (offline)."""
    src = TinySource(tiny_dir)
    babble = src.noise()
    pink = _pink(int(TINY_CLIP_S * aio.PIPELINE_SR), np.random.default_rng(seed), seed)
    return make_noise_only_set(out_root, NOISE_ONLY_SETS["tiny"], src.speech(), lambda _t: [babble, pink], seed)


def full_pool() -> tuple[LibriSpeechSource, dict[str, list[Utterance]]]:
    """Stream LibriSpeech test-clean once; shared by ``make_full`` and ``make_full_noise``."""
    src = LibriSpeechSource()
    return src, src.collect(FULL_MIN_S, FULL_MAX_S, per_speaker=3)


def make_full_noise(
    out_root: Path,
    seed: int,
    n_pairs: int = FULL_PAIRS,
    pool: tuple[LibriSpeechSource, dict[str, list[Utterance]]] | None = None,
) -> list[Manifest]:
    """``full-noise``: the targets of ``full``'s pairs x {babble, pink} x SNR grid (96 by default).

    Babble talkers exclude the target speaker. Noise draws use their own generator, so the
    ``full`` set is byte-identical whether or not this set is built alongside it.
    """
    src, utts = pool or full_pool()
    targets = [t for t, _ in pick_pairs(utts, n_pairs, np.random.default_rng(seed))]
    rng = np.random.default_rng([seed, 1])

    def noises_for(target: Utterance) -> list[tuple[str, np.ndarray, dict]]:
        n = len(target.audio)
        talkers = pick_babble_talkers(utts, {target.speaker_id}, BABBLE_TALKERS, rng)
        babble_source = {
            "type": "babble",
            "dataset": src.dataset,
            "split": src.split,
            "utterances": [t.id for t in talkers],
        }
        return [("babble", make_babble(talkers, n), babble_source), _pink(n, rng, seed)]

    return make_noise_only_set(out_root, NOISE_ONLY_SETS["full"], targets, noises_for, seed)


def make_full(
    out_root: Path,
    seed: int,
    n_pairs: int = FULL_PAIRS,
    pool: tuple[LibriSpeechSource, dict[str, list[Utterance]]] | None = None,
) -> list[Manifest]:
    """12 LibriSpeech pairs (4-8 s, different speakers) x SNR x SIR, babble/pink alternating."""
    set_name = "full"
    set_root = out_root / set_name
    rng = np.random.default_rng(seed)
    src, pool = pool or full_pool()
    pairs = pick_pairs(pool, n_pairs, rng)
    manifests: list[Manifest] = []
    total = n_pairs * len(SNR_GRID_DB) * len(SIR_GRID_DB)
    idx = 0
    for p, (target, interferer) in enumerate(pairs):
        n = len(target.audio)
        noise_type = FULL_NOISE_TYPES[p % len(FULL_NOISE_TYPES)]
        if noise_type == "babble":
            talkers = pick_babble_talkers(
                pool, {target.speaker_id, interferer.speaker_id}, BABBLE_TALKERS, rng
            )
            noise = make_babble(talkers, n)
            noise_source = {
                "type": "babble",
                "dataset": src.dataset,
                "split": src.split,
                "utterances": [t.id for t in talkers],
            }
        else:
            noise = pink_noise(n, rng)
            noise_source = {"type": "pink", "generator": "1/f-shaped gaussian", "seed": seed}
        for snr, sir in _grid():
            manifest, stems = build_fixture(
                idx, set_name, target, interferer, noise, noise_type, noise_source, snr, sir, seed
            )
            write_fixture(set_root, manifest, stems)
            manifests.append(manifest)
            idx += 1
            _progress(idx, total, manifest.id)
    print(file=sys.stderr)
    write_index(set_root, set_name, seed, manifests)
    return manifests


# --- audio-visual sets (GRID video + the same mixing) ----------------------------------------

AV_TINY_PAIRS, AV_FULL_PAIRS = 2, 10
AV_TINY_SNR_DB, AV_TINY_SIR_DB = (0.0, 5.0), (0.0, 5.0)
AV_LAYOUTS = ("single", "two_face")
AV_FILES = {**STEM_FILES, "video": "video.mp4"}
SAMPLES_PER_FRAME = aio.PIPELINE_SR // fxvideo.FPS  # 640: audio and video lengths agree exactly


@dataclass
class AVManifest(Manifest):
    """Audio manifest plus the video fields (see ``eval/fixtures/README.md``)."""

    layout: str = ""  # "single" (target visible, interferer off-screen) or "two_face"
    video: str = AV_FILES["video"]
    fps: float = float(fxvideo.FPS)
    frame_size: list[int] = field(default_factory=list)  # [width, height]
    n_frames: int = 0
    faces: list[dict] = field(default_factory=list)  # {"actor", "role", "side", "box": [x, y, w, h] normalised}
    target_side: str = ""  # "left" / "right" (two_face) or "center" (single)
    interferer_visible: bool = False
    target_intervals: list[list[float]] = field(default_factory=list)  # [start_s, end_s] speech spans
    interferer_intervals: list[list[float]] = field(default_factory=list)  # looped like the interferer audio
    box_source: str = "grid_approx"


@dataclass
class AVClip:
    """One talker's clip: audio fitted to ``len(frames) * 640`` samples, 25 fps RGB frames."""

    utt: Utterance
    frames: np.ndarray
    intervals: list[list[float]]
    face_box: tuple[float, float, float, float]

    @property
    def actor(self) -> str:
        return self.utt.speaker_id


def _pad_or_trim(x: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float32)
    out[: min(n, len(x))] = x[:n]
    return out


def av_clip_from_grid(g: fxgrid.GridUtterance) -> AVClip:
    frames, _ = fxvideo.read_video(g.video_path, fps=fxvideo.FPS)
    n = len(frames) * SAMPLES_PER_FRAME
    utt = Utterance(
        id=g.id,
        speaker_id=f"grid-{g.speaker}",
        text=g.transcript,
        audio=_pad_or_trim(g.audio, n),
        sex=g.sex,
        source={"dataset": fxgrid.DATASET, "split": f"zenodo:{fxgrid.ZENODO_RECORD}"},
    )
    duration = n / aio.PIPELINE_SR
    return AVClip(utt, frames, fxgrid.speech_intervals(g.words, duration), fxgrid.face_box(g.speaker))


def render_layout(
    layout: str, target: AVClip, interferer: AVClip, target_side: str
) -> tuple[np.ndarray, list[dict]]:
    """Frames (``len(target.frames)`` long, <= 640x360) and face list for one layout."""
    n = len(target.frames)
    if layout == "single":
        frames = fxvideo.fit_resize(target.frames)  # uniform scale: normalised boxes are unchanged
        faces = [{"actor": target.actor, "role": "target", "side": "center", "box": [round(v, 4) for v in target.face_box]}]
        return frames, faces
    if layout != "two_face":
        raise ValueError(f"unknown layout {layout!r}")
    if target_side not in ("left", "right"):
        raise ValueError(f"target_side must be left or right for two_face, got {target_side!r}")
    other = dataclasses.replace(interferer, frames=fxvideo.fit_frames(interferer.frames, n))
    left, right = (target, other) if target_side == "left" else (other, target)
    frames = fxvideo.side_by_side(left.frames, right.frames)
    lbox, rbox = fxvideo.side_by_side_boxes(
        left.face_box, (left.frames.shape[2], left.frames.shape[1]),
        right.face_box, (right.frames.shape[2], right.frames.shape[1]),
    )
    faces = [
        {"actor": left.actor, "role": "target" if left is target else "interferer", "side": "left", "box": lbox},
        {"actor": right.actor, "role": "target" if right is target else "interferer", "side": "right", "box": rbox},
    ]
    return frames, faces


def av_fixture_id(idx: int, layout: str, snr_db: float, sir_db: float, noise_type: str) -> str:
    return f"{idx:03d}_{layout}_snr{_fmt_db(snr_db)}_sir{_fmt_db(sir_db)}_{noise_type}"


def two_face_side(pair_index: int) -> str:
    """L, R, R, L, L, R, ...: both sides covered and each talker seen on both sides over 4 pairs."""
    return "left" if (pair_index + pair_index // 2) % 2 == 0 else "right"


def make_av_set(
    out_root: Path,
    set_name: str,
    pairs: list[tuple[AVClip, AVClip]],
    noises: list[tuple[str, np.ndarray, dict]],
    conditions: list[tuple[float, float]],
    seed: int,
    layouts: tuple[str, ...] = AV_LAYOUTS,
) -> list[AVManifest]:
    """Every pair x layout x (snr, sir): stems, manifest and ``video.mp4`` with the mix muxed in."""
    set_root = out_root / set_name
    manifests: list[AVManifest] = []
    total = len(pairs) * len(layouts) * len(conditions)
    idx = 0
    with tempfile.TemporaryDirectory(prefix="avfx_") as tmp:
        for p, (target, interferer) in enumerate(pairs):
            noise_type, noise, noise_source = noises[p]
            duration = len(target.utt.audio) / aio.PIPELINE_SR
            period = len(interferer.utt.audio) / aio.PIPELINE_SR
            inter_spans = fxgrid.loop_intervals(interferer.intervals, period, duration)
            for layout in layouts:
                side = two_face_side(p) if layout == "two_face" else "center"
                frames, faces = render_layout(layout, target, interferer, side)
                silent = fxvideo.write_video(Path(tmp) / f"{p}_{layout}.mp4", frames, fxvideo.FPS)
                for snr, sir in conditions:
                    base, stems = build_fixture(
                        idx, set_name, target.utt, interferer.utt, noise, noise_type, noise_source, snr, sir, seed
                    )
                    fields = base.to_dict()
                    fields.update(id=av_fixture_id(idx, layout, snr, sir, noise_type), paths=dict(AV_FILES))
                    fields["source"]["layout"] = layout
                    m = AVManifest(
                        **fields,
                        layout=layout,
                        frame_size=[int(frames.shape[2]), int(frames.shape[1])],
                        n_frames=len(frames),
                        faces=faces,
                        target_side=side,
                        interferer_visible=layout == "two_face",
                        target_intervals=target.intervals,
                        interferer_intervals=inter_spans,
                    )
                    out = write_fixture(set_root, m, stems)
                    fxvideo.mux_audio(silent, out / m.paths["mix"], out / m.paths["video"])
                    manifests.append(m)
                    idx += 1
                    _progress(idx, total, m.id)
    print(file=sys.stderr)
    write_index(set_root, set_name, seed, manifests)
    return manifests


def pick_grid_pairs(
    corpus: fxgrid.GridCorpus, n_pairs: int, rng: np.random.Generator
) -> list[tuple[fxgrid.GridUtterance, fxgrid.GridUtterance]]:
    """Pair p: target from talker p % 2, interferer from the other; every utterance used once."""
    speakers = list(fxgrid.SPEAKERS)
    ids = {s: corpus.pick(s, n_pairs, rng) for s in speakers}
    pairs = []
    for p in range(n_pairs):
        t, i = speakers[p % 2], speakers[(p + 1) % 2]
        pairs.append((corpus.load(t, ids[t][p]), corpus.load(i, ids[i][p])))
    return pairs


def make_av(
    out_root: Path,
    seed: int,
    full: bool = False,
    corpus: fxgrid.GridCorpus | None = None,
    tiny_dir: Path = TINY_DIR,
) -> list[AVManifest]:
    """``av_tiny``: 2 GRID pairs x 2 layouts x SNR {0, 5} x SIR {0, 5}; ``av_full``: 10 pairs x 2 x 8."""
    corpus = corpus or fxgrid.GridCorpus()
    rng = np.random.default_rng(seed)
    n_pairs = AV_FULL_PAIRS if full else AV_TINY_PAIRS
    snrs, sirs = (SNR_GRID_DB, SIR_GRID_DB) if full else (AV_TINY_SNR_DB, AV_TINY_SIR_DB)
    conditions = [(snr, sir) for snr in snrs for sir in sirs]
    pairs = [(av_clip_from_grid(t), av_clip_from_grid(i)) for t, i in pick_grid_pairs(corpus, n_pairs, rng)]
    babble = TinySource(tiny_dir).noise()
    noises = []
    for p, (target, _) in enumerate(pairs):
        if not full or FULL_NOISE_TYPES[p % len(FULL_NOISE_TYPES)] == "babble":
            noises.append(babble)
        else:
            noise_source = {"type": "pink", "generator": "1/f-shaped gaussian", "seed": seed}
            noises.append(("pink", pink_noise(len(target.utt.audio), rng), noise_source))
    return make_av_set(out_root, "av_full" if full else "av_tiny", pairs, noises, conditions, seed)


def load_av_fixture(fixture_dir: str | Path) -> tuple[AVManifest, Stems]:
    """Like ``mix.load_fixture`` for AV fixtures (whose manifests carry the extra video fields)."""
    d = Path(fixture_dir)
    manifest = AVManifest.from_dict(json.loads((d / "manifest.json").read_text()))
    arrays = {name: aio.load(d / manifest.paths[name], manifest.sample_rate)[0] for name in STEM_NAMES}
    return manifest, Stems(**arrays)


# --- committed tiny source clips (network; run once, result is committed) --------------------

# Chosen from LibriSpeech test-clean: utterances whose full length is within 0.2 s of 3.0 s so the
# transcript stays exact after padding / trimming silence to exactly 3.0 s. Sex from LibriSpeech
# SPEAKERS.TXT, cross-checked by median f0 (1580 ≈ 232 Hz, 61 ≈ 101 Hz).
TINY_SPEECH = (
    {"id": "1580-141084-0019", "sex": "F", "file": "speech_a.wav"},
    {"id": "61-70968-0002", "sex": "M", "file": "speech_b.wav"},
)
# Six further speakers (three F, three M by f0), each ≥ 3 s, summed into noise_babble.wav.
TINY_BABBLE_IDS: tuple[str, ...] = (
    "6930-76324-0026",
    "2830-3980-0042",
    "1284-1181-0019",
    "2094-142345-0045",
    "7176-92135-0013",
    "1221-135767-0008",
)


def _fit_exact(x: np.ndarray, n: int, silence_rel: float = 0.02) -> np.ndarray:
    """Pad with zeros or trim to exactly ``n`` samples, trimming leading silence before the tail."""
    if len(x) >= n:
        excess = len(x) - n
        thr = silence_rel * float(np.max(np.abs(x)))
        loud = np.flatnonzero(np.abs(x) > thr)
        lead = int(loud[0]) if loud.size else 0
        start = min(lead, excess)
        return np.ascontiguousarray(x[start : start + n], dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[: len(x)] = x
    return out


def make_tiny_source(tiny_dir: Path) -> None:
    """Rebuild ``eval/fixtures/tiny/`` (speech_a, speech_b, noise_babble, manifest) from LibriSpeech."""
    wanted = {e["id"]: e for e in TINY_SPEECH}
    babble_ids = set(TINY_BABBLE_IDS)
    found: dict[str, Utterance] = {}
    src = LibriSpeechSource()
    # The whole split is consumed on purpose: abandoning a ``datasets`` streaming iterator early
    # leaves its parquet reader half-open and the interpreter hangs at exit (seen with datasets 5.0).
    for u in src.stream():
        if u.id in wanted or u.id in babble_ids:
            found[u.id] = u
    missing = (set(wanted) | babble_ids) - set(found)
    if missing:
        raise RuntimeError(f"utterances not found in {src.split}: {sorted(missing)}")
    n = int(TINY_CLIP_S * aio.PIPELINE_SR)
    tiny_dir.mkdir(parents=True, exist_ok=True)
    speech_entries = []
    for e in TINY_SPEECH:
        u = found[e["id"]]
        aio.save(tiny_dir / e["file"], _fit_exact(u.audio, n), aio.PIPELINE_SR)
        speech_entries.append(
            {
                "file": e["file"],
                "id": u.id,
                "speaker_id": u.speaker_id,
                "sex": e["sex"],
                "text": u.text,
                "source_duration_s": round(u.duration_s, 3),
            }
        )
    babble = make_babble([found[i] for i in TINY_BABBLE_IDS], n)
    babble = (0.5 * babble / np.max(np.abs(babble))).astype(np.float32)
    aio.save(tiny_dir / "noise_babble.wav", babble, aio.PIPELINE_SR)
    manifest = {
        "dataset": src.dataset,
        "split": src.split,
        "license": "CC BY 4.0 (LibriSpeech, Panayotov et al. 2015)",
        "sample_rate": aio.PIPELINE_SR,
        "duration_s": TINY_CLIP_S,
        "speech": speech_entries,
        "noise": {
            "file": "noise_babble.wav",
            "type": "babble",
            "source_ids": list(TINY_BABBLE_IDS),
            "source_speakers": [found[i].speaker_id for i in TINY_BABBLE_IDS],
        },
    }
    (tiny_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


# --- CLI ------------------------------------------------------------------------------------


def make_demo(generated: Path = DEFAULT_OUT, out: Path | None = None) -> Path:
    """Legacy ('legacy loop'): loop one cached two-face GRID mixture to 12 s, without fetching anything.
    ``--demo`` now builds the demo-shaped scenes in ``eval/demo_scene.py``."""
    import subprocess

    candidates = sorted((generated / "av_full").glob("*two_face*/video.mp4"))
    if not candidates:
        raise RuntimeError("no cached av_full two-face GRID clip; prepare --av --full once before going offline")
    source = next((p for p in candidates if "snr5_sir0" in str(p)), candidates[0])
    out = out or generated.parent / "demo"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "grid-two-face.mp4"
    subprocess.run([fxvideo.ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
                    "-stream_loop", "-1", "-i", str(source), "-t", "12",
                    "-vf", "fps=25,scale=640:360:force_original_aspect_ratio=decrease",
                    "-af", "aresample=async=1:first_pts=0,apad",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-ac", "1", "-ar", "16000", "-movflags", "+faststart", str(dest)], check=True)
    manifest = json.loads((source.parent / "manifest.json").read_text())
    # The clean stems of one loop, so scripts/session_score.py can score a replayed session
    # (the MP4 loops this 3 s mixture; the scorer finds each loop in the recording).
    files = manifest.get("paths") or {}
    stems = {}
    for key, name in (("mix", "mix"), ("target", "target"), ("other", "interferer"), ("noise", "noise")):
        src = source.parent / files.get(name, f"{name}.wav")
        if src.exists():
            (out / "stems").mkdir(exist_ok=True)
            shutil.copyfile(src, out / "stems" / f"{key}.wav")
            stems[key] = f"stems/{key}.wav"
    target = next((f for f in manifest.get("faces", []) if f.get("role") == "target"), {})
    (out / "manifest.json").write_text(json.dumps({"source_fixture": source.parent.name,
                                                  "duration_s": 12, "stems": stems, "target": target,
                                                  "source": manifest}, indent=2) + "\n")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--demo", action="store_true",
                   help="offline: four ~30 s demo scenes (demo-mf-easy/-hard, demo-mm, demo-ff) from cached GRID")
    g.add_argument("--demo-team", action="store_true",
                   help="offline: demo-team-easy/-hard(/-same) from $HEARING_FIXTURES_DIR/team/<name>/fixture* takes "
                        "(two cached GRID stand-ins until two teammates have recorded)")
    g.add_argument("--demo-legacy", action="store_true", help="offline: the old 12 s loop of one av_full two-face mixture")
    g.add_argument("--tiny", action="store_true", help="offline: mix the committed clips (tiny + tiny-noise, 16 each)")
    g.add_argument("--full", action="store_true", help="network: stream LibriSpeech test-clean (full + full-noise, 96 each)")
    g.add_argument("--tiny-noise", action="store_true", help="offline: only the noise-only tiny set (16 fixtures)")
    g.add_argument("--full-noise", action="store_true", help="network: only the noise-only full set (96 fixtures)")
    g.add_argument(
        "--make-tiny-source",
        action="store_true",
        help="network: rebuild the committed clips under --tiny-dir from LibriSpeech",
    )
    ap.add_argument("--av", action="store_true", help="audio-visual set from GRID (with --tiny or --full)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output root (default {DEFAULT_OUT})")
    ap.add_argument("--tiny-dir", type=Path, default=TINY_DIR, help="where the committed tiny clips live")
    ap.add_argument("--team-dir", type=Path, default=None, help="--demo-team: takes root (default <--out>/team)")
    ap.add_argument("--team-target", default=None, help="--demo-team: which teammate is the target (default: first by name)")
    ap.add_argument("--pairs", type=int, default=FULL_PAIRS, help="utterance pairs for --full")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    if args.demo:
        from eval import demo_scene

        for path in demo_scene.make_all(args.out.parent / "demo"):
            print(path)
        return 0
    if args.demo_team:
        from eval import demo_team

        for path in demo_team.build(args.out.parent / "demo", args.team_dir or args.out / "team",
                                    target=args.team_target, tiny_dir=args.tiny_dir):
            m = json.loads(path.read_text())
            print(f"{path}  ({m['duration_s']} s, {m['note']})")
        return 0
    if args.demo_legacy:
        print(make_demo(args.out))
        return 0
    if args.make_tiny_source:
        make_tiny_source(args.tiny_dir)
        print(f"wrote tiny source clips to {args.tiny_dir} in {time.perf_counter() - t0:.1f} s")
        return 0
    if args.av:
        if not (args.tiny or args.full):
            ap.error("--av needs --tiny or --full")
        corpus = fxgrid.GridCorpus()
        manifests = make_av(args.out, args.seed, full=args.full, corpus=corpus, tiny_dir=args.tiny_dir)
        set_name = "av_full" if args.full else "av_tiny"
        print(
            f"{set_name}: {len(manifests)} fixtures, {sum(m.duration_s for m in manifests):.1f} s of audio, "
            f"{time.perf_counter() - t0:.1f} s wall, downloaded {corpus.bytes_downloaded / 1e6:.1f} MB "
            f"(GRID cache {corpus.root}) -> {args.out / set_name}"
        )
        return 0
    built: list[tuple[str, list[Manifest]]] = []
    if args.tiny or args.tiny_noise:
        if args.tiny:
            built.append(("tiny", make_tiny(args.out, args.seed, args.tiny_dir)))
        built.append((NOISE_ONLY_SETS["tiny"], make_tiny_noise(args.out, args.seed, args.tiny_dir)))
    else:
        pool = full_pool()
        if args.full:
            built.append(("full", make_full(args.out, args.seed, args.pairs, pool)))
        built.append((NOISE_ONLY_SETS["full"], make_full_noise(args.out, args.seed, args.pairs, pool)))
    for set_name, manifests in built:
        total_s = sum(m.duration_s for m in manifests)
        print(
            f"{set_name}: {len(manifests)} fixtures, {total_s:.1f} s of audio, "
            f"{time.perf_counter() - t0:.1f} s wall -> {args.out / set_name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
