"""Demo-shaped two-face scenes from cached GRID talkers (T-097; ``make_fixtures --demo``).

One ~30 s clip per variant, in the shape of the live demo:

    0-5 s    target alone (GRID sentences with ~0.3 s pauses)
    5-15 s   alternating turns: target, other, target, ...
    15-25 s  overlap: both talk at SIR 0 dB
    25-30 s  target alone again

plus babble at the variant's SNR (relative to the target's active-speech level). A silent face is
held on its last frame (a listener does not move their lips). Each clip is written as
``<out>/<name>/{clip.mp4, manifest.json, stems/{mix,target,other,noise}.wav}``: the manifest
carries the stems (``scripts/session_score.py``) and the per-face speaking timeline in the
ingested-clip schema (``scripts/session_score_nr.py``, ``hearing.replay.list_clips``). Offline:
only the GRID utterances already cached under the corpus root are used.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.fixtures import grid as fxgrid
from eval.fixtures import video as fxvideo
from eval.fixtures.sources import TINY_DIR, TinySource
from hearing.audio import io as aio

SR = aio.PIPELINE_SR
FPS = fxvideo.FPS
SPF = SR // FPS  # 640 samples per video frame
GAP_FRAMES = 8  # 0.32 s pause between sentences
SLACK_S = 0.5  # a sentence may run this far past its section's planned end
DURATION_S = 30.0
SPEECH_LEVEL_DB = -18.5  # target active-speech level (dBFS), like eval/real_clip.GRID_LEVEL_DB
PEAK = 0.95
DEFAULT_CLIP = "demo-mf-easy"
# (section, planned end s, who talks): "target", "other", "alternate" or "both"
SECTIONS = (("target_alone", 5.0, "target"), ("alternate", 15.0, "alternate"),
            ("overlap", 25.0, "both"), ("target_alone_2", DURATION_S, "target"))


@dataclass(frozen=True)
class Variant:
    name: str
    target: str  # GRID speaker
    other: str
    snr_db: float
    target_side: str = "left"
    note: str = ""


VARIANTS = (
    Variant("demo-mf-easy", "s1", "s4", 10.0, note="default: male target, female other, babble SNR 10 dB"),
    Variant("demo-mf-hard", "s1", "s4", 5.0, note="male target, female other, babble SNR 5 dB"),
    Variant("demo-mm", "s1", "s2", 10.0, note="same-gender (male) pair, babble SNR 10 dB"),
    Variant("demo-ff", "s4", "s7", 10.0, note="same-gender (female) pair, babble SNR 10 dB"),
)


@dataclass
class Sentence:
    utt: str
    audio: np.ndarray  # trimmed to whole frames
    frames: np.ndarray
    text: str


def load_sentences(corpus: fxgrid.GridCorpus, speaker: str) -> list[Sentence]:
    """Every cached utterance of ``speaker``, trimmed to its speech (frame-aligned, 1 frame margin)."""
    out = []
    for path in sorted((corpus.root / speaker).glob("*.mpg")):
        g = corpus.load(speaker, path.stem)
        frames, _ = fxvideo.read_video(g.video_path, fps=FPS)
        spans = fxgrid.speech_intervals(g.words, len(frames) / FPS)
        if not spans:
            continue
        f0 = max(0, int(spans[0][0] * FPS) - 1)
        f1 = min(len(frames), int(np.ceil(spans[-1][1] * FPS)) + 1)
        audio = np.zeros((f1 - f0) * SPF, np.float32)
        seg = g.audio[f0 * SPF: f1 * SPF]
        audio[: len(seg)] = seg
        out.append(Sentence(path.stem, audio, frames[f0:f1], g.transcript))
    if not out:
        raise RuntimeError(f"no cached GRID clips for {speaker} under {corpus.root}; run make_fixtures --av --full once")
    return out


def schedule(n_frames: dict[str, list[int]], sections=SECTIONS, gap: int = GAP_FRAMES,
             fps: int = FPS, slack_s: float = SLACK_S) -> tuple[list[dict], list[dict]]:
    """Place sentences (lengths in frames, used cyclically per talker) into the sections.

    Returns ``(placements, sections)``: ``{"who", "index", "start", "end"}`` in frames and
    ``{"name", "start_s", "end_s"}``. A sentence is added only while it ends before the section's
    planned end + ``slack_s``; in ``both`` each talker fills the section on its own track."""
    used = {k: 0 for k in n_frames}
    placements, out_sections = [], []
    cursor = 0
    for name, end_s, who in sections:
        end = round((end_s + slack_s) * fps)
        start = cursor
        tracks = ["target", "other"] if who == "both" else [who]
        ends = []
        for track in tracks:
            c, turn = cursor + gap, 0
            while True:
                speaker = track if track != "alternate" else ("target", "other")[turn % 2]
                n = n_frames[speaker][used[speaker] % len(n_frames[speaker])]
                if c + n > end:
                    break
                placements.append({"who": speaker, "index": used[speaker] % len(n_frames[speaker]),
                                   "start": c, "end": c + n})
                used[speaker] += 1
                c += n + gap
                turn += 1
            ends.append(c - gap)
        cursor = max(ends)
        out_sections.append({"name": name, "start_s": round(start / fps, 2), "end_s": round(cursor / fps, 2)})
    return placements, out_sections


def spans(placements: list[dict], who: str) -> list[list[float]]:
    return [[round(p["start"] / FPS, 2), round(p["end"] / FPS, 2)] for p in placements if p["who"] == who]


def active_rms(x: np.ndarray, segs: list[list[float]]) -> float:
    parts = [x[int(s * SR): int(e * SR)] for s, e in segs]
    a = np.concatenate(parts) if parts else x
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2)) + 1e-12)


def render(sentences: dict[str, list[Sentence]], placements: list[dict], n_total: int):
    """Audio tracks (target, other) and per-talker frame stacks; silent faces hold their last frame."""
    audio = {k: np.zeros(n_total * SPF, np.float32) for k in sentences}
    frames = {}
    for who, sents in sentences.items():
        h, w = sents[0].frames.shape[1:3]
        f = np.empty((n_total, h, w, 3), np.uint8)
        last = sents[0].frames[0]
        mine = sorted((p for p in placements if p["who"] == who), key=lambda p: p["start"])
        i = 0
        for p in mine:
            f[i:p["start"]] = last
            s = sents[p["index"]]
            f[p["start"]:p["end"]] = s.frames
            audio[who][p["start"] * SPF: p["end"] * SPF] = s.audio
            last = s.frames[-1]
            i = p["end"]
        f[i:] = last
        frames[who] = f
    return audio, frames


def babble(n: int, tiny_dir: Path = TINY_DIR) -> tuple[np.ndarray, dict]:
    """The committed 3 s six-talker babble, tiled to ``n`` samples (a 3 s period)."""
    _, x, source = TinySource(tiny_dir).noise()
    return np.resize(np.asarray(x, np.float32), n).astype(np.float32), {**source, "tiled_period_s": len(x) / SR}


def build(variant: Variant, corpus: fxgrid.GridCorpus, out_root: Path, tiny_dir: Path = TINY_DIR) -> Path:
    """Write one variant's clip, stems and manifest; returns the manifest path."""
    sents = {"target": load_sentences(corpus, variant.target), "other": load_sentences(corpus, variant.other)}
    spk = {"target": variant.target, "other": variant.other}
    actors = {who: {"actor": f"grid-{s}", "sex": fxgrid.SPEAKERS.get(s) or fxgrid.EXTRA_SPEAKERS.get(s, ""),
                    "box": fxgrid.face_box(s)} for who, s in spk.items()}
    noise, noise_source = babble(round(DURATION_S * SR) * 2, tiny_dir)
    return compose(variant.name, sents, actors, variant.snr_db, noise, noise_source, out_root,
                   target_side=variant.target_side, note=variant.note,
                   source={"dataset": fxgrid.DATASET, "license": fxgrid.LICENSE, "speakers": spk})


def compose(name: str, sents: dict[str, list[Sentence]], actors: dict[str, dict], snr_db: float,
            noise: np.ndarray, noise_source: dict, out_root: Path, *, target_side: str = "left", note: str = "",
            source: dict | None = None, kind: str = "synthetic_scene", extra: dict | None = None) -> Path:
    """Schedule ``sents`` (``target``/``other`` sentence lists) into the demo shape and write the clip.

    ``actors[who]`` = ``{"actor", "sex", "box"}`` (box normalised in that talker's own frames);
    ``noise`` is tiled to the clip's length and scaled to ``snr_db`` below the target's speech."""
    placements, sections = schedule({k: [len(s.frames) for s in v] for k, v in sents.items()}, SECTIONS)
    n_total = round(DURATION_S * FPS)
    n_total = max(n_total, max(p["end"] for p in placements))
    audio, frames = render(sents, placements, n_total)
    sections[-1]["end_s"] = round(n_total / FPS, 2)
    timeline = {k: spans(placements, k) for k in ("target", "other")}
    # equal active-speech level (SIR 0 in the overlap), target at SPEECH_LEVEL_DB, noise at the SNR
    ref = 10 ** (SPEECH_LEVEL_DB / 20)
    target = audio["target"] * (ref / active_rms(audio["target"], timeline["target"]))
    other = audio["other"] * (ref / active_rms(audio["other"], timeline["other"]))
    noise = np.resize(np.asarray(noise, np.float32), len(target)).astype(np.float32)
    noise *= ref / active_rms(noise, []) / 10 ** (snr_db / 20)
    mix = target + other + noise
    g = min(1.0, PEAK / float(np.max(np.abs(mix))))
    stems = {"mix": mix * g, "target": target * g, "other": other * g, "noise": noise * g}

    left, right = ("target", "other") if target_side == "left" else ("other", "target")
    composite = fxvideo.side_by_side(frames[left], frames[right])
    boxes = fxvideo.side_by_side_boxes(
        actors[left]["box"], (frames[left].shape[2], frames[left].shape[1]),
        actors[right]["box"], (frames[right].shape[2], frames[right].shape[1]))
    out = Path(out_root) / name
    (out / "stems").mkdir(parents=True, exist_ok=True)
    for k, x in stems.items():
        aio.save(out / "stems" / f"{k}.wav", x.astype(np.float32), SR)
    with tempfile.TemporaryDirectory(prefix="demo_scene_") as tmp:
        silent = fxvideo.write_video(Path(tmp) / "v.mp4", composite, FPS)
        fxvideo.mux_audio(silent, out / "stems" / "mix.wav", out / "clip.mp4")

    faces = []
    for track_id, (side, who, box) in enumerate(((("left", left, boxes[0])), ("right", right, boxes[1]))):
        segs = timeline[who]
        faces.append({"track_id": track_id, "side": side, "role": who, "actor": actors[who]["actor"],
                      "sex": actors[who].get("sex", ""),
                      "center_x": round(box[0] + box[2] / 2, 4), "mean_box": [round(v, 4) for v in box],
                      "speaking": segs, "speaking_s": round(sum(e - s for s, e in segs), 2)})
    tface = next(f for f in faces if f["role"] == "target")
    manifest = {
        "version": 2, "name": name, "kind": kind, "note": note,
        "clip": "clip.mp4", "duration_s": round(n_total / FPS, 2), "fps": FPS, "sample_rate": SR,
        "frame_size": [int(composite.shape[2]), int(composite.shape[1])],
        "snr_db": snr_db, "sir_db_overlap": 0.0, "speech_level_dbfs": SPEECH_LEVEL_DB,
        "stems": {k: f"stems/{k}.wav" for k in stems},
        "sections": sections,
        "target": {"rule": target_side, "track_id": tface["track_id"], "side": tface["side"],
                   "box": tface["mean_box"], "actor": tface["actor"]},
        "faces": faces,
        "transcript": " ".join(t for t in (sents["target"][p["index"]].text for p in placements
                                           if p["who"] == "target") if t),
        "sentences": [{"who": p["who"], "utt": sents[p["who"]][p["index"]].utt, "text": sents[p["who"]][p["index"]].text,
                       "start_s": round(p["start"] / FPS, 2), "end_s": round(p["end"] / FPS, 2)} for p in placements],
        "source": {**(source or {}), "noise": noise_source},
        **(extra or {}),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def make_all(out_root: Path, corpus: fxgrid.GridCorpus | None = None, variants=VARIANTS) -> list[Path]:
    corpus = corpus or fxgrid.GridCorpus()
    return [build(v, corpus, out_root) for v in variants]
