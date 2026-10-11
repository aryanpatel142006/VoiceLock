"""Teammates' demo scenes from their own solo takes (T-126; ``make_fixtures --demo-team``).

Same shape as ``eval/demo_scene.py`` (5 s target alone, turns, overlap at SIR 0, alone again),
built from each teammate's solo recordings instead of GRID:

    $HEARING_FIXTURES_DIR/team/<name>/fixture*.{wav,m4a,aac,mp4,mov}   takes cut into the scene
    $HEARING_FIXTURES_DIR/team/<name>/profile*.{wav,m4a,aac,mp4,mov}   enrollment takes, never in a scene
    $HEARING_FIXTURES_DIR/team/<name>/meta.json            optional {"bucket": "male_adult"}
    $HEARING_FIXTURES_DIR/team/venue_noise.wav             optional room noise (else babble)

``scripts/record_take.py`` writes .wav/.mp4; QuickTime / Voice Memos files (.mov, .m4a) work as
they are. A take is one stem: audio from ``<stem>.wav`` if present (the lossless mic file), else
``.m4a``/``.aac``, else the video's track; video from ``<stem>.mp4``/``.mov`` (else a grey card:
the clip still plays, but the camera finds no face). Each take is cut into sentence-sized pieces
at its pauses (energy VAD on 40 ms frames); a long take (a 2-5 min QuickTime recording) gives
only its first ``TAKE_BUDGET_S`` s of speech, and only that much video is decoded. Other files
(``noisy.wav``) and ``team/conversation/`` (the two-person Take 4: room video + close-talk phone
refs synced by claps) are not used here; scoring the conversation against its refs is T-132's. Writes ``demo-team-easy`` (noise SNR 10 dB),
``demo-team-hard`` (SNR 5 dB) and, when both teammates are in the same voice bucket,
``demo-team-same`` (SNR 10 dB, the scene the voice matcher cannot tell apart). Until two
teammates have fixture takes the same clips are built from two cached GRID talkers as stand-ins
(``STANDINS``; manifest ``"standin": true``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval import demo_scene as ds
from eval.fixtures import grid as fxgrid
from eval.fixtures import video as fxvideo
from eval.fixtures.sources import TINY_DIR
from hearing.audio import io as aio

SR = ds.SR
FPS = ds.FPS
SPF = ds.SPF
AUDIO_EXT = (".wav", ".m4a", ".aac")  # preference order for a take's audio
VIDEO_EXT = (".mp4", ".mov")  # video (+ audio track when no audio file shares the stem)
MEDIA = AUDIO_EXT + VIDEO_EXT
CONVERSATION_DIR = "conversation"  # team/conversation/: Take 4, not a teammate
TAKE_BUDGET_S = 40.0  # speech kept per teammate across takes: the scene uses ~25 s of the target
STANDINS = ("s2", "s7")  # GRID male/female pair not used by demo-mf-*; target = first
SNR_DB = {"demo-team-easy": 10.0, "demo-team-hard": 5.0, "demo-team-same": 10.0}
# A centred face, hairline to chin, in a webcam frame: record_take.py asks the speaker to sit centred.
FACE_BOX_DEFAULT = (0.33, 0.12, 0.34, 0.66)
CARD_SIZE = (320, 360)  # (w, h) grey card for a take without video
# VAD on video-frame (40 ms) energy
VAD_FLOOR_DB = -50.0  # never call quieter than this speech
VAD_RANGE_DB = 30.0  # speech = within this of the take's loud (95th percentile) frames ...
VAD_ABOVE_NOISE_DB = 8.0  # ... and this far above its quiet (20th percentile) frames
MIN_GAP_FRAMES = 8  # a pause this long (0.32 s) ends a piece
MIN_PIECE_FRAMES = 10  # pieces shorter than 0.4 s are dropped (clicks, breaths)
MAX_PIECE_FRAMES = 100  # a run longer than 4 s is split at its quietest frame


@dataclass
class Teammate:
    name: str
    fixture: list[dict]  # [{"stem", "audio", "video", "files"}]
    profile: list[Path]
    bucket: str | None = None


def team_dir(fixtures: Path | None = None) -> Path:
    return Path(fixtures or ds_fixtures()) / "team"


def ds_fixtures() -> Path:
    from eval import make_fixtures

    return make_fixtures.DEFAULT_OUT


def _takes(d: Path, prefix: str) -> dict[str, dict]:
    """``{stem: {"stem", "audio", "video", "files"}}`` for the media files starting with ``prefix``."""
    by_stem: dict[str, dict[str, Path]] = {}
    for p in sorted(d.glob(f"{prefix}*")):
        if p.suffix.lower() in MEDIA and p.is_file():
            by_stem.setdefault(p.stem, {})[p.suffix.lower()] = p
    out = {}
    for stem, files in by_stem.items():
        video = next((files[e] for e in VIDEO_EXT if e in files), None)
        audio = next((files[e] for e in AUDIO_EXT if e in files), video)
        out[stem] = {"stem": stem, "audio": audio, "video": video, "files": sorted(files.values())}
    return out


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover(root: Path) -> list[Teammate]:
    """Teammates under ``root`` with at least one fixture take, sorted by name.

    Raises ``ValueError`` when a fixture file is byte-identical to a profile file (the scene must
    never contain the audio the voice was enrolled from)."""
    out = []
    for d in sorted(p for p in Path(root).glob("*") if p.is_dir() and p.name != CONVERSATION_DIR):
        fixture = list(_takes(d, "fixture").values())
        if not fixture:
            continue
        profile = [p for t in _takes(d, "profile").values() for p in t["files"]]
        fixture_files = [p for t in fixture for p in t["files"]]
        shared = {sha(p) for p in profile} & {sha(p) for p in fixture_files}
        if shared:
            raise ValueError(f"{d.name}: a fixture take is the same file as a profile take; record a separate take")
        meta = d / "meta.json"
        bucket = json.loads(meta.read_text()).get("bucket") if meta.exists() else None
        out.append(Teammate(d.name, fixture, profile, bucket))
    return out


def frame_db(audio: np.ndarray) -> np.ndarray:
    n = len(audio) // SPF
    x = audio[: n * SPF].astype(np.float64).reshape(n, SPF)
    return 10 * np.log10(np.mean(x**2, axis=1) + 1e-12)


def speech_runs(db: np.ndarray) -> list[tuple[int, int]]:
    """Frame runs ``[a, b)`` of speech: pauses < ``MIN_GAP_FRAMES`` bridged, short runs dropped,
    long runs split at their quietest frame."""
    if not len(db):
        return []
    thr = max(VAD_FLOOR_DB, np.percentile(db, 95) - VAD_RANGE_DB, np.percentile(db, 20) + VAD_ABOVE_NOISE_DB)
    on = db > thr
    runs, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        if runs and i - runs[-1][1] < MIN_GAP_FRAMES:
            runs[-1] = (runs[-1][0], j)
        else:
            runs.append((i, j))
        i = j
    out = []
    stack = [r for r in runs if r[1] - r[0] >= MIN_PIECE_FRAMES]
    while stack:
        a, b = stack.pop(0)
        if b - a <= MAX_PIECE_FRAMES:
            out.append((a, b))
            continue
        lo, hi = a + MIN_PIECE_FRAMES * 2, b - MIN_PIECE_FRAMES * 2
        cut = lo + int(np.argmin(db[lo:hi])) if hi > lo else (a + b) // 2
        stack[:0] = [(a, cut), (cut, b)]
    return out


def load_audio(take: dict, name: str) -> np.ndarray:
    path = take["audio"]
    audio = aio.load(path, SR)[0] if path.suffix.lower() == ".wav" else fxvideo.read_audio(path, SR)
    if not len(audio):
        raise ValueError(f"{name}/{take['stem']}: no audio")
    return np.asarray(audio, np.float32)


def card(n: int) -> np.ndarray:
    w, h = CARD_SIZE
    return np.full((n, h, w, 3), 96, np.uint8)


def take_sentences(take: dict, name: str, budget_s: float = TAKE_BUDGET_S) -> list[ds.Sentence]:
    """One take cut into sentence-sized pieces with their video frames (frame-aligned, 1 frame margin).

    Keeps pieces from the start until ``budget_s`` s of speech; the video is decoded only up to the
    last piece kept, so a 5-minute take costs what its first minute does."""
    audio = load_audio(take, name)
    n = len(audio) // SPF
    runs, total = [], 0
    for a, b in speech_runs(frame_db(audio)):
        if runs and total + (b - a) > budget_s * FPS:
            break
        runs.append((max(0, a - 1), min(n, b + 1)))
        total += b - a
    if not runs:
        return []
    end = runs[-1][1]
    frames = fxvideo.read_video(take["video"], fps=FPS, duration_s=end / FPS)[0] if take["video"] else card(end)
    frames = fxvideo.fit_frames(frames, end)  # AAC/encoder padding: a frame more or less
    return [ds.Sentence(f"{take['stem']}#{k}", audio[a * SPF: b * SPF].copy(), frames[a:b], "")
            for k, (a, b) in enumerate(runs)]


def teammate_sentences(t: Teammate, budget_s: float = TAKE_BUDGET_S) -> list[ds.Sentence]:
    sents = []
    for take in t.fixture:
        left = budget_s - sum(len(s.audio) for s in sents) / SR
        if left <= 0:
            break
        sents += take_sentences(take, t.name, left)
    if not sents:
        raise RuntimeError(f"{t.name}: no speech found in {[x['stem'] for x in t.fixture]}")
    if len({s.frames.shape[1:] for s in sents}) > 1:  # takes from different cameras: one size per face
        h, w = sents[0].frames.shape[1:3]
        for s in sents:
            if s.frames.shape[1:3] != (h, w):
                s.frames = fxvideo.resize_frames(s.frames, (w, h))
    return sents


def venue_noise(root: Path, n: int, tiny_dir: Path = TINY_DIR) -> tuple[np.ndarray, dict]:
    """``root/venue_noise.wav`` tiled to ``n`` samples, else the committed babble."""
    path = Path(root) / "venue_noise.wav"
    if path.exists():
        x, _ = aio.load(path, SR)
        if len(x) >= SR:
            return np.resize(np.asarray(x, np.float32), n), {"kind": "venue", "path": str(path),
                                                             "tiled_period_s": round(len(x) / SR, 2)}
    noise, src = ds.babble(n, tiny_dir)
    return noise, {**src, "kind": "babble"}


def variants(buckets: tuple[str | None, str | None]) -> list[str]:
    names = ["demo-team-easy", "demo-team-hard"]
    if buckets[0] and buckets[0] == buckets[1]:
        names.append("demo-team-same")
    return names


def standin_team(corpus: fxgrid.GridCorpus, speakers=STANDINS) -> tuple[dict, dict, dict]:
    sents = {who: ds.load_sentences(corpus, s) for who, s in zip(("target", "other"), speakers, strict=True)}
    sex = {s: fxgrid.SPEAKERS.get(s) or fxgrid.EXTRA_SPEAKERS.get(s, "") for s in speakers}
    actors = {who: {"actor": f"grid-{s}", "sex": sex[s], "box": fxgrid.face_box(s), "bucket":
                    {"M": "male_adult", "F": "female_adult"}.get(sex[s])} for who, s in
              zip(("target", "other"), speakers, strict=True)}
    return sents, actors, {"dataset": fxgrid.DATASET, "license": fxgrid.LICENSE, "speakers": dict(zip(
        ("target", "other"), speakers, strict=True))}


def build(out_root: Path, root: Path | None = None, corpus: fxgrid.GridCorpus | None = None,
          target: str | None = None, tiny_dir: Path = TINY_DIR) -> list[Path]:
    """Build every team variant; returns the manifest paths."""
    root = Path(root) if root else team_dir()
    team = discover(root) if root.exists() else []
    if len(team) >= 2:
        if target:
            team.sort(key=lambda t: t.name != target)
            if team[0].name != target:
                raise ValueError(f"no teammate {target!r} under {root} (have {[t.name for t in team]})")
        pair = team[:2]
        sents = {who: teammate_sentences(t) for who, t in zip(("target", "other"), pair, strict=True)}
        actors = {who: {"actor": f"team-{t.name}", "sex": "", "box": FACE_BOX_DEFAULT, "bucket": t.bucket}
                  for who, t in zip(("target", "other"), pair, strict=True)}
        source = {"dataset": "team recordings", "license": "own recordings (consented, not redistributed)",
                  "team": {t.name: {"fixture_takes": [x["stem"] for x in t.fixture],
                                    "profile_takes": sorted(p.name for p in t.profile), "bucket": t.bucket}
                           for t in pair}}
        standin = False
        conv = root / CONVERSATION_DIR
        if conv.is_dir():
            source["conversation"] = {"files": sorted(p.name for p in conv.iterdir() if p.is_file()), "used": False,
                                      "why": "two-person take with close-talk refs; scored by the T-132 eval"}
    else:
        sents, actors, source = standin_team(corpus or fxgrid.GridCorpus())
        source["why"] = f"fewer than two teammates with fixture takes under {root}"
        standin = True
    noise, noise_source = venue_noise(root, round(ds.DURATION_S * SR) * 2, tiny_dir)
    paths = []
    for name in variants((actors["target"]["bucket"], actors["other"]["bucket"])):
        note = (f"{'stand-in GRID ' if standin else ''}{actors['target']['actor']} (target, left) vs "
                f"{actors['other']['actor']}, {noise_source['kind']} SNR {SNR_DB[name]:g} dB")
        paths.append(ds.compose(name, sents, actors, SNR_DB[name], noise, noise_source, out_root, note=note,
                                source=source, kind="team_scene",
                                extra={"standin": standin, "buckets": {k: a["bucket"] for k, a in actors.items()}}))
    return paths
