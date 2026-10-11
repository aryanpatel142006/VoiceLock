"""Active-speaker detection accuracy on the GRID audio-visual fixture bank (T-008).

For every fixture under ``eval/fixtures/generated/av_<set>/<id>/`` the video is decoded at
25 fps and run through the MediaPipe face pipeline (``hearing.vision.faces``) once; the
resulting tracks (boxes + 478 landmarks) are then replayed, together with ``mix.wav`` in 10 ms
chunks, through each ASD backend from ``hearing.vision.asd.get_asd`` (``lipmotion`` and
``light_asd`` by default; a backend whose weights are missing is skipped and the reason
printed). Each frame, tracks are matched to the manifest faces by box IoU (greedy, IoU >=
``iou_min``); a face with no matching track scores ``p = 0``. A face is labelled speaking when
the frame centre lies inside one of its manifest intervals (``target_intervals`` for the
``target`` role, ``interferer_intervals`` for the ``interferer``).

Timing convention: frame ``i`` has capture time ``i / 25`` s; before its ``update`` the detector
receives the 10 ms audio chunks covering ``[i / 25, (i + 1) / 25)``, and the label is read at
the frame centre ``(i + 0.5) / 25``.

Metrics (pooled over face-frames): frame accuracy and precision/recall of ``p >= 0.5``,
ROC-AUC of ``p`` (Mann-Whitney, ties count half), and onset latency: per speaking interval,
the time from its start to the first frame inside the interval with ``p >= 0.5`` (median over
detected onsets; onsets never reached inside the interval count as missed). One table per
backend, rows per ``layout snr sir`` condition plus overall. ``acc±c`` is the same frame
accuracy ignoring frames within ``COLLAR_S`` (0.25 s) of an interval boundary, the usual
diarization collar; the headline ``acc`` has no collar. A second table pools rows by SNR,
plus ``snr0+5`` (the T-051 acceptance row). Target selection is scored elsewhere
(``eval/target_eval.py``).

Calibration (T-051): ``--fit`` refits each backend's score-to-probability map from the raw
per-frame scores (Light-ASD logits; lipmotion features, with ``w_motion``/``w_coupling`` held
at the detector's values) by logistic regression, and reports the fitted parameters plus
leave-one-pair-out frame accuracy (fit on 9 GRID pairs, score the 10th). ``--param
backend.key=value`` overrides a detector keyword, for sweeps.

    .venv/bin/python -m eval.asd_eval --set tiny                 # av_tiny, two-face fixtures
    .venv/bin/python -m eval.asd_eval --set tiny --layout all    # also the single-face layout
    .venv/bin/python -m eval.asd_eval --set av_full --fit        # T-051 operating points
    .venv/bin/python -m eval.asd_eval --set full --backends lipmotion --param lipmotion.window_s=0.5

Writes ``eval/results/asd_<stamp>.json`` with ``hearing.device.describe()``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hearing import device
from hearing.types import AudioChunk, FaceTrack, VideoFrame
from hearing.vision.asd import SpeakerDetector, get_asd
from hearing.vision.tracker import iou_matrix

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURES = Path(os.environ["HEARING_FIXTURES_DIR"]) if os.environ.get("HEARING_FIXTURES_DIR") else ROOT / "eval" / "fixtures" / "generated"
DEFAULT_OUT = ROOT / "eval" / "results"
DEFAULT_BACKENDS = ("lipmotion", "light_asd")
FPS = 25
SR = 16000
HOP = SR // 100  # 10 ms audio chunks.
THRESHOLD = 0.5
IOU_MIN = 0.1  # Manifest boxes are approximate (box_source "grid_approx"); faces never overlap.
COLLAR_S = 0.25  # Secondary accuracy ignores frames this close to an interval boundary.
ACCEPT_SNRS = (0.0, 5.0)  # T-051: the pooled "snr0+5" row.


# --------------------------------------------------------------------- inputs
@dataclass
class FaceLabel:
    """One manifest face: its (normalised x, y, w, h) box and speaking intervals in seconds."""

    name: str
    box: tuple[float, float, float, float]
    intervals: list[tuple[float, float]]

    def speaking(self, t: float) -> bool:
        return any(s <= t < e for s, e in self.intervals)


@dataclass
class FrameInput:
    """One video frame: tracks from the face pipeline, its image, and the audio since the last frame."""

    ts: float
    tracks: list[FaceTrack]
    audio: np.ndarray | None = None
    image: np.ndarray | None = None


@dataclass
class FaceSeries:
    """Per-frame scores of one manifest face under one backend."""

    face: FaceLabel
    t: np.ndarray  # Label times (frame centres), s.
    p: np.ndarray  # p_speaking of the matched track (0 when unmatched).
    y: np.ndarray  # Ground truth, bool.
    matched: np.ndarray  # A track matched this face in this frame, bool.
    x: np.ndarray | None = None  # (frames, k) raw scores/features before calibration (NaN when unmatched).


@dataclass
class FixtureResult:
    id: str
    group: str
    series: list[FaceSeries]
    update_ms: list[float] = field(default_factory=list)
    snr_db: float | None = None
    pair: str = ""  # GRID target|interferer utterances: the unit of leave-one-out calibration.


# --------------------------------------------------------------------- metrics
def roc_auc(p: np.ndarray, y: np.ndarray) -> float | None:
    """Mann-Whitney ROC-AUC (ties count half); ``None`` when ``y`` has a single class."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.empty(len(p), dtype=np.float64)
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0  # Average 1-based rank of the tie block.
        i = j + 1
    u = float(ranks[y].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def onset_latencies(
    t: np.ndarray, p: np.ndarray, intervals: Iterable[tuple[float, float]], threshold: float = THRESHOLD
) -> list[float | None]:
    """Per interval ``[s, e)``: ``t_first - s`` of the first frame in it with ``p >= threshold``, else None."""
    out: list[float | None] = []
    for s, e in intervals:
        inside = np.flatnonzero((t >= s) & (t < e))
        if inside.size == 0:
            continue  # No frame centre falls in the interval: not scoreable.
        hits = inside[p[inside] >= threshold]
        out.append(float(t[hits[0]] - s) if hits.size else None)
    return out


def near_boundary(t: np.ndarray, intervals: Iterable[tuple[float, float]], collar_s: float) -> np.ndarray:
    """Frames within ``collar_s`` of any interval start or end."""
    edges = np.array([b for iv in intervals for b in iv], dtype=np.float64)
    if edges.size == 0 or collar_s <= 0:
        return np.zeros(len(t), dtype=bool)
    return np.min(np.abs(np.asarray(t)[:, None] - edges[None, :]), axis=1) < collar_s


def summarize(series: Sequence[FaceSeries], threshold: float = THRESHOLD, collar_s: float = COLLAR_S) -> dict:
    """Pooled frame metrics over ``series`` (see module doc)."""
    if not series:
        return {"face_frames": 0}
    p = np.concatenate([s.p for s in series])
    y = np.concatenate([s.y for s in series]).astype(bool)
    keep = ~np.concatenate([near_boundary(s.t, s.face.intervals, collar_s) for s in series])
    matched = np.concatenate([s.matched for s in series]).astype(bool)
    pred = p >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    lats = [lat for s in series for lat in onset_latencies(s.t, s.p, s.face.intervals, threshold)]
    found = [lat for lat in lats if lat is not None]
    return {
        "face_frames": len(p),
        "speaking_frames": int(y.sum()),
        "matched_frac": float(matched.mean()),
        "accuracy": float((pred == y).mean()),
        "accuracy_collar": float((pred == y)[keep].mean()) if keep.any() else None,
        "collar_s": collar_s,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "auc": roc_auc(p, y),
        "onsets": len(lats),
        "onsets_missed": len(lats) - len(found),
        "onset_latency_median_s": float(np.median(found)) if found else None,
        "onset_latency_s": lats,
    }


# --------------------------------------------------------------------- scoring
def match_faces(tracks: Sequence[FaceTrack], faces: Sequence[FaceLabel], iou_min: float = IOU_MIN) -> dict[int, int]:
    """Greedy one-to-one face index -> track index by box IoU (>= ``iou_min``)."""
    if not tracks or not faces:
        return {}
    iou = iou_matrix(np.asarray([f.box for f in faces], dtype=np.float64), np.asarray([t.box for t in tracks], dtype=np.float64))
    out: dict[int, int] = {}
    used: set[int] = set()
    for flat in np.argsort(iou, axis=None, kind="stable")[::-1]:
        fi, ti = divmod(int(flat), iou.shape[1])
        if iou[fi, ti] < iou_min:
            break
        if fi not in out and ti not in used:
            out[fi] = ti
            used.add(ti)
    return out


LIPMOTION_FEATURES = ("motion", "coupling", "activity")


def raw_scores(detector: SpeakerDetector, track_id: int) -> list[float] | None:
    """The detector's last pre-calibration output for a track: Light-ASD's logit or lipmotion's features."""
    feats = getattr(detector, "features", None)
    if isinstance(feats, dict) and track_id in feats:
        return [float(feats[track_id][k]) for k in LIPMOTION_FEATURES]
    scores = getattr(detector, "scores", None)
    if isinstance(scores, dict) and track_id in scores:
        return [float(scores[track_id])]
    return None


def _copy_tracks(tracks: Sequence[FaceTrack]) -> list[FaceTrack]:
    return [replace(t, p_speaking=0.0) for t in tracks]


def _feed_audio(detector: SpeakerDetector, audio: np.ndarray | None, ts: float, seq: int) -> AudioChunk | None:
    """Deliver ``audio`` in 10 ms chunks via ``push_audio``; detectors without it get one chunk in ``update``."""
    if audio is None or len(audio) == 0:
        return None
    chunk = AudioChunk(np.asarray(audio, dtype=np.float32), SR, seq, ts)
    push = getattr(detector, "push_audio", None)
    if push is None:
        return chunk
    for k in range(0, len(audio), HOP):
        push(AudioChunk(chunk.samples[k : k + HOP], SR, seq, ts + k / SR))
    return None


def score_frames(
    detector: SpeakerDetector,
    frames: Sequence[FrameInput],
    faces: Sequence[FaceLabel],
    fps: float = FPS,
    iou_min: float = IOU_MIN,
) -> tuple[list[FaceSeries], list[float]]:
    """Run ``detector`` over ``frames`` and read each face's ``p`` at the frame centre."""
    detector.reset()
    n = len(frames)
    p = np.zeros((len(faces), n))
    matched = np.zeros((len(faces), n), dtype=bool)
    t = np.array([fr.ts + 0.5 / fps for fr in frames])
    raw: dict[tuple[int, int], list[float]] = {}
    update_ms: list[float] = []
    for i, fr in enumerate(frames):
        tracks = _copy_tracks(fr.tracks)
        audio = _feed_audio(detector, fr.audio, fr.ts, i)
        image = VideoFrame(fr.image, i, fr.ts) if fr.image is not None else None
        start = time.perf_counter()
        detector.update(tracks, fr.ts, audio, image)
        update_ms.append((time.perf_counter() - start) * 1000.0)
        for fi, ti in match_faces(tracks, faces, iou_min).items():
            p[fi, i] = tracks[ti].p_speaking
            matched[fi, i] = True
            r = raw_scores(detector, tracks[ti].track_id)
            if r is not None:
                raw[fi, i] = r
    k = max((len(v) for v in raw.values()), default=0)
    xs = np.full((len(faces), n, k), np.nan)
    for (fi, i), v in raw.items():
        xs[fi, i] = v
    series = [
        FaceSeries(face, t, p[fi], np.array([face.speaking(x) for x in t], dtype=bool), matched[fi], xs[fi] if k else None)
        for fi, face in enumerate(faces)
    ]
    return series, update_ms


# --------------------------------------------------------------------- fixtures
def fixture_set_dir(name: str, root: Path = DEFAULT_FIXTURES) -> Path:
    """``tiny`` / ``full`` (or ``av_tiny`` / ``av_full``) -> ``<root>/av_<name>``; anything else is a path."""
    name = name.removeprefix("av_") if name in ("av_tiny", "av_full") else name
    if name in ("tiny", "full"):
        return root / f"av_{name}"
    return Path(name)


def fixture_dirs(set_dir: Path, layout: str = "two_face") -> list[Path]:
    dirs = sorted(d for d in set_dir.iterdir() if (d / "manifest.json").exists())
    if layout != "all":
        dirs = [d for d in dirs if json.loads((d / "manifest.json").read_text()).get("layout") == layout]
    return dirs


def faces_from_manifest(m: dict) -> list[FaceLabel]:
    spans = {"target": m.get("target_intervals", []), "interferer": m.get("interferer_intervals", [])}
    return [
        FaceLabel(f["role"], tuple(float(v) for v in f["box"]), [(float(s), float(e)) for s, e in spans[f["role"]]])
        for f in m["faces"]
    ]


def condition(m: dict) -> str:
    return f"{m.get('layout', '?')} snr{m['snr_db']:g} sir{m['sir_db']:g}"


def pair_key(m: dict) -> str:
    return f"{m.get('target_id', '?')}|{m.get('interferer_id', '?')}"


def decode_fixture(fixture_dir: Path, face_backend: Callable[[], object]) -> tuple[dict, list[FrameInput], float]:
    """Decode video + mix.wav and run the face pipeline once; returns manifest, frames, faces ms/frame."""
    from eval.fixtures import video as fxvideo
    from hearing.audio import io as aio
    from hearing.vision.faces import FacePipeline

    m = json.loads((fixture_dir / "manifest.json").read_text())
    images, _ = fxvideo.read_video(fixture_dir / m["paths"]["video"], fps=FPS)
    audio, _ = aio.load(fixture_dir / m["paths"]["mix"], SR)
    spf = SR // FPS
    backend = face_backend()
    pipe = FacePipeline(backend)
    frames: list[FrameInput] = []
    ms: list[float] = []
    try:
        for i, img in enumerate(images):
            ts = i / FPS
            tracks = pipe.update(VideoFrame(img, i, ts))
            ms.append(pipe.last_ms)
            seg = audio[i * spf : (i + 1) * spf]
            frames.append(FrameInput(ts, _copy_tracks(tracks), seg if len(seg) else None, img))
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
    return m, frames, float(np.mean(ms)) if ms else 0.0


# --------------------------------------------------------------------- backends
def build_detector(name: str, **kwargs) -> tuple[SpeakerDetector | None, str | None]:
    """Fresh detector, or ``(None, reason)`` when its weights/runtime are missing (never a silent mock)."""
    if name == "light_asd":
        from hearing.vision import light_asd

        if not light_asd.available():
            return None, (
                f"weights or torch/python_speech_features missing ({light_asd.model_path()}); "
                "run .venv/bin/python -m hearing.vision.light_asd --download"
            )
        try:
            det = get_asd("light_asd", backend="torch", **kwargs)
            _ = det.backend  # Load now so a broken checkpoint is reported, not crashed on.
        except (OSError, RuntimeError, ValueError) as exc:
            return None, f"failed to load: {exc}"
        return det, None
    try:
        return get_asd(name, **kwargs), None
    except (KeyError, ImportError, RuntimeError) as exc:
        return None, str(exc)


def backend_device(det: SpeakerDetector) -> str:
    inner = getattr(det, "_backend", None)
    return str(getattr(inner, "device", "numpy"))


# --------------------------------------------------------------------- report
def _fmt(v: float | None, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


def table(name: str, rows: dict[str, dict]) -> str:
    head = f"{'condition':<24}{'frames':>7}{'acc':>7}{'acc±c':>7}{'prec':>7}{'rec':>7}{'auc':>7}{'onset med s':>12}{'missed':>8}{'match':>7}"
    lines = [f"== {name}", head]
    for key, r in rows.items():
        lines.append(
            f"{key:<24}{r['face_frames']:>7}{_fmt(r['accuracy'], '.3f'):>7}"
            f"{_fmt(r['accuracy_collar'], '.3f'):>7}{_fmt(r['precision'], '.3f'):>7}"
            f"{_fmt(r['recall'], '.3f'):>7}{_fmt(r['auc'], '.3f'):>7}{_fmt(r['onset_latency_median_s'], '.2f'):>12}"
            f"{str(r['onsets_missed']) + '/' + str(r['onsets']):>8}{_fmt(r['matched_frac'], '.2f'):>7}"
        )
    return "\n".join(lines)


def rows_for(results: Sequence[FixtureResult]) -> dict[str, dict]:
    groups: dict[str, list[FaceSeries]] = {}
    for r in results:
        groups.setdefault(r.group, []).extend(r.series)
    rows = {g: summarize(groups[g]) for g in sorted(groups)}
    rows["overall"] = summarize([s for r in results for s in r.series])
    return rows


def by_role(results: Sequence[FixtureResult]) -> dict[str, dict]:
    roles: dict[str, list[FaceSeries]] = {}
    for r in results:
        for s in r.series:
            roles.setdefault(s.face.name, []).append(s)
    return {k: summarize(v) for k, v in sorted(roles.items())}


def by_snr(results: Sequence[FixtureResult], pooled: Sequence[float] = ACCEPT_SNRS) -> dict[str, dict]:
    """Rows pooled per SNR (all layouts/SIRs), plus ``snr0+5`` pooling ``pooled``."""
    groups: dict[float, list[FaceSeries]] = {}
    for r in results:
        if r.snr_db is not None:
            groups.setdefault(r.snr_db, []).extend(r.series)
    rows = {f"snr{k:g}": summarize(groups[k]) for k in sorted(groups)}
    picked = [s for k in pooled for s in groups.get(k, [])]
    if picked:
        rows["snr" + "+".join(f"{k:g}" for k in pooled)] = summarize(picked)
    return rows


# --------------------------------------------------------------------- calibration
def fit_logistic(x: np.ndarray, y: np.ndarray, offset: np.ndarray | None = None, l2: float = 1e-3) -> np.ndarray:
    """Newton-fitted ``w`` (last entry = bias) of ``p = sigmoid(x @ w[:-1] + w[-1] + offset)``."""
    x1 = np.concatenate([np.asarray(x, np.float64), np.ones((len(x), 1))], axis=1)
    y = np.asarray(y, np.float64)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, np.float64)
    w = np.zeros(x1.shape[1])
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(x1 @ w + off, -50, 50)))
        grad = x1.T @ (p - y) + l2 * w
        hess = (x1 * (p * (1 - p))[:, None]).T @ x1 + l2 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-9:
            break
    return w


def calibration_design(name: str, det: SpeakerDetector | None, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fitted columns, fixed logit offset and parameter names (bias last) for a backend's raw scores.

    Light-ASD: ``scale * logit + bias`` (the ``calibration`` pair). Lipmotion: ``w_motion_activity``,
    ``w_activity`` and ``bias`` are fitted; ``w_motion`` and ``w_coupling`` stay at the detector's values
    (they guard cases the GRID set cannot show, see :class:`hearing.vision.asd.LipMotionASD`).
    """
    if x.shape[1] == 1:
        return x, np.zeros(len(x)), ["scale", "bias"]
    motion, coupling, activity = x[:, 0], x[:, 1], x[:, 2]
    w_motion = float(getattr(det, "w_motion", 0.0))
    w_coupling = float(getattr(det, "w_coupling", 0.0))
    cols = np.stack([motion * activity, activity], axis=1)
    return cols, w_motion * motion + w_coupling * coupling, ["w_motion_activity", "w_activity", "bias"]


def calibrate(
    name: str, det: SpeakerDetector | None, results: Sequence[FixtureResult], pooled: Sequence[float] = ACCEPT_SNRS
) -> dict | None:
    """Logistic fit of labels on raw scores (all pairs) + leave-one-pair-out frame accuracy."""
    rows = [
        (s.x, s.y, r.pair, r.snr_db)
        for r in results
        for s in r.series
        if s.x is not None and s.x.shape[1] > 0
    ]
    if not rows:
        return None
    x = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows]).astype(bool)
    pairs = np.concatenate([[r[2]] * len(r[1]) for r in rows])
    snr = np.concatenate([[r[3] if r[3] is not None else np.nan] * len(r[1]) for r in rows])
    ok = ~np.isnan(x).any(axis=1)
    cols, off, names = calibration_design(name, det, np.nan_to_num(x))
    w = fit_logistic(cols[ok], y[ok], off[ok])
    out = {"params": dict(zip(names, (float(v) for v in w), strict=True)), "pairs": len(set(pairs))}
    if out["pairs"] < 2:
        return out
    correct = np.zeros(len(y), dtype=bool)
    for pr in sorted(set(pairs)):
        test = pairs == pr
        wp = fit_logistic(cols[ok & ~test], y[ok & ~test], off[ok & ~test])
        say = cols[test] @ wp[:-1] + wp[-1] + off[test] >= 0
        correct[test] = (say & ok[test]) == y[test]  # Unmatched frames say "silent", as in summarize.
    sel = np.isin(snr, pooled)
    out["lopo_accuracy"] = float(correct.mean())
    out["lopo_accuracy_snr" + "+".join(f"{k:g}" for k in pooled)] = float(correct[sel].mean()) if sel.any() else None
    return out


def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_json_safe(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, np.generic):
        return _json_safe(x.item())
    return x


def write_results(out_dir: Path, report: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["stamp"]
    path = out_dir / f"asd_{stamp}.json"
    k = 1
    while path.exists():
        path = out_dir / f"asd_{stamp}-{k}.json"
        k += 1
    path.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
    return path


# --------------------------------------------------------------------- runner
def evaluate(
    dirs: Sequence[Path],
    backends: Sequence[str] = DEFAULT_BACKENDS,
    face_backend: Callable[[], object] | None = None,
    iou_min: float = IOU_MIN,
    log=print,
    params: dict[str, dict] | None = None,
    fit: bool = False,
) -> dict:
    """Score every backend on every fixture directory; returns the JSON report (minus ``stamp``).

    ``params`` maps backend name -> detector keyword overrides; ``fit`` adds :func:`calibrate`.
    """
    params = params or {}
    if face_backend is None:
        from hearing.vision.faces import MediaPipeBackend

        face_backend = MediaPipeBackend
    detectors: dict[str, SpeakerDetector] = {}
    out: dict[str, dict] = {}
    for name in backends:
        det, reason = build_detector(name, **params.get(name, {}))
        if det is None:
            out[name] = {"available": False, "reason": reason}
            log(f"== {name}: unavailable: {reason}")
        else:
            detectors[name] = det
            out[name] = {"available": True, "device": backend_device(det), "params": params.get(name, {})}
    results: dict[str, list[FixtureResult]] = {n: [] for n in detectors}
    face_ms: list[float] = []
    for d in dirs:
        m, frames, ms = decode_fixture(d, face_backend)
        face_ms.append(ms)
        faces = faces_from_manifest(m)
        for name, det in detectors.items():
            series, update_ms = score_frames(det, frames, faces, iou_min=iou_min)
            results[name].append(FixtureResult(m["id"], condition(m), series, update_ms, m.get("snr_db"), pair_key(m)))
    for name, res in results.items():
        rows = rows_for(res)
        snr_rows = by_snr(res)
        ms = [x for r in res for x in r.update_ms]
        out[name].update(
            rows=rows,
            by_snr=snr_rows,
            by_role=by_role(res),
            update_ms_mean=float(np.mean(ms)) if ms else None,
            update_ms_p95=float(np.percentile(ms, 95)) if ms else None,
            per_fixture={r.id: summarize(r.series) for r in res},
        )
        log(table(name, rows))
        log(table(f"{name} by SNR", snr_rows))
        if fit:
            cal = out[name]["calibration"] = calibrate(name, detectors[name], res)
            log(f"   calibration fit: {json.dumps(cal)}")
        if ms:
            log(f"   update ms/frame mean {np.mean(ms):.2f} p95 {np.percentile(ms, 95):.2f} ({out[name]['device']})")
    return {
        "fixtures": [d.name for d in dirs],
        "threshold": THRESHOLD,
        "iou_min": iou_min,
        "face_pipeline_ms_per_frame": float(np.mean(face_ms)) if face_ms else None,
        "backends": out,
    }


def parse_params(items: Sequence[str]) -> dict[str, dict]:
    """``["lipmotion.window_s=0.3", "light_asd.ema_s=0"]`` -> ``{"lipmotion": {"window_s": 0.3}, ...}``."""
    out: dict[str, dict] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        backend, dot, kw = key.partition(".")
        if not (sep and dot and backend and kw):
            raise ValueError(f"--param {item!r}: expected BACKEND.KEY=VALUE")
        try:
            value = json.loads(raw.lower() if raw.lower() in ("true", "false", "null") else raw)
        except json.JSONDecodeError:
            value = raw  # A bare string.
        out.setdefault(backend, {})[kw] = value
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="tiny", help="tiny | full | av_tiny | av_full (under the fixture root) or a set directory")
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES, help="fixture root")
    ap.add_argument("--layout", default="two_face", choices=["two_face", "single", "all"])
    ap.add_argument("--backends", nargs="+", default=list(DEFAULT_BACKENDS))
    ap.add_argument("--iou-min", type=float, default=IOU_MIN)
    ap.add_argument("--limit", type=int, default=None, help="first N fixtures only")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--param", action="append", default=[], metavar="BACKEND.KEY=VALUE", help="detector keyword override")
    ap.add_argument("--fit", action="store_true", help="refit calibration + leave-one-pair-out accuracy")
    args = ap.parse_args(argv)
    try:
        params = parse_params(args.param)
    except ValueError as exc:
        ap.error(str(exc))

    set_dir = fixture_set_dir(args.set, args.fixtures)
    if not set_dir.is_dir():
        flag = "--full" if set_dir.name == "av_full" else "--tiny"
        print(f"no AV fixtures at {set_dir}; run .venv/bin/python -m eval.make_fixtures --av {flag}", file=sys.stderr)
        return 2
    dirs = fixture_dirs(set_dir, args.layout)[: args.limit]
    if not dirs:
        print(f"no {args.layout} fixtures in {set_dir}", file=sys.stderr)
        return 2
    from hearing.vision import faces as hfaces

    if not hfaces.weights_available():
        print("MediaPipe face weights missing; run .venv/bin/python -m hearing.vision.faces --download", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    report = evaluate(dirs, args.backends, iou_min=args.iou_min, params=params, fit=args.fit)
    report.update(
        stamp=datetime.now(UTC).astimezone().strftime("%Y%m%d-%H%M%S"),
        set=str(set_dir.name),
        layout=args.layout,
        wall_s=time.perf_counter() - t0,
        machine=device.describe(),
        command=" ".join(["python -m eval.asd_eval", *(argv if argv is not None else sys.argv[1:])]),
    )
    path = write_results(args.out, report)
    print(f"{len(dirs)} fixtures, {report['wall_s']:.1f} s; device {report['machine']['device']}; wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
