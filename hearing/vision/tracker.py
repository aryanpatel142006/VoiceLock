"""IoU face tracker with stable ``track_id``s through short occlusions.

Pure numpy (scipy's Hungarian solver is used when importable, otherwise a greedy
IoU matcher). Turns per-frame :class:`Detection`s into
:class:`hearing.types.FaceTrack`s whose ids survive ``max_missing_s`` of
missed detections, so the target selector can keep holding a face that the
detector dropped for a few frames.

Boxes are normalised ``(x, y, w, h)`` in [0, 1] (top-left origin), as produced by
the detector adapter and consumed by the console (see ``docs/CONTRACTS.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hearing.types import FaceTrack

try:  # Hungarian matching if scipy is already installed; never a hard dependency.
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:  # pragma: no cover - depends on the environment
    _lsa = None

Box = tuple[float, float, float, float]


@dataclass
class Detection:
    """One face found in one frame."""

    box: Box  # Normalised x, y, w, h.
    score: float = 1.0
    landmarks: np.ndarray | None = None


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes ``a`` (N×4) and ``b`` (M×4), both ``x, y, w, h``."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[None, :, 0], b[None, :, 1]
    bx2, by2 = bx1 + b[None, :, 2], by1 + b[None, :, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0.0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0.0, None)
    inter = iw * ih
    area_a = (a[:, 2:3] * a[:, 3:4]).clip(0.0, None)
    area_b = (b[None, :, 2] * b[None, :, 3]).clip(0.0, None)
    union = area_a + area_b - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0.0, inter / union, 0.0)
    return iou


def _greedy_assign(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Greedy matching by descending IoU; each row/column used at most once."""
    pairs: list[tuple[int, int]] = []
    if iou.size == 0:
        return pairs
    order = np.argsort(iou, axis=None)[::-1]
    used_r: set[int] = set()
    used_c: set[int] = set()
    n_cols = iou.shape[1]
    for flat in order:
        r, c = divmod(int(flat), n_cols)
        if iou[r, c] < threshold:
            break
        if r in used_r or c in used_c:
            continue
        pairs.append((r, c))
        used_r.add(r)
        used_c.add(c)
    return pairs


def _hungarian_assign(iou: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Optimal (max total IoU) matching via scipy, then drop pairs below threshold."""
    if iou.size == 0:
        return []
    rows, cols = _lsa(-iou)
    return [(int(r), int(c)) for r, c in zip(rows, cols) if iou[r, c] >= threshold]


@dataclass
class _Track:
    track_id: int
    raw: np.ndarray  # Last matched box (x, y, w, h), unsmoothed.
    smooth: np.ndarray  # EMA-smoothed box reported to consumers.
    velocity: np.ndarray  # Box delta per second (x, y, w, h), EMA-smoothed.
    landmarks: np.ndarray | None
    last_seen_ts: float
    hits: int = 1
    confirmed: bool = False
    score: float = 0.0
    _box_cache: Box | None = field(default=None, repr=False)

    def predicted(self, ts: float) -> np.ndarray:
        """Constant-velocity guess of where this face is at ``ts``."""
        dt = max(ts - self.last_seen_ts, 0.0)
        return self.raw + self.velocity * dt

    def box(self) -> Box:
        if self._box_cache is None:
            s = self.smooth
            self._box_cache = (float(s[0]), float(s[1]), float(s[2]), float(s[3]))
        return self._box_cache


class IouTracker:
    """Associate per-frame face detections into tracks with stable ids.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU between a detection and a track's predicted box for a match.
    max_missing_s:
        A track that has not been matched for this long is dropped. Until then it
        is still returned (box held at its last smoothed position) so downstream
        can keep the target through brief detector drop-outs or occlusions.
    min_hits:
        Consecutive matched frames before a new track is reported, suppressing
        one-frame false positives. Tentative tracks die on their first miss.
    smoothing_alpha:
        EMA weight of the newest box (1.0 = no smoothing).
    use_hungarian:
        Use scipy's optimal assignment when available; otherwise greedy by IoU.
    centre_gate:
        Second pass (T-107): a confirmed track and a detection left unmatched by IoU (the face moved
        or the box jumped during a miss) still match when their centres are within this many face
        widths and their sizes within 2x, so a re-detection keeps its id instead of becoming a new
        track. 0 disables it.
    reattach_s:
        T-119: a confirmed track dropped after ``max_missing_s`` is remembered this much longer. A new
        detection whose centre is within ``centre_gate`` face widths of a remembered track (sizes
        within 2x) takes that track's id back, confirmed at once, instead of a fresh id. The first
        live /judge session saw the target's id change 8 times for one or two people (face at the
        frame's bottom edge, detector misses of 1-1.8 s, re-detections within 0.04 of the old
        centre); each new id can restart enrollment and selection. 0 disables it.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missing_s: float = 0.7,
        min_hits: int = 2,
        smoothing_alpha: float = 0.5,
        use_hungarian: bool | None = None,
        centre_gate: float = 0.75,
        reattach_s: float = 2.0,
    ) -> None:
        self.centre_gate = float(centre_gate)
        if reattach_s < 0.0:
            raise ValueError("reattach_s must be >= 0")
        self.reattach_s = float(reattach_s)
        self._lost: list[_Track] = []  # dropped confirmed tracks, newest last (reattach_s)
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_missing_s < 0.0:
            raise ValueError("max_missing_s must be >= 0")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if not 0.0 < smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in (0, 1]")
        self.iou_threshold = float(iou_threshold)
        self.max_missing_s = float(max_missing_s)
        self.min_hits = int(min_hits)
        self.smoothing_alpha = float(smoothing_alpha)
        if use_hungarian is None:
            use_hungarian = _lsa is not None
        if use_hungarian and _lsa is None:
            raise RuntimeError("scipy is not installed; cannot use Hungarian matching")
        self.use_hungarian = bool(use_hungarian)
        self._velocity_alpha = 0.5
        self._tracks: list[_Track] = []
        self._next_id = 1
        self._last_ts: float | None = None

    # ------------------------------------------------------------------ public
    @property
    def tracks(self) -> list[FaceTrack]:
        """Currently confirmed tracks (same as the last ``update`` result)."""
        return [self._to_face_track(t) for t in self._tracks if t.confirmed]

    def reset(self) -> None:
        """Forget every track and restart ids from 1."""
        self._tracks = []
        self._lost = []
        self._next_id = 1
        self._last_ts = None

    def update(self, detections: list[Detection], ts: float) -> list[FaceTrack]:
        """Ingest one frame's detections at capture time ``ts`` (seconds).

        Returns every confirmed track, including those missing for less than
        ``max_missing_s``; ``last_seen_ts`` is the time of the last real match.
        """
        ts = float(ts)
        if self._last_ts is not None and ts < self._last_ts:
            # Out-of-order frame: do not let dt go negative, treat as same instant.
            ts = self._last_ts
        self._last_ts = ts

        dets = np.asarray([d.box for d in detections], dtype=np.float64).reshape(-1, 4)
        tracks = self._tracks
        matched_t: set[int] = set()
        matched_d: set[int] = set()

        if tracks and len(dets):
            predicted = np.stack([t.predicted(ts) for t in tracks])
            held = np.stack([t.raw for t in tracks])
            iou = np.maximum(iou_matrix(predicted, dets), iou_matrix(held, dets))
            pairs = (
                _hungarian_assign(iou, self.iou_threshold)
                if self.use_hungarian
                else _greedy_assign(iou, self.iou_threshold)
            )
            for ti, di in pairs:
                self._apply_match(tracks[ti], detections[di], dets[di], ts)
                matched_t.add(ti)
                matched_d.add(di)

        if self.centre_gate > 0.0:
            self._match_centres(tracks, detections, dets, matched_t, matched_d, ts)

        # Age unmatched tracks; drop stale or still-tentative ones.
        survivors: list[_Track] = []
        for i, t in enumerate(tracks):
            held = t.confirmed and (ts - t.last_seen_ts) <= self.max_missing_s
            # Dropped: tentative track missed a frame, or confirmed track too old.
            if i in matched_t or held:
                survivors.append(t)
            elif t.confirmed and self.reattach_s > 0.0:
                self._lost.append(t)
        horizon = self.max_missing_s + self.reattach_s
        self._lost = [t for t in self._lost if ts - t.last_seen_ts <= horizon]

        # Spawn tentative tracks for unmatched detections.
        for di, det in enumerate(detections):
            if di in matched_d:
                continue
            box = dets[di]
            old = self._reattach(box)
            if old is not None:
                self._apply_match(old, det, box, ts)
                old.confirmed = True
                old.smooth = box.copy()  # the face may have moved during the gap: no EMA from there
                old.velocity = np.zeros(4)
                old._box_cache = None
                survivors.append(old)
                continue
            t = _Track(
                track_id=self._next_id,
                raw=box.copy(),
                smooth=box.copy(),
                velocity=np.zeros(4),
                landmarks=det.landmarks,
                last_seen_ts=ts,
                hits=1,
                confirmed=self.min_hits <= 1,
                score=float(det.score),
            )
            self._next_id += 1
            survivors.append(t)

        self._tracks = survivors
        return [self._to_face_track(t) for t in survivors if t.confirmed]

    # ----------------------------------------------------------------- private
    def _reattach(self, box: np.ndarray) -> _Track | None:
        """The remembered (dropped) track nearest ``box`` within the centre gate, taken off the list."""
        if not self._lost or self.centre_gate <= 0.0:
            return None
        dx, dy, dw, dh = box
        best, best_d = None, float("inf")
        for t in self._lost:
            tx, ty, tw, th = t.raw
            ratio = (dw * dh) / max(tw * th, 1e-9)
            d = float(np.hypot(dx + dw / 2 - tx - tw / 2, dy + dh / 2 - ty - th / 2))
            if 0.5 <= ratio <= 2.0 and d <= self.centre_gate * max(tw, dw) and d < best_d:
                best, best_d = t, d
        if best is not None:
            self._lost.remove(best)
        return best

    def _match_centres(self, tracks: list[_Track], detections: list[Detection], dets: np.ndarray,
                       matched_t: set[int], matched_d: set[int], ts: float) -> None:
        """Greedy nearest-centre matching of confirmed, still-held tracks to leftover detections."""
        pairs = []
        for ti, t in enumerate(tracks):
            if ti in matched_t or not t.confirmed or ts - t.last_seen_ts > self.max_missing_s:
                continue
            tx, ty, tw, th = t.raw
            for di in range(len(dets)):
                if di in matched_d:
                    continue
                dx, dy, dw, dh = dets[di]
                ratio = (dw * dh) / max(tw * th, 1e-9)
                d = float(np.hypot(dx + dw / 2 - tx - tw / 2, dy + dh / 2 - ty - th / 2))
                if 0.5 <= ratio <= 2.0 and d <= self.centre_gate * max(tw, dw):
                    pairs.append((d, ti, di))
        for _d, ti, di in sorted(pairs):
            if ti in matched_t or di in matched_d:
                continue
            self._apply_match(tracks[ti], detections[di], dets[di], ts)
            matched_t.add(ti)
            matched_d.add(di)

    def _apply_match(self, t: _Track, det: Detection, box: np.ndarray, ts: float) -> None:
        dt = ts - t.last_seen_ts
        if dt > 0.0:
            v = (box - t.raw) / dt
            a = self._velocity_alpha
            t.velocity = a * v + (1.0 - a) * t.velocity
        t.raw = box.copy()
        a = self.smoothing_alpha
        t.smooth = a * box + (1.0 - a) * t.smooth
        t._box_cache = None
        t.landmarks = det.landmarks
        t.score = float(det.score)
        t.last_seen_ts = ts
        t.hits += 1
        if t.hits >= self.min_hits:
            t.confirmed = True

    @staticmethod
    def _to_face_track(t: _Track) -> FaceTrack:
        return FaceTrack(
            track_id=t.track_id,
            box=t.box(),
            landmarks=t.landmarks,
            p_speaking=0.0,  # Filled in by active-speaker detection (T-016).
            last_seen_ts=t.last_seen_ts,
        )
