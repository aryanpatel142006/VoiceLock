"""Offline causal EMA of speaker embeddings; never changes enrolled profiles.

Replay NPZ keys: embeddings [N,D], profiles [K,D], labels [N], elapsed_s [N].
Each replay is one stable, identity-associated stream, including speaker changes.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time

import numpy as np


def _unit(vector):
    x = np.asarray(vector, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError('expected a finite nonempty embedding vector')
    norm = np.linalg.norm(x)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('embedding norm must be finite and nonzero')
    return x / norm


class EMACropBuffer:
    """Bounded per-stream context. elapsed_s is time since the previous update.

    Call reset on target changes or separator identity discontinuities. Stream
    keys must be associated upstream, not raw permutation-unstable stem indices.
    """

    def __init__(self, half_life_s=1.0, max_gap_s=2.0, max_streams=2):
        if not all(math.isfinite(v) and v > 0 for v in (half_life_s, max_gap_s)):
            raise ValueError('half life and gap must be positive and finite')
        if not isinstance(max_streams, int) or max_streams < 1:
            raise ValueError('max_streams must be a positive integer')
        self.half_life_s = half_life_s
        self.max_gap_s = max_gap_s
        self.max_streams = max_streams
        self._state = {}

    def reset(self):
        self._state.clear()

    def update(self, stream_id, embedding, *, elapsed_s):
        if not math.isfinite(elapsed_s) or elapsed_s <= 0:
            raise ValueError('elapsed_s must be positive and finite')
        x = _unit(embedding)
        old = self._state.get(stream_id)
        if old is not None and old.shape != x.shape:
            raise ValueError('embedding dimension changed; reset first')
        if old is not None and elapsed_s <= self.max_gap_s:
            decay = math.exp2(-elapsed_s / self.half_life_s)
            mixed = decay * old + (1 - decay) * x
            # Exact cancellation carries no direction: reacquire the current crop.
            x = _unit(mixed) if np.linalg.norm(mixed) > 1e-12 else x
        if stream_id not in self._state and len(self._state) >= self.max_streams:
            raise ValueError('stream capacity reached; reset before reassociation')
        self._state[stream_id] = x.copy()
        return x.copy()


def evaluate_stream(embeddings, profiles, labels, elapsed_s, *, half_life_s=1.0):
    x = np.asarray([_unit(e) for e in embeddings])
    p = np.asarray([_unit(e) for e in profiles])
    y = np.asarray(labels)
    dt = np.asarray(elapsed_s, dtype=float)
    if (x.ndim != 2 or p.ndim != 2 or x.shape[1] != p.shape[1]
            or y.shape != (len(x),) or dt.shape != y.shape
            or not np.isfinite(dt).all() or np.any(dt <= 0)
            or not np.issubdtype(y.dtype, np.integer) or np.any(y < 0)
            or np.any(y >= len(p))):
        raise ValueError('invalid replay shapes, labels or times')
    b = EMACropBuffer(half_life_s=half_life_s)
    timings = []
    smooth = []
    for e, seconds in zip(x, dt, strict=True):
        start = time.perf_counter_ns()
        smooth.append(b.update(0, e, elapsed_s=float(seconds)))
        timings.append((time.perf_counter_ns() - start) / 1e6)
    result = {}
    for name, vectors in [('raw', x), ('ema', np.asarray(smooth))]:
        scores = vectors @ p.T
        picks = scores.argmax(axis=1)
        stable = y[1:] == y[:-1]
        transitions = np.flatnonzero(~stable) + 1
        delays = []
        for i in transitions:
            end = next((int(j) for j in transitions if j > i), len(y))
            hits = np.flatnonzero(picks[i:end] == y[i])
            delays.append(float(dt[i:i+hits[0]].sum()) if len(hits) else None)
        result[name] = {
            'accuracy': float(np.mean(picks == y)),
            'wrong_speaker_time_fraction': float(dt[picks != y].sum()/dt.sum()),
            'spurious_flips': int(np.sum((picks[1:] != picks[:-1]) & stable)),
            'mean_stable_cosine_step': float(np.mean(
                (1 - np.sum(vectors[1:]*vectors[:-1], axis=1))[stable]
            )) if stable.any() else None,
            'switch_delay_s': delays,
        }
    result['update_ms'] = {'p50': float(np.percentile(timings, 50)),
                           'p95': float(np.percentile(timings, 95))}
    result['device'] = 'numpy CPU'
    result['machine'] = platform.platform()
    result['half_life_s'] = half_life_s
    return result


def synthetic_replay(crop_s, seed=275):
    """Embedding-space noise surrogate, NOT measured ECAPA or noisy speech."""
    rng = np.random.default_rng(seed)
    profiles = np.zeros((2, 192))
    profiles[0, 0] = 1
    profiles[1, :2] = [0.6, 0.8]
    labels = np.repeat([0, 1, 0, 1], 80)
    embeddings = profiles[labels] + rng.normal(
        scale=0.45 / math.sqrt(crop_s), size=(len(labels), 192))
    return embeddings, profiles, labels, np.full(len(labels), crop_s)


def audio_replay(paths, embedder, crop_s, seed=275):
    """Clean 2.5s enrollment, then disjoint noisy crops; fixed A/B/A order.

    Uses already separated recordings and 0 dB white noise, not a separator
    benchmark. Keeps at most 12 crops per segment and never opens devices.
    """
    import soundfile as sf

    if not 0.5 <= crop_s <= 1.0 or len(paths) != 2:
        raise ValueError('provide two recordings and crop_s in [0.5, 1.0]')
    audios = []
    for path in paths:
        x, sr = sf.read(path, dtype='float32')
        if sr != 16000 or x.ndim != 1 or len(x) < 40000 + round(16000*crop_s):
            raise ValueError('recordings must be mono 16kHz with enrollment plus a full crop')
        audios.append(x)
    profiles = np.array([embedder.embed(x[:40000]) for x in audios])
    rng = np.random.default_rng(seed)
    n = round(16000*crop_s)
    xs, ys = [], []
    for label in (0, 1, 0):
        audio = audios[label]
        for i in range(40000, min(len(audio)-n+1, 40000+12*n), n):
            clean = audio[i:i+n]
            noise = rng.normal(size=n)
            noise *= np.sqrt(np.mean(clean**2))/np.sqrt(np.mean(noise**2))
            xs.append(embedder.embed(np.clip(clean+noise, -1, 1).astype('float32')))
            ys.append(label)
    return np.asarray(xs), profiles, np.asarray(ys), np.full(len(xs), crop_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', help='NPZ with ordered, labelled crop embeddings')
    parser.add_argument('--half-life', type=float, default=1.0)
    parser.add_argument('--audio', nargs=2, help='two separated mono 16kHz WAVs; local ECAPA only')
    args = parser.parse_args()
    if args.audio:
        from hearing import models
        from hearing.tse import enrolled

        backend = models.load(enrolled.MODEL)
        report = {'source': 'separated speech plus 0dB white noise', 'runs': {
            str(s): evaluate_stream(*audio_replay(args.audio, backend, s),
                                   half_life_s=args.half_life) for s in (0.5, 0.75, 1.0)}}
    elif args.replay:
        with np.load(args.replay, allow_pickle=False) as data:
            report = evaluate_stream(*(data[k] for k in
                ('embeddings', 'profiles', 'labels', 'elapsed_s')), half_life_s=args.half_life)
        report['source'] = args.replay
    else:
        report = {'source': 'synthetic embedding noise only', 'runs': {
            str(s): evaluate_stream(*synthetic_replay(s), half_life_s=args.half_life)
            for s in (0.5, 0.75, 1.0)}}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
