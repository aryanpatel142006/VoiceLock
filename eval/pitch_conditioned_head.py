"""Offline 192+2 -> 64 -> 32 speaker metric; no live-path registration."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import correlate, find_peaks
from torch import nn
from torch.nn import functional as F

from hearing.audio.io import resample
from hearing.device import pick


def acoustic_features(audio: np.ndarray) -> np.ndarray:
    """Median voiced autocorrelation F0 and magnitude centroid at 16 kHz.

    40 ms Hann frames / 20 ms hop; F0 range 65-400 Hz, periodicity >= .6.
    Unvoiced frames contribute to centroid, but not F0. Silence returns zeros.
    Centroid is a spectral-envelope proxy, NOT an estimated formant frequency.
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim != 1 or len(x) < 640 or not np.isfinite(x).all():
        raise ValueError('expected finite mono audio with at least 40 ms at 16 kHz')
    peak = np.max(np.abs(x))
    if peak == 0:
        return np.zeros(2, dtype=np.float32)
    x = x / peak  # independent of recording gain; no identity from loudness
    pitches, centroids = [], []
    for start in range(0, len(x) - 639, 320):
        frame = x[start:start + 640].copy()
        frame -= frame.mean()
        if np.mean(frame ** 2) < 1e-6:
            continue
        windowed = frame * np.hanning(640)
        spectrum = np.abs(np.fft.rfft(windowed))
        centroids.append(float(spectrum @ np.fft.rfftfreq(640, 1 / 16000) / spectrum.sum()))
        ac = correlate(frame, frame, method='fft')[639:]
        # Normalize by overlap energies to avoid bias toward short lags.
        lags = np.arange(40, 247)
        energy = np.array([np.linalg.norm(frame[:-lag]) * np.linalg.norm(frame[lag:])
                           for lag in lags])
        scores = ac[lags] / np.maximum(energy, 1e-12)
        peaks, _ = find_peaks(scores)
        if len(peaks):
            best = peaks[np.argmax(scores[peaks])]
            if scores[best] >= 0.6:
                pitches.append(16000 / lags[best])
    return np.array([np.median(pitches) if pitches else 0,
                     np.median(centroids) if centroids else 0], dtype=np.float32)


def validate_manifest(records: list[dict]) -> list[dict]:
    """Reject same resolved path or decoded audio hash across file-level splits."""
    checked, seen = [], {}
    for record in records:
        if record['split'] not in ('train', 'test') or record['speaker'] not in ('aryan', 'joseph'):
            raise ValueError('split must be train/test and speaker aryan/joseph')
        path = Path(record['path']).resolve(strict=True)
        audio, sr = sf.read(path, dtype='float32')
        if audio.ndim != 1 or not np.isfinite(audio).all() or np.max(np.abs(audio), initial=0) > 1:
            raise ValueError(f'{path}: expected finite mono audio in [-1,1]')
        audio = resample(audio, sr, 16000) if sr != 16000 else audio
        digest = hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()
        for key in (str(path), digest):
            if key in seen:
                raise ValueError(f'duplicate/overlap audio: {path}')
            seen[key] = record['split']
        checked.append({**record, 'path': str(path), 'sha256_16k': digest, 'audio': audio})
    for split in ('train', 'test'):
        if {r['speaker'] for r in checked if r['split'] == split} != {'aryan', 'joseph'}:
            raise ValueError(f'{split} requires both speakers')
    return checked


class PitchConditionedHead(nn.Module):
    """Train-only acoustic standardization persisted with learned weights."""
    def __init__(self):
        super().__init__()
        self.register_buffer('feature_mean', torch.zeros(2))
        self.register_buffer('feature_std', torch.ones(2))
        self.net = nn.Sequential(nn.Linear(194, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, embeddings, features):
        emb = F.normalize(embeddings, dim=-1)
        # Balance two side features against a unit-norm 192-d embedding.
        side = ((torch.log1p(features) - self.feature_mean) / self.feature_std).clamp(-5, 5)
        return F.normalize(self.net(torch.cat((emb, side), dim=-1)), dim=-1)


def fit_head(embeddings, features, labels, *, epochs=120, device=None, seed=280):
    """Frozen inputs; full-batch supervised pairwise cosine metric learning."""
    if epochs < 1:
        raise ValueError('epochs must be positive')
    torch.manual_seed(seed)
    dev = device or pick()
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=dev)
    a = torch.as_tensor(features, dtype=torch.float32, device=dev)
    y = torch.as_tensor(labels, dtype=torch.long, device=dev)
    if (x.ndim != 2 or x.shape[1] != 192 or a.shape != (len(x), 2)
            or y.shape != (len(x),) or set(y.tolist()) != {0, 1}
            or any(int((y == i).sum()) < 2 for i in (0, 1))
            or not torch.isfinite(x).all() or not torch.isfinite(a).all() or (a < 0).any()):
        raise ValueError('need finite Nx192 embeddings, nonnegative Nx2 features, >=2 rows per speaker')
    model = PitchConditionedHead().to(dev)
    with torch.no_grad():
        model.feature_mean.copy_(torch.log1p(a).mean(0))
        model.feature_std.copy_(torch.log1p(a).std(0).clamp_min(0.05))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    same = y[:, None] == y[None, :]
    offdiag = ~torch.eye(len(y), dtype=torch.bool, device=dev)
    losses = []
    started = time.perf_counter()
    for _ in range(epochs):
        z = model(x, a)
        sim = z @ z.T
        loss = (1 - sim[same & offdiag]).mean() + F.relu(sim[~same] - 0.1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    return model, {'initial_loss': losses[0], 'final_loss': losses[-1], 'epochs': epochs,
                   'device': str(dev), 'train_ms': (time.perf_counter() - started) * 1000}


def project(model, embeddings, features):
    dev = next(model.parameters()).device
    with torch.inference_mode():
        return model(torch.as_tensor(embeddings, dtype=torch.float32, device=dev),
                     torch.as_tensor(features, dtype=torch.float32, device=dev)).cpu().numpy()


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def run_experiment(records, *, output: Path, crop_s=0.5, epochs=120, gains_db=(-12, 0, 6)):
    """Train on full nonoverlapping crops; test held-out files at safe digital gains."""
    from hearing import models
    from hearing.tse import enrolled

    if not 0.04 <= crop_s < 1:
        raise ValueError('crop_s must be >=40 ms and sub-second')
    checked = validate_manifest(records)  # fail before loading weights
    backend = models.load(enrolled.MODEL)  # local-only; never silently use mock
    n = round(crop_s * 16000)
    extraction_ms = []

    def extract(split, gain_db):
        vectors, side, labels = [], [], []
        for record in checked:
            if record['split'] != split:
                continue
            count = 0
            for start in range(0, len(record['audio']) - n + 1, n):
                crop = record['audio'][start:start + n] * 10 ** (gain_db / 20)
                if np.max(np.abs(crop)) > 1:
                    raise ValueError('gain would clip; lower gains or use quieter source files')
                if np.sqrt(np.mean(crop ** 2)) < 1e-5:
                    continue
                t = time.perf_counter()
                vector = np.asarray(backend.embed(crop)).reshape(-1)
                if vector.shape != (192,) or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                    raise ValueError('expected real nonzero 192-d ECAPA embedding')
                vectors.append(vector)
                side.append(acoustic_features(crop))
                extraction_ms.append((time.perf_counter() - t) * 1000)
                labels.append(0 if record['speaker'] == 'aryan' else 1)
                count += 1
            if count < 2:
                raise ValueError(f"{record['path']}: need >=2 non-silent complete crops")
        return np.asarray(vectors), np.asarray(side), np.asarray(labels)

    x, a, y = extract('train', 0)
    model, training = fit_head(x, a, y, epochs=epochs)
    z = project(model, x, a)
    centers = _unit(np.stack([z[y == i].mean(0) for i in (0, 1)]))
    raw_centers = _unit(np.stack([_unit(x[y == i]).mean(0) for i in (0, 1)]))
    conditions = []
    for gain in gains_db:
        xt, at, yt = extract('test', gain)
        t = time.perf_counter()
        projected_scores = project(model, xt, at) @ centers.T
        project_ms = (time.perf_counter() - t) * 1000 / len(xt)
        raw_scores = _unit(xt) @ raw_centers.T
        row = {'gain_db': gain, 'crops': len(yt), 'projection_ms_per_crop': project_ms}
        for name, scores in (('raw_ecapa', raw_scores), ('pitch_head', projected_scores)):
            predicted = scores.argmax(1)
            row[name] = {speaker: {'n': int((yt == i).sum()),
                                   'error_rate': float((predicted[yt == i] != i).mean()),
                                   'mean_margin': float((scores[yt == i, i] - scores[yt == i, 1-i]).mean())}
                         for i, speaker in enumerate(('aryan', 'joseph'))}
        conditions.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), output.with_suffix('.pt'))
    result = {'experiment': 'T-280', 'crop_s': crop_s, 'seed': 280, 'training': training,
              'encoder_device': backend.device, 'host': platform.node(), 'conditions': conditions,
              'feature_embedding_ms_p50': float(np.median(extraction_ms)),
              'feature_embedding_ms_p95': float(np.percentile(extraction_ms, 95)),
              'rtf_p50': float(np.median(extraction_ms) / (crop_s * 1000)),
              'files': [{k: v for k, v in r.items() if k != 'audio'} for r in checked],
              'limitations': 'Digital gain only; closed-set two-speaker identification, no room/noise/unknown-speaker proof.'}
    output.write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, help='JSON list: split, speaker, absolute WAV path')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--crop-s', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--gains-db', type=float, nargs='+', default=[-12, 0, 6])
    args = parser.parse_args()
    result = run_experiment(json.loads(args.manifest.read_text()), output=args.output,
                            crop_s=args.crop_s, epochs=args.epochs, gains_db=args.gains_db)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
