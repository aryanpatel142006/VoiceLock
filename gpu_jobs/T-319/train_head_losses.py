"""Offline frozen-embedding experiment; no audio, network, or production mutation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, x):
        return F.normalize(self.net(F.normalize(x, dim=-1)), dim=-1)


class AngularPrototypicalLoss(nn.Module):
    """Clean-room cosine-to-episode-centroid cross entropy; query excluded from centroid."""
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.)))

    def forward(self, episode):
        if episode.ndim != 3 or min(episode.shape[:2]) < 2:
            raise ValueError('need at least two speakers and two distinct crops')
        query = F.normalize(episode[:, 0], dim=-1)
        proto = F.normalize(episode[:, 1:].mean(1), dim=-1)
        logits = self.log_scale.exp().clamp(max=100) * (query @ proto.T)
        return F.cross_entropy(logits, torch.arange(len(query), device=query.device))


class SubCenterArcFace(nn.Module):
    """Max cosine over k centers/class, then monotonic angular target margin."""
    def __init__(self, dim, classes, k=3, margin=.3, scale=16.):
        super().__init__()
        if k < 1 or not 0 < margin < math.pi / 2 or scale <= 0:
            raise ValueError('invalid subcenter settings')
        self.weight = nn.Parameter(torch.randn(classes, k, dim))
        self.margin, self.scale = margin, scale

    def forward(self, z, labels):
        cos = torch.einsum('bd,ckd->bck', F.normalize(z, dim=-1),
                           F.normalize(self.weight, dim=-1)).max(-1).values
        cos = cos.clamp(-1 + 1e-6, 1 - 1e-6)
        phi = cos * math.cos(self.margin) - (1 - cos.square()).sqrt() * math.sin(self.margin)
        phi = torch.where(cos > math.cos(math.pi - self.margin), phi,
                          cos - math.sin(self.margin) * self.margin)
        mask = F.one_hot(labels, num_classes=cos.shape[1]).bool()
        return self.scale * torch.where(mask, phi, cos)


def load_data(path):
    """Flat NPZ only. Require split and original recording ID, including for augmentations."""
    with np.load(path, allow_pickle=False) as archive:
        aliases = {'x': ('x', 'embeddings'), 'speaker': ('speaker', 'labels', 'y'),
                   'duration': ('duration', 'duration_s', 'crop_duration_s'),
                   'source_id': ('source_id', 'recording_id'), 'split': ('split',)}
        data = {}
        for key, options in aliases.items():
            found = next((name for name in options if name in archive), None)
            if found is None:
                raise ValueError(f'missing {key}; see README schema (no inferred splits)')
            data[key] = archive[found]
    x = np.asarray(data['x'], dtype=np.float32)
    n = len(x)
    if x.ndim != 2 or n == 0 or not np.isfinite(x).all() or (np.linalg.norm(x, axis=1) <= 0).any():
        raise ValueError('x must be a finite nonzero flat embedding matrix')
    if any(data[key].shape != (n,) for key in aliases if key != 'x'):
        raise ValueError('metadata must have one value per embedding')
    if not np.isfinite(data['duration']).all() or (data['duration'] <= 0).any():
        raise ValueError('duration must be positive finite seconds')
    data['x'] = x
    for key in ('speaker', 'split', 'source_id'):
        data[key] = data[key].astype(str)
    required = {'train', 'enroll', 'calibrate', 'test'}
    if set(data['split']) != required:
        raise ValueError('split must contain exactly train/enroll/calibrate/test')
    for source in np.unique(data['source_id']):
        if len(np.unique(data['split'][data['source_id'] == source])) != 1:
            raise ValueError(f'source {source} crosses splits; augmentation leakage')
    speakers = set(data['speaker'][data['split'] == 'enroll'])
    if len(speakers) < 2:
        raise ValueError('need enrollment for at least two speakers')
    for split in ('calibrate', 'test'):
        if not set(data['speaker'][data['split'] == split]) <= speakers:
            raise ValueError(f'{split} speaker lacks enrollment')
    return data


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def scores(enroll, ey, query, qy):
    speakers = np.unique(ey)
    protos = unit(np.stack([unit(enroll[ey == s]).mean(0) for s in speakers]))
    cos = unit(query) @ protos.T
    own = np.array([np.flatnonzero(speakers == s)[0] for s in qy])
    return cos, own


def metrics(enroll, ey, query, qy):
    cos, own = scores(enroll, ey, query, qy)
    other = cos.copy()
    other[np.arange(len(query)), own] = -np.inf
    margin = cos[np.arange(len(query)), own] - other.max(1)
    return {'pair_margin': float(margin.mean()), 'margin_p05': float(np.percentile(margin, 5)),
            'flip_rate': float((margin <= 0).mean()), 'queries': len(query)}


def make_synthetic(path, seed=309):
    """Explicit stand-in when the externally named T-309 artifact is unavailable."""
    rng = np.random.default_rng(seed)
    a = unit(rng.normal(size=(1, 192)))[0]
    r = rng.normal(size=192)
    r = unit((r - r.dot(a) * a)[None])[0]
    centers = [a, .85 * a + math.sqrt(1 - .85**2) * r]
    rows = {k: [] for k in ('x', 'speaker', 'duration', 'source_id', 'split')}
    for split, count in [('train', 96), ('enroll', 24), ('calibrate', 48), ('test', 48)]:
        for speaker, center in enumerate(centers):
            for i in range(count):
                duration = (.5, 1., 2.)[i % 3]
                x = center + .12 / math.sqrt(duration) * rng.normal(size=192)
                for key, value in zip(rows, (x, str(speaker), duration, f'{split}-{speaker}-{i//8}', split), strict=True):
                    rows[key].append(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in rows.items()})


def run(args):
    torch.set_num_threads(args.threads)
    data = load_data(args.data)
    device = args.device
    if device == 'auto':
        from hearing.device import pick
        device = pick()
    masks = {s: data['split'] == s for s in ('train', 'enroll', 'calibrate', 'test')}
    x = torch.from_numpy(data['x'][masks['train']]).to(device)
    names, y = np.unique(data['speaker'][masks['train']], return_inverse=True)
    groups = [np.flatnonzero(y == i) for i in range(len(names))]
    if len(groups) < 2 or any(len(g) < args.crops for g in groups):
        raise ValueError('training needs >=2 speakers with --crops distinct samples each')
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'device': device, 'host': platform.node(), 'seed': args.seed,
              'steps': args.steps, 'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'protocol': 'recording-disjoint closed-set; flip = nonpositive prototype margin', 'models': {}}
    for method in ('raw', 'ap', 'subcenter', 'ce'):
        start = time.perf_counter()
        head = None
        if method != 'raw':
            torch.manual_seed(args.seed)
            rng = np.random.default_rng(args.seed)
            head = Head(x.shape[1]).to(device)
            criterion = (AngularPrototypicalLoss() if method == 'ap' else
                         SubCenterArcFace(32, len(names), args.k) if method == 'subcenter' else
                         nn.Linear(32, len(names))).to(device)
            opt = torch.optim.Adam([*head.parameters(), *criterion.parameters()], lr=args.lr)
            for _ in range(args.steps):
                chosen = rng.choice(len(groups), min(args.speakers, len(groups)), replace=False)
                indices = np.stack([rng.choice(groups[s], args.crops, replace=False) for s in chosen])
                idx = torch.as_tensor(indices.ravel(), device=device)
                labels = torch.as_tensor(y[indices.ravel()], device=device)
                z = head(x[idx])
                loss = (criterion(z.reshape(len(chosen), args.crops, -1)) if method == 'ap' else
                        F.cross_entropy(criterion(z, labels) if method == 'subcenter' else criterion(z), labels))
                opt.zero_grad()
                loss.backward()
                opt.step()
            head.cpu().eval()
            torch.save({'head': head.state_dict(), 'loss_layer': criterion.cpu().state_dict(),
                        'input_dim': x.shape[1], 'output_dim': 32, 'classes': names.tolist(),
                        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}},
                       args.out / f'{method}.pt')
        with torch.no_grad():
            embedded = unit(data['x']) if head is None else head(torch.from_numpy(data['x'])).numpy()
        en, ey = embedded[masks['enroll']], data['speaker'][masks['enroll']]
        res = metrics(en, ey, embedded[masks['test']], data['speaker'][masks['test']])
        res['wall_s'] = time.perf_counter() - start
        report['models'][method] = res
        pair_rows = {'features': [], 'labels': [], 'split': []}
        for split in ('calibrate', 'test'):
            mask = masks[split]
            cos, own = scores(en, ey, embedded[mask], data['speaker'][mask])
            for i in range(len(cos)):
                for j in range(cos.shape[1]):
                    pair_rows['features'].append([cos[i, j], data['duration'][mask][i],
                                                  np.linalg.norm(data['x'][mask][i])])
                    pair_rows['labels'].append(int(j == own[i]))
                    pair_rows['split'].append(split)
        np.savez_compressed(args.out / f'{method}_pairs.npz', **pair_rows)
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--make-synthetic', action='store_true')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--speakers', type=int, default=32)
    p.add_argument('--crops', type=int, default=4)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    if min(args.steps, args.threads, args.speakers) < 1 or args.crops < 2 or args.speakers < 2:
        p.error('steps/threads positive; speakers/crops >=2 required')
    if args.make_synthetic:
        if args.data.exists():
            p.error('refusing to overwrite existing data with synthetic')
        make_synthetic(args.data)
    run(args)


if __name__ == '__main__':
    main()
