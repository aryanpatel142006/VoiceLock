"""Fit quality-aware P(same speaker) on calibration recordings, evaluate untouched test rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler


def fit(features, labels):
    features, labels = np.asarray(features), np.asarray(labels)
    if features.ndim != 2 or features.shape[1] != 3 or not np.isfinite(features).all():
        raise ValueError('features must be finite [score, duration_s, raw_embedding_norm]')
    if set(labels.tolist()) != {0, 1} or len(labels) != len(features):
        raise ValueError('need both binary classes')
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=1., max_iter=1000, random_state=0).fit(scaler.transform(features), labels)
    return {'features': ['score', 'duration_s', 'raw_embedding_norm'],
            'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'coef': model.coef_[0].tolist(), 'intercept': float(model.intercept_[0]),
            'probability_threshold': .5, 'training_prior': float(labels.mean())}


def save(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2) + '\n')


def predict(path, features):
    model = json.loads(path.read_text())
    x = (np.asarray(features) - model['mean']) / model['scale']
    return expit(x @ model['coef'] + model['intercept'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    with np.load(a.pairs, allow_pickle=False) as data:
        x, y, split = data['features'], data['labels'], data['split']
    if set(split.tolist()) != {'calibrate', 'test'}:
        p.error('pairs must contain calibration and held-out test splits')
    train, test = split == 'calibrate', split == 'test'
    save(fit(x[train], y[train]), a.out)
    prob = predict(a.out, x[test])
    fixed = x[test, 0] >= .7
    learned = prob >= .5
    result = {'test_pairs': int(test.sum()), 'brier': float(brier_score_loss(y[test], prob)),
              'log_loss': float(log_loss(y[test], prob, labels=[0, 1])),
              'fixed_0_7_error': float((fixed != y[test]).mean()),
              'calibrated_error': float((learned != y[test]).mean())}
    a.out.with_suffix('.metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
