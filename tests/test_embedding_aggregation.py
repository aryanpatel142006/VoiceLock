"""T-311: Embedding aggregation unit tests (mean-then-normalize vs normalize-then-mean).

Documents and verifies the difference between:
1. Normalize-then-mean (current pattern in hearing/tse/enrolled.py:323-327 and hearing/server/namecall.py:648-649):
   Unit-normalizes each piece embedding first, then averages unit vectors, then normalizes.
   This equalizes weights across all segments, allowing noisy or low-energy segments to distort
   the aggregate direction on the unit sphere.
2. Mean-then-normalize (SOTA steal fix):
   Averages raw unnormalized embeddings first, then unit-normalizes once.
   Raw embedding magnitudes carry energy and confidence information; averaging before
   normalization naturally downweights low-energy noise segments and preserves true speaker geometry.
"""

from __future__ import annotations

import numpy as np

from hearing.tse.enroll import profile_clip
from hearing.tse.enrolled import (
    EnrolledCandidate,
    MockBackend,
)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_then_mean(raw_embeddings: list[np.ndarray]) -> np.ndarray:
    """Legacy pattern: normalize each embedding to unit length first, then average."""
    units = []
    for r in raw_embeddings:
        n = np.linalg.norm(r)
        units.append(r / n if n > 1e-12 else np.zeros_like(r))
    mean = np.mean(units, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-12 else mean


def mean_then_normalize(raw_embeddings: list[np.ndarray]) -> np.ndarray:
    """Recommended SOTA pattern: average raw embeddings first, then normalize."""
    mean = np.mean(raw_embeddings, axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 1e-12 else mean


def test_mean_then_normalize_downweights_noisy_pieces():
    """Synthetic test: mean-then-normalize stays significantly closer to the true
    speaker direction than normalize-then-mean when low-energy noisy pieces are present.
    """
    rng = np.random.default_rng(42)
    dim = 64
    true_dir = rng.standard_normal(dim)
    true_dir /= np.linalg.norm(true_dir)

    # 4 clean speech pieces: large raw magnitude (~10.0), tightly clustered around true_dir
    clean = [
        true_dir * 10.0 + rng.standard_normal(dim) * 0.2 for _ in range(4)
    ]
    # 4 noisy speech pieces: small raw magnitude (~0.5), random directions
    noisy = [rng.standard_normal(dim) * 0.5 for _ in range(4)]
    raws = clean + noisy

    old_emb = normalize_then_mean(raws)
    new_emb = mean_then_normalize(raws)

    cos_old = _cos(old_emb, true_dir)
    cos_new = _cos(new_emb, true_dir)

    # Mean-then-normalize should be substantially closer to the true direction
    assert cos_new > cos_old, (
        f"mean-then-normalize ({cos_new:.4f}) should beat normalize-then-mean ({cos_old:.4f})"
    )
    # The improvement should be statistically and practically significant (> 0.05)
    assert cos_new - cos_old > 0.05, (
        f"Expected gain > 0.05, got {cos_new - cos_old:.4f}"
    )


def test_equal_norms_mathematical_equivalence():
    """When all raw embeddings have identical magnitude, mean-then-normalize
    and normalize-then-mean are mathematically identical.
    """
    rng = np.random.default_rng(123)
    dim = 32
    # Generate vectors with different directions but exactly identical norm
    raws = []
    target_norm = 5.0
    for _ in range(6):
        v = rng.standard_normal(dim)
        v = v / np.linalg.norm(v) * target_norm
        raws.append(v)

    old_emb = normalize_then_mean(raws)
    new_emb = mean_then_normalize(raws)

    assert np.allclose(old_emb, new_emb, atol=1e-12), (
        "With equal norms, mean-then-normalize and normalize-then-mean must match bit-for-bit."
    )


def test_norm_collapse_on_unit_sphere():
    """Demonstrates norm shrinkage when unit vectors are averaged.
    When angular spread is present, averaging unit vectors produces a vector
    whose norm is strictly less than 1.
    """
    rng = np.random.default_rng(999)
    dim = 16
    # 5 unit vectors with substantial angular separation
    unit_vectors = []
    for _ in range(5):
        v = rng.standard_normal(dim)
        unit_vectors.append(v / np.linalg.norm(v))

    unnorm_mean = np.mean(unit_vectors, axis=0)
    norm = np.linalg.norm(unnorm_mean)

    # The norm of the mean of non-collinear unit vectors is strictly < 1.0
    assert norm < 0.99, f"Norm should shrink on unit sphere, got {norm}"


def test_adversarial_outlier_suppression():
    """Low-amplitude interfering segments (e.g. background noise/speaker)
    severely degrade normalize-then-mean (where unit normalization magnifies them
    to equal 1:1 weight with clean segments), whereas mean-then-normalize naturally
    suppresses them proportionally to their low energy.
    """
    rng = np.random.default_rng(77)
    dim = 32
    true_dir = rng.standard_normal(dim)
    true_dir /= np.linalg.norm(true_dir)

    # Construct an orthogonal interference direction
    rand_vec = rng.standard_normal(dim)
    orth_dir = rand_vec - float(np.dot(rand_vec, true_dir)) * true_dir
    orth_dir /= np.linalg.norm(orth_dir)

    # 2 clean pieces along true_dir with high amplitude (norm = 10.0)
    clean = [true_dir * 10.0 for _ in range(2)]
    # 2 quiet interference pieces along orth_dir with low amplitude (norm = 0.1)
    interfering = [orth_dir * 0.1 for _ in range(2)]
    raws = clean + interfering

    old_emb = normalize_then_mean(raws)
    new_emb = mean_then_normalize(raws)

    cos_old = _cos(old_emb, true_dir)
    cos_new = _cos(new_emb, true_dir)

    # normalize-then-mean: 2 unit vectors in true_dir + 2 unit vectors in orth_dir
    # -> resultant points at 45 degrees, cos ~ 1/sqrt(2) ~ 0.707
    assert cos_old < 0.75, f"normalize-then-mean should suffer from interference, got {cos_old}"

    # mean-then-normalize: 20 in true_dir vs 0.2 in orth_dir
    # -> cos ~ 20 / sqrt(20^2 + 0.2^2) ~ 0.99995
    assert cos_new > 0.999, f"mean-then-normalize should suppress interference, got {cos_new}"
    assert cos_new - cos_old > 0.25


def test_mock_backend_energy_weighting():
    """Integration test with MockBackend:
    MockBackend produces embedding magnitudes proportional to audio spectral energy.
    Clean loud audio + quiet noise shows clear separation between methods.
    """
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    clean = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    rng = np.random.default_rng(7)
    noisy = [(0.005 * rng.standard_normal(sr)).astype(np.float32) for _ in range(3)]
    pieces = [clean] + noisy

    backend = MockBackend()
    raws = [backend.embed(p) for p in pieces]

    old_emb = normalize_then_mean(raws)
    new_emb = mean_then_normalize(raws)

    clean_dir = backend.embed(clean)
    clean_dir /= np.linalg.norm(clean_dir)

    assert _cos(new_emb, clean_dir) > _cos(old_emb, clean_dir)
    assert _cos(new_emb, clean_dir) > 0.999
    assert _cos(old_emb, clean_dir) < 0.90


def test_enrolled_candidate_profile_piece_finding_documented():
    """Documents the finding at hearing/tse/enrolled.py:323-327.
    EnrolledCandidate._clip_embedding currently implements normalize-then-mean.
    This test verifies that the current live code behaves as documented.
    """
    sr = 16000
    t = np.arange(sr, dtype=np.float64) / sr
    clean = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    rng = np.random.default_rng(3)
    noisy = [(0.01 * rng.standard_normal(sr)).astype(np.float32) for _ in range(3)]
    pieces = [clean] + noisy
    clip = profile_clip(pieces)

    cand = EnrolledCandidate(embedder=MockBackend())
    got = cand._clip_embedding(clip, "target")

    backend = MockBackend()
    raws = [backend.embed(p) for p in pieces]
    expected_old = normalize_then_mean(raws)

    # Document that the existing codebase currently matches normalize-then-mean
    assert np.allclose(got, expected_old, atol=1e-6), (
        "Documenting finding: hearing/tse/enrolled.py:323-327 currently implements normalize-then-mean."
    )
