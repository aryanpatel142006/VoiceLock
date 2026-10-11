# Hard-negative triplet head for team voice profiles (T-281)

`eval/triplet_margin_head.py` is an offline research component for the Aryan
and Joseph profiles. It assumes a frozen ECAPA encoder produces one feature
vector per **0.5 s crop**; only a compact projection head is trained. The
module does not load audio, weights, or profile files and does not change the
live enrollment path.

## Method

`DynamicMarginTripletLoss` performs batch-hard mining. For every crop it
selects the closest crop belonging to another speaker by cosine similarity,
and the farthest same-speaker crop (equivalently, the largest positive
distance). The margin is:

```text
min(0.8, 0.2 + 0.4 * max(hard_negative_cosine, 0))
```

The dynamic term makes confusing cross-speaker crops pay a larger separation
penalty without making already-distant negatives dominate. A batch must contain
at least two crops for each of two speakers. `train_cross_entropy_baseline`
provides a matched projection-plus-classifier baseline using static
cross-entropy; both heads can be scored with `evaluate_separation`.

## Evaluation protocol

1. Split each speaker's consented profile recordings by time before embedding;
   keep the held-out crops disjoint from training crops.
2. Produce frozen ECAPA features for 0.5 s crops, with the same preprocessing
   for both methods.
3. Train the triplet and static-CE heads on the same crops and seed.
4. Build an enrolled prototype from training embeddings. On held-out positive
   crops and the other speaker's negative crops, report cosine separation
   margin, true-accept rate, and false-accept rate at the pre-declared
   threshold (calibrate it on development negatives only).

This worktree has no committed Aryan/Joseph audio or ECAPA weights, so no
real-profile comparison is claimed here. The unit tests use synthetic feature
vectors only and verify hard-negative selection, dynamic margins, output
normalization, and metric calculation. A future run should record device,
crop counts, threshold, separation margin, and false-accept rate for both
methods before considering promotion.

## Constraints and rollback

Derived embeddings are sensitive and must remain outside git. This module is
not wired into `hearing.tse.enrolled`; adoption would require a separately
versioned profile experiment and an audio/word-retention check. Removing the
research file has no live-pipeline effect.
