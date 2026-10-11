# Borrowed speaker-embedding adaptation (T-302)

Decision: vendor the small WeSpeaker SimCLR MLP plus InfoNCE assembly as an
**offline candidate** in `third_party/borrowed/T-302/projection.py`. Default
192→64→32 dimensions match cached ECAPA vectors and T-273's output space.
No live matcher, profile format, Stage contract or runtime default changes.

## Sources evaluated

Primary source code was inspected on 2026-10-10. Top three were shallow-cloned
outside the repo, with exact targets and revisions recorded in
`eval/borrowed/T-302/*_notes.md`. Licenses below concern code, not datasets/weights.

| Source | Borrow target | License/evaluation | Outcome |
|---|---|---|---|
| [WeSpeaker](https://github.com/wenet-e2e/wespeaker/blob/9fecd6cb4f47475d01761d87c826298dff4ef18c/wespeaker/ssl/models/simclr_wrapper.py) | SimCLR MLP and positive-first InfoNCE | Apache-2.0; tiny torch-only excerpt | **Vendored/adapted** |
| [WeSpeaker PLDA](https://github.com/wenet-e2e/wespeaker/blob/9fecd6cb4f47475d01761d87c826298dff4ef18c/wespeaker/utils/plda/two_cov_plda.py) | TwoCovPLDA EM, likelihood ratio, covariance adaptation | Apache-2.0; full covariances and Kaldi I/O | Defer until broader speaker statistics exist |
| [SpeechBrain](https://github.com/speechbrain/speechbrain/blob/89ead74d163463d30c62329a09cfdb4c54f5abc1/speechbrain/processing/PLDA_LDA.py) | epsilon-safe norm, whitening, PLDA | Apache-2.0; training recipe also exposes 192-d ECAPA/AAM | Useful normalization ablation; no copy |
| [TaoRuijie ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN/blob/4904fda6c5da208998321e3ea75e24320bab8aad/ECAPAModel.py) | train_network, eval_network, initial-model fine-tuning | MIT; CUDA assumptions and augmentation assets | Full-encoder option; too much machinery for this task |
| [Kaldi](https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/plda.cc) | GetNormalizationFactor, TransformIvector, LogLikelihoodRatio, PldaUnsupervisedAdaptor | Apache-2.0 header; C++ dependencies | Reference for count-aware scoring; no copy |
| [NeuralPlda](https://github.com/iiscleap/NeuralPlda/blob/ac99bb4f19ca598dedff7a9f01ca3c81665d060d/utils/models.py) | NeuralPlda.extract_plda_embeddings and quadratic pair scoring | No LICENSE/COPYING file found in cloned tree; permission not established | Research-only; **do not vendor** without permission |

NeuralPlda was additionally cloned into `/tmp/T-302-NeuralPlda`; its model methods
were inspected. Kaldi's source header/methods were retrieved directly, not cloned.
The top-three notes describe what was read, dependencies and exact future targets.

## Adaptation and smoke

Kept Linear/ReLU/Linear and normalized similarity/diagonal-removal/positive-first
loss assembly. Detached the audio encoder, changed square layers to 192→64→32,
added unit output, device-local labels and shape/batch/temperature validation.
Included upstream copyright, full LICENSE and modification details in ATTRIBUTION.md.
Torch is already in the `ml` extra; no dependencies installed or added.

Run from repo root:

```sh
.venv/bin/python -m pytest -q tests/test_borrowed_embedding.py
.venv/bin/python eval/borrowed/T-302/smoke.py
```

Seven tests check random output shape/unit norm, finite nonzero training gradients,
batch-one equivalence, hand-derived orthogonal-pair logits, pairing sensitivity
and invalid inputs. A separate comparison against the pinned upstream wrapper
produced bit-exact logits/targets for seeded random two-view vectors (batch 4,
dimension 32, temperature 0.07). No audio, weights, network or hardware needed. The smoke CLI
compares compute only. This path is not packaged into the hearing wheel.

## Explicit comparison with T-273

Read `eval/projection_head_pilot.py:TeamProjectionHead`, `ArcFaceLoss` and
`ContrastiveProjectionLoss`. Both heads are 192→64→32 with final L2 normalization.
T-273 uses LayerNorm/GELU; borrowed uses ReLU without LayerNorm.

| Property | Borrowed | T-273 |
|---|---|---|
| Trainable parameters | 14,432 | 14,560 |
| Batch-one CPU p50 / p95 | 0.016542 / 0.018333 ms | 0.019000 / 0.022000 ms |
| Included objective | Two-view instance InfoNCE | Speaker-labeled contrastive or ArcFace |
| Repeated same-speaker examples | False negatives unless sampling/loss is changed | Labels explicitly identify same speaker |

Measured locally: macOS arm64, CPU, torch 2.14.1, float32 random embeddings,
seed 302, one thread, 100 warmup calls, 1,000 timed calls per head, inference mode.
These microseconds exclude ECAPA extraction and audio buffering; no RTF, SI-SDRi,
WER or speaker accuracy was measured. Small timing differences are not robust
performance rankings. **Different, slightly cheaper in this smoke, not proven
better in verification.** T-273 is better aligned to two known speakers because
its objectives use speaker labels; vanilla InfoNCE can incorrectly repel clips
of the same speaker. T-273's trained checkpoint cannot load into this head due
to the changed layer structure. Keep original embeddings/profiles for rollback.

Upstream SSL recipes remove projection modules for embedding extraction; retaining
our 32-d head at inference is an experimental adaptation, not upstream validation.
Before promotion: compare identical cached train/holdout splits with T-273, using
speaker-aware loss/sampling; hold out whole recordings/scenes before crops; measure
target-absent false accepts, wrong-speaker flips, target retention and blind listening.
Never infer improvement from random vectors or reuse current cosine thresholds
without calibration. Training, centering statistics and checkpoints are not delivered.

Model inventory row for reviewer / muse-c:

| Model | Role | Weights | License | Status |
|---|---|---|---|---|
| T-302 adapted WeSpeaker projection head | Frozen ECAPA 192-d → 32-d research projection | None; random initialization only | Apache-2.0 | Offline candidate, not deployed |
