# SpeechBrain: second choice

Cloned `https://github.com/speechbrain/speechbrain.git` with `--depth 1` into
`/tmp/T-302-speechbrain`; read at `89ead74d163463d30c62329a09cfdb4c54f5abc1`.
Apache-2.0 in root LICENSE. No code copied for this task.

Exact borrow targets at that revision:
- `speechbrain/processing/PLDA_LDA.py:StatObject_SB.norm_stat1` (253–258):
  epsilon-clipped length normalization; `whiten_stat1` (270–326): centered
  covariance whitening. Read these methods, PLDA/LDA API and scoring signatures.
- `recipes/VoxCeleb/SpeakerRec/train_speaker_embeddings.py:SpeakerBrain.compute_forward`
  and `compute_objectives`: augmentation, feature normalization, encoder/classifier,
  label replication. Read complete training recipe.
- `recipes/VoxCeleb/SpeakerRec/hparams/train_ecapa_tdnn.yaml:embedding_model`
  and `compute_cost`: 192-d output and additive-angular-margin configuration.
  Read complete configuration.

Good fit to our existing SpeechBrain ECAPA, but the recipe updates the encoder;
it is not a ready-made frozen-vector head. PLDA needs statistics/trial containers;
copying the entire module is unnecessary for this smoke. Borrow normalization
only in a later train-only preprocessing ablation, with the mean persisted and
identically applied to enrollment/query. Never estimate it using held-out scenes.
