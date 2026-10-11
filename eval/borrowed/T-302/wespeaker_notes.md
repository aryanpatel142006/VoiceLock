# WeSpeaker: first choice

Cloned `https://github.com/wenet-e2e/wespeaker.git` with `--depth 1` into
`/tmp/T-302-wespeaker`; read at `9fecd6cb4f47475d01761d87c826298dff4ef18c`.
Apache-2.0 (`LICENSE`, plus copyright headers in selected files).

Exact targets at that revision:
- `wespeaker/ssl/models/simclr_wrapper.py:36-40`: MLP; `:54-85`: InfoNCE
  assembly. Read the complete wrapper. **Borrowed** both into projection.py.
- `wespeaker/utils/plda/two_cov_plda.py`: `PldaStats.add_samples`,
  `TwoCovPLDA.em_one_iter`, `get_output`, `log_likelihood_ratio`, `adapt`.
  Read complete module. Useful later for covariance-aware scores; brings scipy,
  h5py, Kaldi-style I/O and covariance inversion requirements.
- `wespeaker/utils/plda/plda_utils.py:norm_embeddings`: sqrt(d) normalization
  differs from unit cosine normalization; zero vectors lack an epsilon guard.
- `examples/voxceleb/v3/simclr/README.md`: recipe uses mean subtraction for scoring.

Selected excerpt removes encoder dependency and exposes cached-vector projection.
Do not claim upstream published speaker scores apply to this frozen-head adaptation.
Two-speaker data cannot supply broad between-speaker covariance evidence for PLDA;
rank/regularization and independent utterance holdouts would need explicit checks.
