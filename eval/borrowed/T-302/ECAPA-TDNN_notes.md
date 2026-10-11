# TaoRuijie/ECAPA-TDNN: third choice

Cloned `https://github.com/TaoRuijie/ECAPA-TDNN.git` with `--depth 1` into
`/tmp/T-302-ECAPA-TDNN`; read at `4904fda6c5da208998321e3ea75e24320bab8aad`.
MIT, copyright 2022 Tao Ruijie, verified in LICENSE.md. No code copied.

Exact targets at that revision:
- `ECAPAModel.py:train_network`: full-encoder Adam training;
  `eval_network`: full-utterance and five-crop normalized embeddings with
  averaged cosine scores. Read complete module.
- `trainECAPAModel.py`: `--initial_model` resume path and training/evaluation
  loop. Read complete script.
- `loss.py:AAMsoftmax`: angular-margin classifier candidate for later comparison.
  Its header credits clovaai/voxceleb_trainer; check that upstream license too
  before any future copy (none borrowed here).

Best as a compact fine-tuning reference, not as our frozen embedding adapter.
Hardcoded `.cuda()` paths conflict with our caller-selected CPU/MPS/CUDA device;
data augmentation requires MUSAN/RIR assets. The evaluation code uses deprecated
`numpy.float`. A future borrow must preserve MIT attribution, replace device/data
assumptions and split by recording before crops. Crops are correlated evidence,
not independent test recordings. Prefer the smaller WeSpeaker excerpt now.
