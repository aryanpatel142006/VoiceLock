# T-247 Separator fine-tuning vs frozen separator + selector (theory, no training run)

All memory/time numbers are **theoretical** (arithmetic from the 639 MB checkpoint in docs/MODELS.md, fp32 ⇒ ≈160 M params). Nothing was trained, downloaded or run on GPU. Prototype: `experiments/theory/T-247/plan.py`; tests: `tests/test_theory_separator_finetune.py`.

## Repo facts (inspected)
- The separator is the **unconditioned 2-stream** `MossFormer2_SS_16K` (24 MossFormer layers, dim 512, kernel 16, `num_spks=2`), loaded in `hearing/tse/blind.py::_load_separator` via `clearvoice.models.mossformer2_ss.mossformer2.MossFormer2_SS_16K(args).model`, then `load_state_dict(strict=True)` from `$HEARING_MODELS_DIR/MossFormer2_SS_16K/last_best_checkpoint`. Output is a list of 2 streams; the stream is *picked after* separation (ECAPA / p_speaking). Per orchestrator answer to Q-717, conditioning-embedding ablations do not apply.
- Hook for a pilot (outside this scope, not edited): a trainer builds the same `network` with the same `args`, loads the same state dict, freezes by parameter-name prefix, and saves a *separate* `.pt`; `_checkpoint()` is never repointed, so production cannot pick it up (no deploy).

## Arms
| Arm | Trainable | Est. VRAM (batch 8, 4 s) | Risk |
|---|---|---|---|
| A selector only (frozen separator, small head on ECAPA + stream features) | ≈0.3 M | ≈5.4 GB | none to separation; cannot fix bad separation |
| B adapters (bottleneck/LoRA on top 6 blocks) | ≈5 M | ≈5.5 GB | low |
| C unfreeze top 6 blocks + decoder | ≈40 M | ≈5.9 GB | medium forgetting |
| D full fine-tune | ≈160 M | ≈14.6 GB | high forgetting |
Activation memory is an assumed 1.5 GB/item (not measured); all arms fit one A100-40GB in theory. Recommendation: **A first (free, CPU-able), then B as the single GPU pilot**; C/D only if B passes and shows headroom. Rationale: T-200 shows the pick already matches the best stream 50/50, so the expected gain is in separation quality on our mics, which only B–D can change.

## Losses
`L = SI-SDR-PIT(streams, [target, other]) + 0.3·CE(stream-pick) + 0.2·absent-energy + 0.5·KD(frozen teacher output)`.
- SI-SDR with permutation alignment (PIT over 2 streams); on target-present mixes the pick head is trained on the PIT-chosen permutation.
- **Target-absent negatives (25 % of batches)**: mixtures of two non-target speakers; the target-selected output is penalised for energy above the input floor (absent-energy), so the model does not hallucinate the enrolled voice.
- **Target-word surrogate** (no ASR in the loop): only a *limit* – a spectral-envelope/energy-correlation term on voiced frames of weak targets; true WER is evaluated, never optimised, to avoid overfitting to a transcript model.
- **KD to the frozen model** (teacher = original checkpoint, same input) is the main catastrophic-forgetting control, plus lr ≤ 1e-5, weight decay, and 20 % public-speaker replay in every batch.

## Splits and eval
- Split by **source group** (speaker/session), seeded hash: `plan.py split <groups.txt> --seed 247`. Whole groups go to dev or heldout; the final heldout team-mic set is never used for tuning or early stopping (early stop on dev only).
- Eval: frozen baseline vs candidate on (1) heldout team-mic mixes, (2) heldout public speakers (forgetting check), (3) weakest-target tercile (target at lowest relative level; erasure check), (4) target-absent mixes, (5) wrong-speaker time. Counts of available inputs are not known here: **planned sizes are placeholders until a manifest from the data owners exists** (actual available counts: 0 inspected by this task).

## Weak-target erasure
Fine-tuning on SI-SDR tends to favour the louder talker. Mitigation: sample target-to-interferer levels uniformly over −10…+5 dB, report SI-SDRi per tercile, and gate on the weak tercile.

## A100 pilot recipe (do not launch from this task)
AdamW lr 1e-5 (adapter 1e-4), cosine, batch 8 × 4 s, fp16/bf16 autocast, grad clip 5, ≤3000 steps and ≤45 min hard stop, ckpt every 250 steps atomically (tmp + rename, config+gate hash in the file), resume skips verified ckpts, seed 247, one checkpoint kept, GPU cases counted against the global 300 cap (smoke 12 first). `python experiments/theory/T-247/plan.py manifest` emits config, config hash and estimate; the trainer itself is **not written** (debt).

## Gates (`plan.py gate metrics.json`)
Success (all): dev and heldout-team SI-SDRi ≥ +0.5 dB over frozen baseline; heldout public regress ≤ 0.3 dB; weak tercile drop ≤ 1.0 dB; target-absent output rise ≤ 3 dB; wrong-speaker ≤ 25 % (T-141 range 6–25 %).
Rollback (any hard fail: public regress, weak erasure, wrong-speaker): discard checkpoint, keep frozen `blind_assoc`. Otherwise **inconclusive**: no deploy, one retry at most with a smaller arm. Falsification: arm B fails to beat arm A+frozen by 0.5 dB on heldout team ⇒ stop fine-tuning, spend effort on selector/data.

## Consumer command
`.venv/bin/python experiments/theory/T-247/plan.py manifest` / `estimate` / `split groups.txt --seed 247` / `gate metrics.json`. Metrics JSON keys: `baseline`/`candidate` × `dev_si_sdri, heldout_team_si_sdri, heldout_public_si_sdri, weak_tercile_si_sdri, absent_out_db, wrong_speaker_pct`.
