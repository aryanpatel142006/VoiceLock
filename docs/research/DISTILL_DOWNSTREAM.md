# T-334: Distillation downstream gate (selector arms on identical streams)

Code: `eval/selector_gate/{gate,run}.py`, tests `tests/test_selector_gate.py`. Run:
`.venv/bin/python -m eval.selector_gate.run --out <dir> [--final]` (writes `frozen_config.json`, `result.json`, `results.jsonl` (Arena-shaped rows: candidate / dimensions / delivered / compute), `audio/<arm>/<case>.wav` = delivered audio with sha256).

## What is frozen across arms
Same per-window ECAPA embeddings (T-291 `eval/embeddings/T-291/team_slices.npz`, 1.0 s), same gallery (each speaker's `profile-*` recording), same EMA tracker (alpha 0.5, argmax, accept above threshold), same threshold rule (95th percentile of dev target-absent tracked scores, per arm space), same source streams. Only the embedding transform differs. `frozen_config.json` carries a sha256 written before any scoring. Final cases are reserved and only scored with `--final`.

## Arms
| Arm | Status |
|---|---|
| `raw_ecapa` | ok |
| `identity_head` | ok; bit-exact parity with raw (max abs score diff 0.0, identical selection) |
| `mahalanobis` | ok (shrinkage 0.3 whitening fit on gallery recordings only) |
| `projection_head` | ok, existing `models/team_projection_head.pt`; **dev-contaminated**: trained on these two speakers |
| `lda` | blocked: a two-speaker gallery gives a 1-D discriminant, no checkpoint |
| `distillation` | **blocked**: no checkpoint (`HEARING_DISTILL_CKPT` unset; R36/distill-student is not pushed). No result fabricated |

## Dev result (clean-recording stand-in, 2 target cases + 2 absent cases, 1 s windows)
Measured by the command above on laptop-a (CPU, isolated selector time ≈ 0.002–0.005 ms/window, load not controlled).

| Arm | top1 | margin | retention | absent false accept | wrong-speaker s | switches |
|---|---|---|---|---|---|---|
| raw_ecapa | 1.000 | 0.448 | 1.000 | 0.052 | 0 | 0 |
| identity_head | 1.000 | 0.448 | 1.000 | 0.052 | 0 | 0 |
| mahalanobis | 0.989 | 0.403 | 1.000 | 0.052 | 0 | 0 |
| projection_head (contaminated) | 1.000 | 1.574 (other space) | 1.000 | 0.052 | 0 | 0 |

Null by design: `recovery_ratio` (needs separated-vs-oracle SI-SDR), `unknown_false_accept` (no third-party stream), CI over groups when fewer than 2 groups. Absent false accept is 5% because the threshold is calibrated to that on the same dev absent cases (not a held-out number).

## Limits
- **These are NOT real separated streams.** No cached separator outputs from T-321/T-318 exist in the main checkout (searched `eval/results`, `models`, Arena dirs). The bank pairs clean team recordings as ideal "streams". It validates plumbing and parity; it says nothing about separator-induced degradation.
- Independent sources: 2 speakers, one target recording each; generalization claims are not allowed (`generalization_claim_allowed: false`). Margins are not comparable across spaces.
- **Blocker for the real A/B:** a frozen bank of separated stems per scene (selected/other stems WAV + scene manifest with target label) from T-321/T-318 (laptop-b `docs/research/ARENA_DECISION_GATES.md` is also unpushed), plus the distillation checkpoint. `gate.Case` takes any such bank; only `build_team_bank` needs a sibling loader.
