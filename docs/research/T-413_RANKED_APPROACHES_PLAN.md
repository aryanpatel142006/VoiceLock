# T-413: response to Ahmad's ranked-approaches report (data prep, identity gate, claim)

Written Oct 10 2026 on laptop-a by the T-413 worker.

**I could not read the report itself.** `/home/ahmad/selective-hearing/docs/ranked-approaches.md` is a
path on Ahmad's machine. It does not exist on laptop-a (macOS), and no copy is in the repo. Everything
below works from the summary in the T-413 task text: the top-5 list, the hard requirements and the new
recordings. If the full report changes any of this (citations, the stop-doing list, the build-sketch
details), please commit it to `docs/research/ranked-approaches.md` so every agent can read it.

## Claim

**T-413 claims approach 4 (tougher training: asymmetric SIR + noisy enrollments), together with
the prototype of approach 1 (identity bouncer).** Approach 4's dataloader core is
`eval/sir_mixtures.py::render_item` / `draw_train_sir`. Approach 1 is `hearing/tse/identity_gate.py`.
Approaches 2, 3, 5 and 8 are unclaimed by this task.

## (b) Data-prep plan: 2-minute recordings → SIR-binned mixtures

Code: `eval/sir_mixtures.py` (tests: `tests/test_sir_mixtures.py`).

**Input.** `<root>/<speaker>/profile-*.wav`, takes of 60 s or longer (pattern and minimum are flags).
The default root is the team fixtures. Today that holds 3 voices: ahmad (2 × 120 s), aryan
(6 × 120 s) and joseph (6 × 120 s). Takes under 60 s (`profile-01` of aryan/joseph) are skipped. New
speakers just add a directory.

**Region ledger** (`python -m eval.sir_mixtures plan`, measured on the team takes):

| speaker | enroll | train | val | test |
|---|---|---|---|---|
| ahmad | 30.0 s | 164.0 s | 20.5 s | 20.5 s |
| aryan | 30.0 s | 541.6 s | 67.7 s | 67.7 s |
| joseph | 30.0 s | 541.6 s | 67.7 s | 67.7 s |

1. **Enrollment split.** The first 30 s of each speaker's first take is enrollment-only. It never
   appears inside a mixture, so a test item cannot carry its own enrollment audio.
2. **Time split.** The rest of every take is cut into contiguous train / val / test regions
   (80 / 10 / 10), with a 1 s guard between regions. Test audio is unseen *audio* of a seen voice.
   Speakers with several takes contribute test audio from every take, so test is cross-session
   relative to the enrollment.
3. **Speaker split.** `--holdout NAME…` sends all of a speaker's audio to `test_unseen` (an
   unseen-voice eval). With 3 voices the default is none. With "lots" of speakers, hold out at least 20 %.
4. **Loudness levelling.** SIR is set on the **active speech level**: mean power over 20 ms frames
   within 25 dB of the 90th-percentile frame, an ITU-T P.56-style approximation. Pauses do not bias
   it, and neither does whole-clip RMS. `sir_db = level(target) − level(interferer)`. Measured SIR
   matches the request within 0.05 dB (test). There is no K-weighting (`pyloudnorm` is not installed);
   both stems are speech, so K-weighting would barely move the difference.
5. **SIR bins.**
   - Eval uses **exact** bins: 0 dB (required), −5 dB (good) and −15 dB (aspirational), with no
     jitter, `--per-bin` items each, targets cycled over the speakers.
   - Training draws an **asymmetric** bin: +5 dB (15 %), 0 dB (30 %), −5 dB (30 %), −10 dB (15 %),
     −15 dB (10 %), plus ±2.5 dB jitter. That puts 70 % or more of items at or below 0 dB, so the
     target is usually the quieter talker (approach 4).
6. **Noisy enrollment** (approach 4). `enroll_snr_db` mixes a different speaker's enrollment audio,
   or a noise file if one is given, into the enrollment crop at 5–20 dB. It is on for training and off
   for eval unless `--noisy-enroll-db` is set.
7. **Crops.** Mixtures are 4 s and enrollments 10 s. Each crop needs 50 % or more of its frames
   active (relative to its whole take's level), or it takes the best of 20 draws. The eval mix sits at
   −26 dBFS active level and training at −35…−18 dBFS. Peaks are capped at 0.99 with the same gain on
   every stem, so `mix == target + interferer + noise` exactly.
8. **Output.** `python -m eval.sir_mixtures render --split test --per-bin 20` writes
   `$HEARING_FIXTURES_DIR/sir_mix_test/<sir±N_k>/{mix,target,interferer,noise,enroll}.wav` +
   `manifest.json`, plus `index.json`. This is the `eval/fixtures/mix.py` layout, so `eval/bench.py`-style
   readers work. `source` records takes, offsets, active shares, the measured SIR and the enrollment's
   take, offset and noise. Everything is deterministic from `--seed`.
9. **Training hookup.** A training `IterableDataset` calls
   `render_item(corpus, "train", draw_train_sir(rng), rng, enroll_snr_db=rng.uniform(*ENROLL_SNR_DB),
   noise=cafe)` per item. That is a dataloader-only change to `experiments/gpu/T-391/train_sep.py`,
   whose `render()` today draws SIR uniformly from −6…+6 dB (or the T-387 `--sir` range), with no
   enrollment and no asymmetry.
10. **Confusion metric.** Every item carries `target_id` / `interferer_id` and both clean stems.
    "Wrong speaker" means the output is closer, in active level after projection or in identity, to the
    interferer stem than to the target stem. It is reported per SIR bin.

Not done here: café noise / RIRs inside `render_item` (pass `noise=`; `GRUNT_NOISE_RIRS.md` has
sources), more speakers, and a fixture render to disk (one command, above).

## (c) Identity gate (approach 1): prototype and first numbers

Code: `hearing/tse/identity_gate.py` (tests: `tests/test_identity_gate.py`). It has three parts:

- `IdentityGate.score(window)`: ECAPA cosine of the output window to the enrollment and to the
  other enrolled voices. It passes when `cos ≥ ACCEPT_COS` and `cos − best_other ≥ MARGIN_COS`.
  Windows quieter than −55 dBFS pass as `silent`, because they carry no identity.
- `bounce(mix, extract, gate, retries=1)`: extract, check, and on a mismatch re-extract from
  `mix − output`. A wrong lock leaves mostly the target in the residual. If the retry also fails, it
  **blocks** and plays the mix 20 dB down (the T-180 gated-window floor), never the rejected voice.
  This fits windowed extractors.
- `IdentityGateStream.process(out_chunk, mix_chunk)`: the causal per-10 ms-chunk version for a
  streaming engine (approach 2 or 3). It re-scores the last 1.5 s every 0.25 s and closes after
  1 failed score, reopens after 2 passes, with a 20 ms gain ramp. It **adds no latency**; a wrong
  voice is heard for up to about one hop plus embed time once the window is dominated by it.
  Retry-on-residual needs a second engine instance on the residual stream; that is not built.

Identity stays a gate: nothing here is differentiable, and nothing feeds a training loss.

**Measurement** (`swarm-lock --shared -- .venv/bin/python -m eval.identity_gate_eval --per-bin 30`;
ECAPA `spkrec-ecapa-voxceleb` on MPS, laptop-a, 63 s for 8 640 windows; raw rows are in
`$HEARING_RESULTS_DIR/T-413_identity_gate.json`):
- Items are 30 per bin on the team test regions at 0, −5 and −15 dB.
- Outputs are simulated with ground truth: perfect `target`; full wrong lock `interferer`;
  `ext20` / `ext10` (the target with the interferer, at its in-mix level, suppressed 20 / 10 dB);
  `wrong20` (locked on the interferer, target −20 dB).
- Truth is "target dominant", so `ext10` at −15 dB counts as wrong.
- Closed set: the other two team voices are the "others".

| window | accept / margin | clean target blocked | wrong lock passed | all outputs: false block / false pass | EER open / margin-only |
|---|---|---|---|---|---|
| 1.0 s | 0.25 / 0.0 | 17.5 % | 2.1 % | 30.3 % / 6.2 % | 18.4 % / 14.6 % |
| **1.5 s** | **0.25 / 0.0 (shipped)** | **3.9 %** | **3.3 %** | **17.3 % / 7.9 %** | 14.6 % / 11.4 % |
| 1.5 s | 0.30 / 0.05 | 10.0 % | 1.1 % | 25.0 % / 5.0 % | |
| 2.0 s | 0.25 / 0.0 | 2.2 % | 1.9 % | 13.8 % / 6.7 % | 11.9 % / 8.3 % |

Per bin at 1.5 s (closed set, 0.30 / 0.05 run): a full wrong lock was blocked in 100 % of windows at
0, −5 and −15 dB. `ext20` passed 80 % at 0 dB but only 53 % at −15 dB. At −15 dB even a 20 dB
suppressor leaves the interferer level with the target, and ECAPA's cosine drops (0.35 against 0.48
clean).

What this means:
- **The bouncer reliably catches a full wrong-speaker lock at every SIR.** That is the failure that
  matters at 0 dB.
- **Leaky outputs are its weak spot.** At −15 dB it cannot tell "mostly target with a loud leak" from
  "mostly interferer".
- **Windows under 1 s are too noisy** to gate on (71 % false block at 0.5 s).
- **The thresholds are fitted on 3 voices**, one of them with 205 s of audio. Re-fit them once more
  speakers arrive (`eval.identity_gate_eval` takes `--root`).
- **Not measured yet:** the bouncer on a real extractor's output (the fixed-slot v3 separator or the
  enrolled path), and live latency of the stream gate. ECAPA on 1.5 s every 0.25 s is the cost to
  measure under `swarm-lock --exclusive` before it goes live.

## Next steps, in order

1. Run `IdentityGateStream` on the v3 fixed-slot separator's replay output (T-391/T-407 features) and
   count the wrong-speaker seconds it removes against the target seconds it mutes. Keep it behind a
   knob until it wins by T-180's adoption rule.
2. Wire `render_item` into `train_sep.py` as `--data sir_mixtures --asym` and run an A/B of uniform
   −6…+6 dB against the asymmetric bins on the A100. Score with the confusion metric at 0, −5 and
   −15 dB.
3. Re-fit the gate thresholds once the new 2-minute recordings land under
   `eval/fixtures/generated/team/<name>/`.
