# Voice embeddings and learning a selected person

> Research note, October 9, 2026. This is a test plan for personalization, not a claim that the current build learns from every conversation. Keep the HackRU stage contract in `PLAN.md`.

## What the current idea means

A speaker encoder turns a voiced sample into a fixed-size vector, or **embedding**, intended to represent speaker characteristics across different words. The current ECAPA-based matching path can compare a saved speaker embedding with separated candidate speech. An embedding is an identity clue: it does not itself remove café noise or force a blind separator to output only that speaker. [ECAPA-TDNN](https://www.isca-archive.org/interspeech_2020/desplanques20_interspeech.html) is a primary reference for fixed-length speaker representations.

The product's “gets better over time” claim should mean a measurable, reversible change: with the speaker's consent, add clean, user-confirmed examples; compare a new profile against the original on **different** recordings; keep it only if target detection improves without increasing wrong-speaker leakage or delay. Do not update from the system's own uncertain output. This protects against gradually learning the other person after a false lock.

## Smallest credible progression

1. **Baseline:** preserve the original enrollment audio, encoder/version and embedding. Measure target-present accuracy, target-absent false accepts, wrong-speaker time, leakage and reacquisition delay on held-out mic/room/script conditions.
2. **More confirmed examples:** save separate embeddings from a few clean utterances in different speaking conditions. Test a simple normalized mean or multiple-prototype match against the original single profile. Plot a **learning curve** with 1, 2, 4 and 8 confirmed examples while keeping a separate holdout fixed. A 2025 [multiple-enrollment aggregation study](https://badap.agh.edu.pl/publikacja/162911) compares simple and attention-based aggregation; its reported gains are evidence to test aggregation, not a guaranteed gain in this system.
3. **Quality and identity gate:** only accept user-confirmed examples with sufficient voiced duration and no obvious overlap/clipping. Compare against known non-target profiles. Keep a versioned anchor and a one-tap rollback. Consent and deletion controls apply to both audio and derived embeddings.
4. **Controlled update:** promote a new profile only when held-out target recognition improves **and** target-absent and wrong-person metrics do not regress. Record profile version, training examples, device and test set. If the new data merely matches a louder/closer microphone, it has not proved speaker learning.
5. **Later extractor study:** compare the present blind-separate-then-select chain against a model conditioned on the target's enrollment voice. [SpeakerBeam](https://arxiv.org/abs/2001.08378) illustrates this approach and also notes confusion for speakers with similar characteristics. It needs a fair accuracy, naturalness and latency test on our actual hardware before adoption.

## What research says about rapid calibration

[Look Once to Hear](https://arxiv.org/abs/2405.06289) demonstrates target-speaker enrollment from a short *noisy* example and subsequent extraction in a hearable setup. Its binaural hardware and training setup differ from this project's single demo microphone, so it supports investigating a tap-to-calibrate flow but does not prove our current backend can do it. The first local test is a tap during the intended person's speech, a clean/ambiguous input decision, a visible lock, a target-silent negative case and Reset. If it fails, the stage uses pre-enrolled profiles honestly.

## Experiments worth assigning after the demo gate

| Comparison | What changes | Required evidence |
|---|---|---|
| One profile vs 2, 4 and 8 confirmed examples | Embedding aggregation only | Held-out target-present and target-absent scores, wrong-speaker seconds, delay and learning curve; same separator and mixer |
| Original matcher vs calibrated thresholds | Verification rule only | Separate threshold sweep for each real target, with quiet-target and loud-wrong-speaker cases |
| Frozen embeddings vs small projection head | Learned mapping only | Distinct training/holdout voices or utterances, hard negatives and rollback; reject if gains vanish outside training scenes |
| Separate-then-select vs conditioned extractor | Extraction architecture | Same recordings, mic, output level and latency budget; blind human quality check |

None of these experiments warrants saying “the app learns people” until a profile update is implemented, consented and shown to improve held-out performance. A truthful near-term line is: **“It uses a saved voice profile to keep track of the person you chose; we are testing how confirmed examples can make that profile better over time.”**


## What has already been tried, and why the chain remains unreliable — October 9 follow-up

Fable already implemented the ECAPA post-separation matcher and saved multi-piece profiles, with cosine/negative contrast, optional closed-set LDA, F0 and overlap continuity. Embeddings are tried and partially successful; the benchmark record includes synthetic M/F accuracy .945 and useful pre-AGC Mac separation. This is evidence for tested conditions, not a room-wide guarantee. The repo/task record inspected here does not establish a completed speaker-conditioned REAL-TSE trial or a trained adaptive extractor.

**Known code behavior:** enrollment never enters the blind MossFormer2 separation network; it only helps choose among its two outputs. If neither output preserves target words, no matcher can restore them. Gated confidence-zero output can mix back half the input through the confidence residual floor. Late/skipped windows use the delayed mixture. Strict pause logic can interpret an all-NaN ASD vector as no finite speaking evidence; T-129's unmerged standard/strict A/B supports a fix opportunity on one clip but nearly ties on the real team clip. T-169's single-AGC fix is already in inspected main, so preserve that work and measure the remaining gain path rather than reintroducing a duplicate fix.

**Unresolved hypotheses:** short/noisy output embeddings, enrollment-microphone mismatch, stand-in confidence calibration, noisy F0 veto, and the `SAME_VOICE=.7` rule discarding the negative profile. Measure duration, similarities, negative-active state, decision margin, pause/gate reason, residual floor and actual hop/late counters per window. Cache reuse has not been demonstrated to be the root cause.

T-200's clean-stem oracle diagnoses separation versus selection, with pre/post-mixer and pre/post-AGC comparisons. T-201 evaluates the existing multi-example proposal on independent utterances. Improving selection is promising only when the separator preserves the desired speech; otherwise the architectural alternative is enrollment-conditioned extraction, with supported pretrained fine-tuning as a conditional later experiment. Do not score success only with ECAPA: manual target-word retention and absent-target false accepts prevent circular proof or gains caused by deletion. The optional task scopes and preserved core priority are in `TODO.md`; primary-source research and the fresh-agent diagnostic request are in `docs/research/DEEP_RESEARCH_HANDOFF.md`.


**Confirmed research target (Joseph, October 9):** strongly separate either of the actual two male speakers with different voices, at equal and unequal levels and swapped positions. Male/female was an illustrative example, not a new participant or demo requirement. Preserve the working cafe-first core if this bonus does not pass.

Male/male T-069 is the relevant prior failure: pooled accuracy .742 and wrong voice 1.75 s/10 s at 1 s context, despite a .930 candidate ceiling; 2 s reaches .804/1.34 s with .951 ceiling, still NO-GO. This is evidence for a substantial selection gap in that synthetic test, alongside unrecoverable windows. Same-bucket routing must be checked, but current saved target+negative profile sessions bypass that guard (`Session`, `enroller.profile_negative`); do not assume all male/male scenes were skipped.
