# Isolation oracle diagnostic (T-200)

Offline, optional track. Tool: `eval/isolation_diagnostic.py` (no live pipeline, registry or server change).
Core demo defaults untouched. The principal case is two distinct male voices (Aryan / Joseph), both target roles, quieter target.

## What it does
Replays a labelled two-stem scene through the live window loop (window 2.0 s, hop 1.0 s, triangular overlap-add, `EnrolledCandidate` with the real MossFormer2_SS + ECAPA/LDA choice, a recording separator keeps both streams). The clean target picks the **oracle** stream for evaluation only. Each target-active window is classified by the first cause that fits: `deadline` (compute > hop, raw-mix fallback), `separation` (oracle stream < 4 dB SI-SDR: target lost before selection), `selector` (actual < oracle - 3 dB), `mix_back` (final mixer/AGC < chosen stream - 3 dB), `ok`. Target-absent windows report output-energy leak vs the mix in dB, never SI-SDR against zero. Table rows say ORACLE or deployable and carry n windows and sample rate (16 kHz). The manifest JSON (`$HEARING_RESULTS_DIR/isolation_diagnostic/<stamp>.json`) holds inputs' sha256 prefixes, config, per-window records (compute ms, choice, margin, floor, held/paused/scoring, skipped/raw-fallback counts).

```
.venv/bin/python -m eval.isolation_diagnostic --dry-run          # fake models, classifier check only
.venv/bin/python -m eval.isolation_diagnostic --scene <scene dir> --profiles <dir>/<name>/profile-*.wav \
    --target-name aryan --other-name joseph --ratios 0 -5 -10 -15 --roles both
```
Missing stems/profiles exit 2 with `BLOCKED: missing inputs: ...`. Dry-run output is labelled FAKE and is not measured separation (tests: `tests/test_isolation_diagnostic.py`: separator-missing-target, wrong-selector, mix-back-leak, missed-deadline each classified).

## Measured (2026-10-09, laptop-a, MPS, scene `t141/aryan/demo-team-same`, profiles `profiles/{aryan,joseph}/profile-01.wav`, heldout = second half of windows, `--ratios 0 -10`, both roles)
Run once, not repeated, machine under other swarm load (compute figures are upper bounds).

| condition | oracle SI-SDR dB | actual choice | final mixer | overlap windows (final) | n windows | oracle==choice | missed/raw/skipped | compute ms mean/max |
|---|---|---|---|---|---|---|---|---|
| role A, 0 dB | 15.1 | 15.1 | 15.2 | 15.1 | 15 | 1.00 | 0/0/0 | 446/604 |
| role A, -10 dB | 12.7 | 12.7 | 12.9 | 11.5 | 15 | 1.00 | 0/0/0 | 397/404 |
| role B, 0 dB | 18.2 | 18.2 | 18.4 | 18.4 | 10 | 1.00 | 0/0/2 | 405/414 |
| role B, -10 dB | 13.8 | 13.8 | 14.0 | 14.0 | 10 | 1.00 | 0/0/2 | 400/404 |

Target-absent leak: -49.7 dB vs mix (role B, 5 windows). The 2 "skipped" in role B are target-absent windows the selector silenced; no target-active window was silenced.
Embedding probes (clean target stem crop, margin = cos(target profile) - cos(other profile)): crop 0.5 s 0.42, 1 s 0.55, 2 s 0.69, 4 s 0.77; other-voice contamination of a 2 s crop 0% 0.69, 10% 0.50, 30% 0.40, 50% 0.26; profile vs the scene's clean target cos 0.78, vs the other speaker 0.15; target-profile vs other-profile cos 0.066, so the cosine > 0.7 negative-profile rejection did **not** fire.

## Mechanisms: demonstrated / ruled out / unknown
- **Separation losing the target before selection**: ruled out on this scene, both roles, 0 and -10 dB (oracle 12.7-18.2 dB; zero `separation` windows). Not tested: -5/-15 dB, other scenes, noise.
- **Wrong selector (ECAPA/LDA)**: ruled out on this scene (oracle==choice in 100% of 50 windows). Actual-vs-oracle gap 0.0 dB. Not shown to be fine on the other demo-team scenes, real-room mics, or T-141's overlap (6-25% wrong-speaker time in earlier work).
- **Short crops**: demonstrated as a margin loss (0.77 at 4 s to 0.42 at 0.5 s) but margins stay positive.
- **Contamination**: demonstrated: margin falls from 0.69 to 0.26 at 50% other-voice share, still positive on clean stems.
- **Negative-profile rejection at cos > 0.7**: ruled out for this pair (cos 0.066). Saved negatives bypassing the gender guard is not exercised here.
- **Mix-back / AGC**: ruled out with the proxy mixer (`est + (1-s)(mix-est)`, s=1, +0.1-0.2 dB). The real `hearing.fusion.mixer` and AGC were not run; use `--agc` for the proxy only.
- **Deadline**: not a cause at hop 1.0 s (compute max 0.6 s on MPS under load); would become one if compute exceeds the hop, which the table counts as raw fallback.
- **Profile vs demo-mic mismatch**: not demonstrated. Profile-to-stem cos 0.78 is on synthetic-room stems, not the demo mic; needs profiles and scenes recorded through the same demo mic (T-172 re-enrol) to test.
- **Strict missing-camera pause decisions, same_bucket, profile_negative/readiness/active tier logging, T-069 male/male ceiling gap**: **unknown** here. The tool records `paused/held/scoring` per window but the replay passes no `asd`, and the scene used is mixed-gender-agnostic stems from the team set; `same_bucket` and tier state live in the wrapper/session, not offline. The T-069 gap (oracle minus actual) is reported as `ceiling_gap_db` in the manifest and was 0.0 dB for this scene; T-069's own male/male figure was not reproduced.

## Interpretation and next measured intervention
On clean stems with the saved profiles, embeddings work; "no wrong-voice output" is not the same as preserved target words, so use the SI-SDR / oracle columns, not wrong-speaker counts, for target loss. Where live runs fail, the likely missing pieces are inputs this tool has not been given: demo-mic profiles/stems, the room noise, the strict camera pause path. Intervention to measure next: run `--noise --ratios 0 -5 -10 -15 --roles both` on the demo-mic re-enrolled profiles and on `demo-team-hard`; if `selector` windows appear, lengthen the crop to >= 2 s before embedding (margin 0.69 vs 0.42 at 0.5 s) and re-measure.

## Limitations
One scene and one run; the `-5/-15` ratios, `--noise` and the other scenes are implemented but not run. Ratios scale the target stem relative to the other stem (whole-clip RMS). Fake dry-run results prove the classifier, not separation. No weights, audio or results are committed.
