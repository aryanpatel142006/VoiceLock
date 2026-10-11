# SepFormer WHAMR16k vs MossFormer2_SS: offline separator comparison (T-202)

Offline evaluation only. Nothing here touches the live pipeline, the stage registry, `config/`, the server or the Sunday demo (T-208 gate: no new live model). The current separator (`hearing.tse.blind.BlindSeparator`, ClearerVoice MossFormer2_SS_16K) is used unchanged.

## Release condition
The task allowed this trial if T-200's diagnostic showed that neither current candidate preserves the quiet target after output-path ablation, **or** if the orchestrator approved a bounded comparison. T-200 (`docs/research/ISOLATION_DIAGNOSTIC.md`) ruled out separation and selector loss for MossFormer2 on `demo-team-same`, so the first condition was **not** met. The trial was dispatched anyway, so it ran as a bounded comparison: one candidate, two 30-minute caps, no installs.

## Candidate and provenance
| field | value |
|---|---|
| repo | `speechbrain/sepformer-whamr16k` (official SpeechBrain checkpoint) |
| pinned revision | `21a5b500c6f52fddc387c5d9e5fb13ffd6f039c5` (HF commit, lastModified 2024-02-20) |
| licence | Apache-2.0 (model card front matter, checked by `provenance()`); the WHAMR! training data has its own licence |
| cache | `$HEARING_MODELS_DIR/sepformer-whamr16k-<revision>/`, separate from MossFormer2's; only `hyperparams.yaml`, `encoder/decoder/masknet.ckpt`, `README.md`, `test_mixture16k.wav` downloaded |
| sha256[:16] | hyperparams.yaml `eeef8d1450e1e064`, encoder.ckpt `31d23d395a408b88`, decoder.ckpt `2d959cb46de5b15f`, masknet.ckpt `e5fb8c690668e5d1` |
| config | `sample_rate: 16000`, `num_spks: 2` (hyperparams.yaml) |
| runtime | speechbrain 1.1.1, torch 2.14.1, already in the shared `.venv` (`speechbrain>=1.0` in `pyproject.toml`); nothing installed |
| two-output check | bundled `test_mixture16k.wav` (33 088 samples, mono 16 kHz) → output shape `(2, 33088)`, finite |
| device | `mps` for both candidates; CPU vs MPS on the same two 2 s windows: per-stream correlation 0.99999999, oracle SI-SDR identical to 1e-5 dB (not an MPS numerical fault) |
| load / startup | `from_hparams` 0.16 s with weights on local disk; 0.21 s including the two-output check (warm OS file cache; a cold first load is not measured) |
| peak accelerator memory | `torch.mps.driver_allocated_memory()` for the whole process (driver pool for every model loaded, not SepFormer alone): run 5 measured 10 010 MB after the MossFormer2 pass and 22 266 MB after the SepFormer pass; run 3 (SepFormer only) measured 24 556 MB. Most of the growth comes from separating the 50 s enrolment profile; see "Costs" |

The published WHAMR! figure (13.5 dB SI-SNRi, model card) is **not** a local result and is not used anywhere below.

## Method
`python -m eval.sepformer_trial --scene <scene> --profiles <dir> --target-name aryan --other-name joseph --ratios 0 -10 --roles both --clip-seconds 12 [--candidates ...]`

- **Same inputs for both candidates.** Scene `t141/aryan/demo-team-same` (manifest sha `09400e6af0d97e5d`, Aryan vs Joseph team voices, male/male: the principal case). Profiles `profiles/{aryan,joseph}/profile-01.wav` (sha `ce3d9ac29754b5e1` / `d16205ba02b75685`). The `mix_sha` of each condition is identical across both candidates and both runs: roleA 0 dB `e004d1cf37b44f37`, roleA −10 dB `e5592568700e643e`, roleB 0 dB `420fd31d07c9be1e`, roleB −10 dB `b825a86ab98bed7a`. Role B swaps target and other (Joseph selected). −10 dB makes the selected voice the quieter one.
- **Same selector, oracle and held-out split as T-200.** Each window goes through T-200's `EnrolledRunner` (live `EnrolledCandidate` with ECAPA/LDA selection) and `replay()` (triangular overlap-add, proxy mixer at s = 1). Only the separator changes. Window statistics use the held-out second half of the windows. The oracle stream is picked with the clean target, for evaluation only.
- **Levels matched.** Both separators' streams are rescaled to the mixture (`rescale_to_mix`), and the final output is RMS-matched to the mix before ASR (`level_match`).
- **Contexts.** 0.5, 1 and 2 s windows (hop = window / 2), plus one whole-clip window as the longer **offline** reference. The reference has no deadline (`NoDeadline`), so a slow call never swaps in the raw mix. Its latency is not deployable, and offline RTF says nothing about streaming.
- **Target loss is counted next to wrong-voice output.** whisper `small.en` transcribes the whole-clip output and the clean stems. `missing` = target words absent from the output. `wrong_voice` = unmatched output words that belong to the other speaker. Silence counts as missing, never as success.
- **SI-SDRi.** Two figures. *Clip*: whole output against the whole mix. *Overlap windows*: only held-out windows where both stems are active. Without the noise stem, a target-only window's mix *is* the target (input SI-SDR at the +100 dB cap), so including such windows produced the −50 to −90 dB "improvements" in attempt 1's saved rows. Those rows are void. Overlap-window SI-SDRi rests on only 1–4 windows per condition and swings by tens of dB, so read the clip figure first.
- **Timing.** Each window length gets one untimed warm-up call (MPS kernel build), and p50/p95 cover only window-length separator calls. Repeated long inputs (the profiles) are memoised: separation is deterministic, and SepFormer took minutes per 50 s profile call. The MossFormer2 rows come from run 2, which predates the window-length filter, so their 1 s / 2 s / offline p95 includes the two cold profile-length calls. Their p50 is unaffected.

## Results (laptop-a, Apple M4 Pro, MPS, 2026-10-10 03:24–03:47 UTC, load average 3.4–5.7 from other workers)
12 s crop of the scene (target-alone 0–3.2 s plus alternating turns 3.2–12 s), no noise stem, 4 conditions per row (2 roles × 2 levels), 74 target words in total (28 per role A condition, 9 per role B condition). MossFormer2 rows: run 2 (`eval/results/sepformer_trial/t202-run2.rows.jsonl`). SepFormer rows: run 3 (`t202-run3-sepformer.json`).

| candidate | context | streaming-valid | clip SI-SDRi dB | SI-SDRi dB (overlap win) | oracle-pick SI-SDR dB | target-absent leak dB | target words missing | wrong-voice words | sep p50 ms | sep p95 ms | RTF (p50) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mossformer2_ss | 0.5 s | yes | 8.7 | -20.9 | 15.3 | -45.4 | 8/74 | 4 | 121.6 | 141.8 | 0.243 |
| sepformer_whamr16k | 0.5 s | yes | -1.1 | -9.1 | 1.8 | -48.6 | 13/74 | 10 | 51.3 | 52.2 | 0.103 |
| mossformer2_ss | 1 s | yes | 16.6 | 14.8 | 14.5 | -84.9 | 0/74 | 6 | 206.0 | 2053.1* | 0.206 |
| sepformer_whamr16k | 1 s | yes | 4.0 | -17.9 | 1.0 | -74.8 | 7/74 | 6 | 86.5 | 87.8 | 0.086 |
| mossformer2_ss | 2 s | yes | 14.4 | 18.1 | 16.9 | -78.2 | 0/74 | 14 | 381.2 | 5624.4* | 0.191 |
| sepformer_whamr16k | 2 s | yes | 8.5 | 3.5 | -1.1 | -90.8 | 1/74 | 10 | 160.5 | 165.5 | 0.080 |
| mossformer2_ss | whole clip (12 s) | **NO (offline)** | 23.4 | 23.4 | 18.4 | - | 0/74 | 0 | 2517.3 | 9386.8* | 0.210 |
| sepformer_whamr16k | whole clip (12 s) | **NO (offline)** | -49.1 | -49.1 | -5.1 | - | 21/74 | 11 | 1100.9 | 1100.9 | 0.092 |

\* includes cold profile-length calls (run 2 code); p50 is the steady-state figure.

Per-condition window categories (held-out windows; separation / selector / mix_back / ok) and target words lost:

| context | condition | MossFormer2 categories | MossFormer2 missing | SepFormer categories | SepFormer missing | SepFormer oracle dB |
|---|---|---|---|---|---|---|
| 0.5 s | A 0 dB | 0/1/0/10 | 1/28 | 11/0/0/0 | 2/28 | -1.0 |
| 0.5 s | A −10 dB | 0/0/0/10 | 1/28 | 10/0/0/0 | 1/28 | 1.1 |
| 0.5 s | B 0 dB | 2/3/3/6 | 3/9 | 4/7/0/3 | 5/9 | 3.0 |
| 0.5 s | B −10 dB | 2/3/3/6 | 3/9 | 4/7/0/3 | 5/9 | 4.1 |
| 1 s | A 0 dB | 1/0/0/5 | 0/28 | 6/0/0/0 | 0/28 | 0.2 |
| 1 s | A −10 dB | 1/0/0/5 | 0/28 | 6/0/0/0 | 1/28 | -0.7 |
| 1 s | B 0 dB | 2/0/0/6 | 0/9 | 3/1/1/3 | 3/9 | 3.2 |
| 1 s | B −10 dB | 0/1/0/7 | 0/9 | 3/1/1/3 | 3/9 | 1.5 |
| 2 s | A 0 dB | 1/0/0/3 | 0/28 | 4/0/0/0 | 0/28 | -2.5 |
| 2 s | A −10 dB | 0/0/0/4 | 0/28 | 4/0/0/0 | 1/28 | -5.7 |
| 2 s | B 0 dB | 0/2/1/2 | 0/9 | 2/1/0/2 | 0/9 | 3.3 |
| 2 s | B −10 dB | 0/1/1/3 | 0/9 | 3/0/0/2 | 0/9 | 0.4 |
| whole clip | A 0 dB | 0/0/0/1 | 0/28 | 1/0/0/0 | 0/28 | -3.9 |
| whole clip | A −10 dB | 0/0/0/1 | 0/28 | 1/0/0/0 | 3/28 | -12.5 |
| whole clip | B 0 dB | 0/0/0/1 | 0/9 | 1/0/0/0 | **9/9** | -1.1 |
| whole clip | B −10 dB | 0/0/0/1 | 0/9 | 1/0/0/0 | **9/9** | -2.8 |

### Overlap section (run 5: 26 s crop, role A, 0 dB)
Same scene and profiles, `--clip-seconds 26 --roles A --ratios 0` (`mix_sha` `6c80946fd0a711fd`). The held-out second half (13–26 s) is almost exactly the scene's overlap section (13.1–25.4 s). Run 5 used the final code (warm-up plus window-length-only timing), with both candidates in one manifest (`t202-run5-overlap.json`), 2026-10-10 03:52–03:57 UTC, load average 4.1–4.2. 75 target words.

| candidate | context | streaming-valid | clip SI-SDRi dB | SI-SDRi dB (overlap win, n) | oracle-pick SI-SDR dB | categories sep/sel/mix/ok | target words missing | wrong-voice words | sep p50 ms | sep p95 ms | RTF (p50) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mossformer2_ss | 0.5 s | yes | 11.9 | 11.3 (44) | 11.8 | 4/1/3/44 | 0/75 | 0 | 119.2 | 123.4 | 0.238 |
| sepformer_whamr16k | 0.5 s | yes | -3.8 | -6.1 (44) | -4.0 | 47/0/1/3 | **28/75** | 10 | 50.5 | 52.0 | 0.101 |
| mossformer2_ss | 1 s | yes | 14.7 | 13.7 (23) | 13.9 | 0/0/0/26 | 0/75 | 0 | 207.6 | 212.2 | 0.208 |
| sepformer_whamr16k | 1 s | yes | -4.2 | -9.4 (23) | -3.0 | 26/0/0/0 | **30/75** | 9 | 86.8 | 88.3 | 0.087 |
| mossformer2_ss | 2 s | yes | 15.6 | 15.6 (12) | 15.4 | 0/0/0/13 | 0/75 | 0 | 383.2 | 392.0 | 0.192 |
| sepformer_whamr16k | 2 s | yes | -4.7 | -9.8 (12) | -2.7 | 13/0/0/0 | **30/75** | 17 | 160.9 | 162.4 | 0.080 |
| mossformer2_ss | whole clip (26 s) | **NO (offline)** | 16.2 | 16.2 (1) | 16.2 | 0/0/0/1 | 0/75 | 0 | 5223.6 | 5223.6 | 0.201 |
| sepformer_whamr16k | whole clip (26 s) | **NO (offline)** | -3.7 | -3.7 (1) | -3.7 | 1/0/0/0 | **28/75** | 34 | 4674.8 | 4674.8 | 0.180 |

No held-out window was target-absent, so this run has no leak figure. No deadline was missed at any streaming context. Accelerator memory (process driver pool, `torch.mps.driver_allocated_memory()`): 10 010 MB after the MossFormer2 pass, 22 266 MB after the SepFormer pass. SepFormer load: 0.20 s.

## Reading
- **SepFormer WHAMR16k loses the target before selection on this scene.** Most of its windows land in `separation` (oracle stream < 4 dB). Its oracle-pick SI-SDR is −5 to +2 dB at every context, against 14–18 dB for MossFormer2. This is the separator itself, not ECAPA, the mixer or the deadline. A 2 s window where only the target speaks (mix = target) gives just 2.5 dB in SepFormer's better stream on both CPU and MPS: a two-speaker model splits one voice across its two outputs.
- **In the overlap section the gap is decisive.** SepFormer loses 28–30 of 75 target words at every context, with 13–47 of its held-out windows in `separation` and oracle-pick SI-SDR at −4 to −3 dB. MossFormer2 loses 0/75 at every context, with oracle-pick SI-SDR at 12–16 dB.
- **Target words (12 s crop).** At 1 s and 2 s MossFormer2 keeps all 74 target words; SepFormer drops 7 and 1. On the whole-clip offline reference SepFormer loses every Joseph word (role B, 9/9 at both levels), while MossFormer2 loses none. Wrong-voice counts are similar for the two (6–14 vs 6–11 words) and do not show target loss: a run can have few wrong-voice words and still drop the target.
- **Compute.** SepFormer's per-window p50 is about 2.4× lower (RTF 0.08–0.10 vs 0.19–0.24 on MPS). That cannot buy back a 12–18 dB separation deficit. These are offline per-window compute figures under other workers' load, not a streaming latency claim.
- **Possible confounds, not tested.** WHAMR! trains toward the anechoic source, so if the scene's stems carry room reverberation, SI-SDR against them penalises a dereverberating model. The word counts are the cross-check, and they also go against SepFormer. The noise stem was not added (`--noise` not run), although WHAMR! is a noisy-reverberant model, so the noisy condition (its home ground) is not measured.
- **Verdict: no-go for this candidate.** Do not add it to the registry. The comparison does not justify replacing MossFormer2_SS.

## What this says about the embedding questions (from T-200 / T-203, not re-measured here)
Each item is labelled demonstrated, ruled out or unknown.
- **Target loss before selection.** For MossFormer2 on this scene: ruled out at 1 s, 2 s and whole clip (0/74 words missing). At 0.5 s it is demonstrated: 8/74 words missing, plus `separation` windows in role B. For SepFormer: demonstrated at every context.
- **ECAPA crop duration / contamination.** Demonstrated as a margin loss by T-200 (0.77 margin at 4 s down to 0.42 at 0.5 s; 0.69 down to 0.26 at 50% contamination), but the margins stayed positive. Here, `selector` windows appear mostly at 0.5 s in role B (3/14 MossFormer2, 7/14 SepFormer) and almost vanish at ≥ 1 s, which fits short crops as a factor.
- **Profile vs demo-mic mismatch.** Unknown: these profiles and stems are not demo-mic recordings.
- **Negative-profile rejection at cos > 0.7.** Ruled out for this pair (T-200: target/other profile cos 0.066). The saved-negative path that bypasses the gender guard was not exercised.
- **Missing-camera strict pause, same_bucket, profile_negative/readiness/active tier.** Unknown offline: the replay passes no ASD, and the tier state lives in the session.
- **Mixer/AGC/deadline fallback.** The proxy mixer appears in `mix_back` in 0–3 windows per condition. No deadline misses occurred at any streaming context, because per-window compute stayed below the hop. The real `hearing.fusion.mixer`/AGC was not run.
- **T-069 male/male ceiling gap** (accuracy 0.742 vs candidate ceiling 0.930 at 1 s): not reproduced. MossFormer2's oracle-minus-actual gap here is −5.0 to +7.1 dB per condition (mean +0.5 dB over 16 rows), so this scene does not show T-069's selection gap.
- **Measured intervention that could make it work:** none from this trial. A different blind separator is not the lever on this scene. The next measurement is T-200's: demo-mic re-enrolled profiles and scenes, `--noise`, −5/−15 dB, crops of at least 2 s.

## Costs and caps
- Attempt 1 hit the 30-minute evaluation cap: `EnrolledCandidate` separates the 50 s Aryan profile and the 14 s Joseph profile again for every condition. MossFormer2 handles that in seconds; SepFormer on MPS took minutes per call (the process was paged out under memory pressure). Memoising the identical profile separations brought the SepFormer pass from about 4 minutes per condition to 4 minutes for all 16 rows.
- Evaluation time this attempt: run 2 MossFormer2, about 6 min; run 3 SepFormer, about 4 min; run 5 both candidates on 26 s, about 5 min. All three ran under `swarm-lock --exclusive`, about 15 min in total, inside the 30-minute evaluation cap.

## Limitations
One scene and one profile pair. A 12 s crop (2 roles × 2 levels) plus a 26 s overlap crop (role A, 0 dB only), no noise stem, one run per configuration, and machine load from other workers (so compute figures are upper bounds). On the 12 s crop the overlap-window SI-SDRi rests on 1–4 windows per condition (12–44 on the 26 s crop). Role B (Joseph selected) and −10 dB were not run on the overlap section. No audio, weights or results are committed: the manifests live in `$HEARING_RESULTS_DIR/sepformer_trial/`.
