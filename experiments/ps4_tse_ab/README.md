# T-373: PS4 and REAL-TSE `spk_emb_100` vs the current demo path (offline A/B, Oct 10 2026)

Investigation only; no runtime change. Code: `experiments/ps4_tse_ab/run.py`. Host laptop-a (M4 Pro, MPS), 4 real-pair rows =
T-323/T-326's `t141/{joseph,aryan}/demo-team-{easy,hard}` (30.2 s each, target:other = joseph:aryan, aryan:joseph, joseph:aryan, aryan:joseph), enrollment =
the first 10 s of the saved profile (PS4 trains on <= 10 s). Raw JSON: `.swarm-run/ab_mps.json`, `ab_demo.json`, `ab_cpu.json`.

## Verdict: REJECT for the live path (PS4 full-file is the best number but needs the whole clip)
* **PS4 full file** is the best: 1.3 % wrong, SI-SDRi +2.5 dB, leak -13.8 dB, no dropouts. It is not usable live: it saw the whole 30 s mixture.
* **PS4 in 1 s / 2 s windows** (what a live tier can do) is worse than the current demo path: 17.3 % / 8.3 % wrong vs 2.8 % (demo with
  `HEARING_GATED_FLOOR_DB=20`), SI-SDRi -3.1 / -0.9 dB, target pushed down >= 10 dB for 10.8 / 9.2 s of 120 s. A 2 s window also costs 1 s of extra delay.
* **REAL-TSE `spk_emb_100`** does not separate these two male voices at all: 26 % wrong in every mode, leak about 0 dB. Control confirms
  the A/B harness works (PS4 vs REAL-TSE differ 20x in full-file wrong %), and that Libri2Mix-trained BSRNN does not transfer to this room.
* Untested: PS4 with 5-10 s windows (a delayed Focus-HQ-style tier). The 1 s -> 2 s -> full trend (17 -> 8 -> 1 %) says context is the lever; that would be the only ADAPT idea.

## Table (mean over the 4 rows, wrong % per row in brackets; sums over 4 x 30.2 s = 120.8 s for the seconds columns)
"stream" = the model's output alone; "+res20" = stream + 0.1 x (mix - stream) (residual 20 dB down); demo "played" = live `tse>mixer`
at the confidence floor (6 dB when gated), "played gated20" = same with `HEARING_GATED_FLOOR_DB=20`. Chunked outputs are stitched like the live
enrolled tier (window 1 s hop 0.5 s, or 2 s / 1 s). ms/1 s = MPS wall time per second of audio fed (chunked: all windows / clip duration).

| Arm | wrong % mean (rows) | SI-SDRi dB | leak dB | true dropout s | target down >= 10 dB s | MPS ms per 1 s | p95 ms per window |
|---|---|---|---|---|---|---|---|
| **demo path played (default floor)** | 12.0 (15.3 / 6.0 / 18.7 / 7.9) | +1.73 | -8.5 | 2.98 | 17.0 | 696 | 479 |
| **demo path played, gated floor 20** | **2.8** (0.0 / 5.3 / 2.0 / 4.0) | +0.39 | -13.0 | 2.98 | 30.2 | 696 | 479 |
| demo stream alone | 2.3 (0.0 / 3.3 / 2.0 / 4.0) | -0.85 | -15.8 | 19.38 | 33.0 | 696 | 479 |
| PS4 full file, stream | 1.3 (1.3 / 0.7 / 2.0 / 1.3) | +2.46 | -13.8 | 0.00 | 2.8 | 380 (incl. first-call warm-up) | n/a (1 call, 11-14 s cold) |
| PS4 full file, +res20 | 0.3 | +1.71 | -17.6 | 0.00 | 5.2 | | |
| PS4 2 s / 1 s, stream | 8.3 (10.0 / 3.3 / 14.7 / 5.3) | -0.85 | -10.6 | 0.58 | 9.2 | 172 | 270 |
| PS4 2 s / 1 s, +res20 | 5.0 | -1.91 | -13.9 | 0.12 | 14.4 | | |
| PS4 1 s / 0.5 s, stream | 17.3 (18.0 / 11.3 / 24.0 / 15.9) | -3.06 | -6.6 | 0.40 | 10.8 | 239 | 230 |
| PS4 1 s / 0.5 s, +res20 | 12.8 | -4.16 | -9.1 | 0.06 | 23.8 | | |
| REAL-TSE full file, stream | 25.9 (22.0 / 29.1 / 24.7 / 27.8) | +0.33 | +1.8 | 0.00 | 1.2 | 365 (incl. warm-up) | n/a |
| REAL-TSE 2 s / 1 s, stream | 26.1 | -1.70 | +0.5 | 0.40 | 16.8 | 167 | 274 |
| REAL-TSE 1 s / 0.5 s, stream | 26.7 | -2.72 | -0.1 | 0.40 | 16.4 | 240 | 238 |

Wrong % = 200 ms frames where the other talker dominates the output (`session_score` rule) / 30.2 s. Leak: the other stem's LS energy in the output vs
in the mix (lower is better). The demo path's SI-SDRi is near 0 because the residual is mixed back; its row is the "current demo path" the task names, run through
the T-323 replay (`EnrolledCandidate`, real weights, saved profiles, no AGC / cold start).

**CPU** (laptop-a, 1 row, 1 s / 0.5 s windows, under `swarm-lock --exclusive`, load average 7-9 before and after): PS4 1635 ms and REAL-TSE 1628 ms per second of
audio, p95 952 / 897 ms per window: **not real time on CPU** (hop is 500 ms). On MPS both are 0.24 RTF (p95 230-240 ms per 1 s window). PS4 needs two small patches to
run on MPS (`.type(wav_input.type())` -> `.to(dtype)` in `inference.py`, `PYTORCH_ENABLE_MPS_FALLBACK=1`); done on a temp copy, upstream untouched.

Timing caveats: the MPS pass ran under `swarm-lock --exclusive` (load 8.3 at the end); the demo-path rerun for the gated-20 variant ran without the lock (accuracy
identical, its timing 696 vs 828 ms is load noise); full-file timings include MPS first-call shape compilation.

## Provenance
* PS4: HF `TaurenMountain/PS4` (`checkpoint_epoch037.pt`, 283 MB, Apache-2.0, `inference.py` self-contained BSRNN + ECAPA-TDNN-512).
* REAL-TSE: Google Drive checkpoint pack from the `REAL-TSE/wesep-real-tse` README (1.05 GB zip, used `spk_emb_100`; the repo ships no LICENSE file and the
  Drive pack has none; upstream WeSep is Apache-2.0, challenge weights terms not stated: treat as research-only). Run through the repo's `wesep` code with
  `silero_vad` stubbed and `wespeaker`'s ECAPA replaced by PS4's identical network (all 972 keys loaded). No P1 post to reuse was found in the repo.
* Weights live in `$TMPDIR/T-373-*` only; nothing committed.
