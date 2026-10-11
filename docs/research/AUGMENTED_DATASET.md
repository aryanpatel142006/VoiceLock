# Augmented team-audio dataset (T-306)

The standard augmented set for muse-c robustness experiments: the team's own voices (Aryan, Joseph) under
pitch shift, reverberation and the Opus codec. The generator is `eval/augment_team.py`. The audio is
gitignored; the committed record is `eval/augmented/T-306/manifest.json` (sha256 per file) and
`eval/augmented/T-306/spot_check.json`.

## Reproduce
Run this from the main checkout or any worktree (needs ffmpeg with libopus, plus librosa/scipy/soundfile from the `.venv`):
```
.venv/bin/python -m eval.augment_team                      # render + spot-check, exit 0 only if every check passes
.venv/bin/python -m eval.augment_team --check-only         # re-check an existing render (sr, mono, peak, sha256)
.venv/bin/python -m eval.augment_team --manifest-copy eval/augmented/T-306   # also copy manifest/spot_check into the repo
```
- Input: `$HEARING_FIXTURES_DIR/team/{aryan,joseph}/*.wav` (default `<main checkout>/eval/fixtures/generated/team`,
  16 kHz mono), `--source DIR` to override. Used on Oct 10 2026: `aryan/fixture-01.wav` (58.7 s), `aryan/profile-01.wav`
  (50.0 s), `joseph/fixture-01.wav` (48.1 s), `joseph/fixture-02-namecall.wav` (27.8 s), `joseph/profile-01.wav` (15.7 s).
  `_incoming/` is not read. Each source's sha256 is stored in every manifest row (`source_sha256`).
- Output: `<main checkout>/eval/augmented/T-306/` (`--out DIR` to override); 63 MB, 12.6 s wall on laptop-a (M4 Pro).
- The render is deterministic: a second render to a scratch dir gave byte-identical sha256 for all 215 files
  (ffmpeg 8.0.1, librosa 0.10.2.post1, scipy 1.17.1, soundfile 0.12.1).

## Layout
`<out>/<group>/<speaker>/<source-stem>_sNN[suffix].wav`, 16 kHz mono PCM16, plus `manifest.json` and `spot_check.json`.

| Group | What | Files |
|---|---|---|
| `clean` | source segment, peak-capped only | 21 |
| `pitch_pm2` | `librosa.effects.pitch_shift` +2 (`_p2`) and -2 (`_m2`) semitones, same duration | 42 |
| `pitch_pm4` | ±4 semitones (`_p4`, `_m4`) | 42 |
| `reverb_02` / `reverb_05` / `reverb_08` | convolution with a synthetic RIR, RT60 0.2 / 0.5 / 0.8 s, dry RMS kept | 21 each |
| `opus16` / `opus32` | ffmpeg `libopus -b:a 16k/32k -vbr off -application voip` (Ogg), decoded back to 16 kHz | 21 each |
| `combo` | hardest 20% of segments: pitch ±4 (alternating by rank) then RT60 0.8 s reverb (`_p4_rt08` / `_m4_rt08`) | 5 |

The segments are 10 s cuts of each source (a tail shorter than 4 s joins the previous segment). That gives 21 segments, 11 Aryan and 10 Joseph.

**Synthetic RIR:** a unit direct path, then 5 ms of silence, then Gaussian noise times `exp(-6.9078 t / RT60)` (-60 dB at RT60).
The tail energy is 0.25 × the direct path, the RIR is 1.2 × RT60 long, and the seed comes from sha256(`speaker/stem`) and is stored as `rir_seed`.
A unit test checks that the Schroeder T20 estimate lands within 20% of the nominal RT60.

**Hardest 20%:** segments are ranked by a speech-to-floor proxy, the p95 minus p10 of the 20 ms frame RMS in dB (lower is harder).
The bottom `ceil(0.2 × 21) = 5` go to `combo`. All five are Aryan's quieter Voice Memo cuts (proxy 20.8-22.3 dB, against 27.0-40.0 dB for Joseph's).
Each combo row stores `difficulty_rank`.

**Level:** every output is scaled *down* (never up) so its peak is at most -3 dBFS. The gain applied is `params.peak_cap_gain_db`.
The Opus groups encode the already-capped clean segment and are capped again after decoding.

## manifest.json
Top level: `task`, `sample_rate`, `segment_s`, `peak_cap_dbfs`, `groups`, `combo_fraction`, `n_segments`, `n_files`,
`counts`, `max_peak_dbfs`, `files`. Each entry in `files` has
`{path, speaker, augmentation, params, source_file, source_sha256, source_start_s, source_end_s, difficulty_db, sample_rate, duration_s, peak_dbfs, sha256}`.
`sha256` is the hash of the written wav. Example:
```
{"path": "reverb_05/joseph/fixture-01_s00.wav", "speaker": "joseph", "augmentation": "reverb_05",
 "params": {"rt60_s": 0.5, "rir": "synthetic_exp", "rir_seed": ..., "peak_cap_gain_db": 0.0}, "source_file": "fixture-01.wav", ...}
```

## Verification (Oct 10 2026, laptop-a)
The spot check re-reads 5 files per group, spread evenly, and checks for 16 kHz, mono, peak < -1 dBFS and a sha256 that matches the manifest.
All 45 checks passed. Peak per group over every file (dBFS, min / max):

| Group | min | max |
|---|---|---|
| clean | -17.58 | -3.00 |
| pitch_pm2 | -22.90 | -3.00 |
| pitch_pm4 | -23.33 | -3.00 |
| reverb_02 | -17.48 | -3.00 |
| reverb_05 | -17.49 | -3.00 |
| reverb_08 | -17.69 | -3.00 |
| opus16 | -17.19 | -3.00 |
| opus32 | -17.39 | -3.00 |
| combo | -22.52 | -20.08 |

No file is above -3.00 dBFS, so none clips. Durations are 5.70-10.00 s, and each augmented file has the same length as its clean segment.
Nobody has listened to the files: this check covers metadata only.

## Caveats
- Small set: 2 speakers, 5 recordings, about 200 s of source audio. Both are male_adult Voice Memos recorded through a phone mic, so the set is not balanced.
- The RIRs are statistical (exponential noise tails), not measured rooms. There are no early reflections and no room geometry.
- Pitch shift uses a phase vocoder (librosa) and leaves a mild phasiness, and the formants move with the pitch. For formant-only shifts see `eval/perturb_profiles.py`.
- "Hardest" means a level-based proxy, not a model error rate. Re-rank by model WER or SI-SDR once a baseline exists.

---

# T-310 Augmented Team-Audio Dataset

Standard robustness dataset for failure-mode validation (T-294 F1–F8) and GPU sweeps (Q-683, Q-708).

## Sources (stand-ins)

Real team recordings (Aryan/Joseph) live on laptop-b (`docs/TEAM_RECORDINGS.md`) and are **not** in this checkout. This dataset uses LibriSpeech stand-ins:

| label | file | LibriSpeech speaker | sex |
|---|---|---|---|
| `speaker_a` | `eval/fixtures/tiny/speech_a.wav` | 1580 | F |
| `speaker_b` | `eval/fixtures/tiny/speech_b.wav` | 61 | M |

Manifests use neutral `speaker_a`/`speaker_b` labels — never real names on stand-in data. `manifest.json` records `sources_are_standins: true` plus the true LibriSpeech speaker ids for provenance. To rebuild with real team audio, point the generator at the new sources (same 16 kHz mono WAV).

## Groups (9)

| group | transform | params |
|---|---|---|
| `clean` | none | — |
| `pitch_pm2` | pitch +2 st | ffmpeg rubberband (or librosa fallback), duration-preserving |
| `pitch_pm4` | pitch +4 st | ffmpeg rubberband (or librosa fallback), duration-preserving |
| `reverb_02` | synthetic RIR | RT60 0.2 s, exp-decay noise, direct path = 1.0 |
| `reverb_05` | synthetic RIR | RT60 0.5 s |
| `reverb_08` | synthetic RIR | RT60 0.8 s |
| `opus16` | Opus codec round-trip | 16 kbps via libopus (48 kHz intermediate) |
| `opus32` | Opus codec round-trip | 32 kbps via libopus |
| `combo` | pitch +4 st, then reverb 0.8 s | hardest-20%-rule: applied to all sources (n=2) |

Layout: `eval/augmented/T-310/<group>/<speaker>/*.wav` (16 kHz mono PCM16). `manifest.json` carries sha256, params, and peak dBFS per file.

## Measured peaks (dBFS, max per group)

| group | peak range |
|---|---|
| clean | −5.8 … −4.4 |
| pitch_pm2 | −11.3 … −8.6 |
| pitch_pm4 | −13.1 … −7.6 |
| reverb_02 | −4.2 … −3.0 |
| reverb_05 | −4.8 … −3.3 |
| reverb_08 | −5.3 … −3.2 |
| opus16 | −6.6 … −4.1 |
| opus32 | −5.9 … −4.0 |
| combo | −12.9 … −7.9 |

No clipping anywhere (all < −1 dBFS). Reverb/combo are quieter by construction (energy spread into tail); normalize per-file if your consumer needs matched loudness.

## Reproduce

```bash
bash eval/augmented/T-310/reproduce.sh
python3 -m pytest tests/test_augmented_t310.py -q
```

## Use for

- T-294 F1–F8 failure-mode validation (esp. F8 phone-AGC — note: these clips lack AGC artifacts; see T-294's phone-AGC simulation)
- Q-683 head-robustness sweep, Q-708 8-condition sweep (ilab1)
- STEAL-DISTILL student training (degraded inputs)
