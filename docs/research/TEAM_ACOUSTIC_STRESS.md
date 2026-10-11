# Team acoustic stress preparation

`eval/prep_team_acoustic_stress.py` creates an offline **synthetic** stress bank
from Aryan and Joseph profile pieces and a user-supplied babble WAV. The RT60
conditions (0.2 s and 0.5 s) are engineering stress assumptions, not measured
HackRU venue acoustics. Neither this bank nor its unit tests authorize demo
promotion; independently review and listen to real rendered pairs first.

```bash
.venv/bin/python -m eval.prep_team_acoustic_stress \
  --profiles "$HEARING_FIXTURES_DIR/profiles" \
  --noise "$HEARING_FIXTURES_DIR/team/venue_noise.wav" \
  --out "$HEARING_FIXTURES_DIR/team_acoustic_stress"
```

Both speaker directories must contain WAVs. All nested versions are included;
point at a curated input tree to select particular versions. Inputs are
explicitly downmixed and resampled to mono 16 kHz by `hearing.audio.io.load`.
A missing noise file, speaker directory, or speaker WAV set raises an error.
No recordings, API, model weights, microphone, or speakers are needed for tests.

Each RIR uses a seeded signed Gaussian diffuse tail multiplied by an exponential
amplitude envelope declining 60 dB per requested RT60. The direct path and tail
each receive half the energy; total squared RIR energy is one. This is a simple
broadband stochastic approximation without geometry, early reflection modeling,
or frequency-dependent absorption. Unit tests estimate RT60 independently from
the Schroeder integrated energy decay (-5 to -35 dB fit extrapolated to -60 dB).
Seeds derive from SHA-256 of each input's path relative to the profile root;
version paths also distinguish output IDs, avoiding same-name overwrites.

FFT convolution retains the full RIR tail. Short noise is repeated (long noise
is cropped) to the resulting length, then RMS-scaled against the reverberant
target at 5 or 10 dB SNR. A shared gain caps both mixture and target peaks at 0.9,
including cases where noise cancellation makes the mixture quieter. Targets are
reverberant references, not dry speech. `measured_snr_db` is measured from the
saved PCM16 target and mixture, so it includes quantization error. Repeating
very short noise can create artificial periodicity; use a long real babble clip
for meaningful listening. Input babble provenance is supplied by the operator,
not inferred by the script.

## Generated manifest contract (version 2)

Real manifests and media stay under the ignored generated directory. The old
tracked placeholder was removed; this documentation is the template. A successful
run writes `manifest.json` with the following fields:

| Field | Meaning |
|---|---|
| `version` | `2` |
| `kind` | `team_acoustic_stress` |
| `acoustics` | `synthetic_not_measured_venue` |
| `rir_method` | `seeded_signed_exponential_unit_energy` |
| `sample_rate` | `16000` |
| `rt60_s`, `snr_db` | `[0.2, 0.5]`, `[5.0, 10.0]` |
| `noise` | Supplied noise source path |
| `pairs` | One entry per source × RT60 × SNR |

Each pair contains `id`, `speaker`, `source`, integer `rir_seed`, requested
`rt60_s` and `snr_db`, saved-pair `measured_snr_db`, and relative `target`/`mix`
WAV filenames. There are four pairs per source piece. Generated manifests expose
local source paths and should remain private with the recordings.

Tests use tones as profile/noise stand-ins solely to verify numerical behavior,
missing inputs, short-noise handling, deterministic output, and version-safe IDs.
They provide no speech separation or venue performance evidence. No learned
model or dependency was introduced. Independent review remains required before
accepting this repaired generator or producing an overnight evaluation bank.

## Local preparation check (2026-10-10)

Fresh continuation verification generated eight pairs using the main checkout's
`profiles/{aryan,joseph}/profile-01.wav` (50.0 s Aryan, 15.7 s Joseph) and
`eval/fixtures/tiny/noise_babble.wav`. The babble is the fixture builder's
six-speaker LibriSpeech mixture, **not a HackRU venue recording**.

Preparation took 0.165 s on `MacBook-Air.local` (arm64 CPU), including WAV
writes and saved-pair SNR measurement. Independent saved-audio checks confirmed
all eight speaker × RT60 × SNR combinations, finite mono 16 kHz audio, equal
reference/mixture lengths, and peaks at most 0.9001. Saved SNRs ranged from
4.999993 to 10.000003 dB, within 0.05 dB of each request. Seed-278 Schroeder
fits estimated 0.200607 s and 0.506573 s RT60. These are preparation checks,
not SI-SDRi, WER, streaming latency, or listening results.

The absent ignored worktree bank was regenerated at
`eval/fixtures/generated/team_acoustic_stress/`. Its manifest and all 16 WAVs
match the persistent main-checkout bank byte for byte (SHA-256). Both copies
remain ignored with private source paths. Use recorded venue babble to evaluate
actual venue noise. No generator or interface changes were needed.

The focused numerical/file-output suite passed 11 tests in 1.58 s. The branch
was clean with no unmerged files or active integration operation; the tracked
worker report contained no conflict markers. No Git integration was performed.
The required `bash scripts/verify_fast.sh` passed lint, 1,916 Python tests
(61 deselected, 159.14 s), and all 67 Node tests.
