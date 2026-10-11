# Profile perturbation bounds

Offline research harness for checking how much an enrolled voice profile can vary before its
ECAPA cosine score degrades. Run it against a consented profile recording:

```bash
.venv/bin/python -m eval.perturb_profiles eval/fixtures/generated/profiles/aryan/profile-01.wav
```

Rendered WAVs are written below `eval/shards/profile_perturbations/` and are gitignored. The
committed `manifest.json` is intentionally hash-only: it contains the source/render SHA-256,
transformation controls, and cosine measurements, but no audio, embeddings, or feature arrays.
Use `--dry-run` only to check wiring; its deterministic spectral proxy is **not** speaker
recognition evidence.

## Controls and interpretation

| Control | Default sweep | Definition |
|---|---:|---|
| Pitch | -3, 0, +3 semitones | Resample to shift pitch, then restore the original duration |
| Speed | 0.9, 1.0, 1.1 | Duration multiplier inverse: `1.1` is 10% faster |
| Formant | -2, 0, +2 semitones | STFT spectral-envelope frequency warp, preserving duration |

For each rendered profile, `cosine` is the cosine between the unmodified profile embedding and
the perturbed embedding; `cosine_degradation = 1 - cosine`. These are identity-margin bounds,
not acceptance thresholds. A threshold must be calibrated from target holdouts and imposters, and
the same profile piece duration, sample rate (16 kHz mono float32), and ECAPA model/cache must be
used for all rows. Report p50/p95 across multiple source takes before changing a live margin.

The transformations are deliberately controlled approximations. Resampling pitch shift also
changes spectral detail, while the formant operation warps the STFT magnitude envelope and is not
a vocal-tract synthesizer. Neither perturbation substitutes for a real-room, real-microphone
evaluation. Do not promote a profile or alter `SAME_VOICE`/`MARGIN` from this experiment alone.

## Reproducibility and privacy

The script loads the existing ECAPA adapter lazily; no network is contacted by the harness. Model
weights stay in the configured model cache. Source recordings and rendered audio remain outside
git, and derived embeddings are held only in memory. A manifest's source hash must be retained
with the result so a later run can detect that the source take changed.
