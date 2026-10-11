# Overnight room-response render preparation

## Purpose

`eval/prep_render_room_noise.py` prepares offline evaluation variants by convolving a local clean
recording with a local, already-verified room impulse response (RIR), then adding local noise at
declared SNRs **0, 5, 10, and 20 dB**. The output contains separate `target.wav` and `noise.wav`
stems plus `mix.wav`, with `mix = target + noise` and no peak normalisation. This is experiment
preparation, not production adoption and not a claim about a real room or distance.

## Consumer instructions

```bash
.venv/bin/python -m eval.prep_render_room_noise \
  --clean-dir <local-clean-recordings> \
  --noise-dir <local-verified-noise-cache> \
  --rir-dir <local-verified-rir-cache> \
  --out <ignored-render-shard> --initial 12
```

The script is offline and bounded to 600 variants. It writes an atomic per-variant directory and
`index.json`; rerun with `--limit N` to extend a shard. If no RIR is available, the default is an
explicit deterministic `SYNTHETIC_acoustic_IR` fallback. Use `--no-synthetic-ir` to require a
local RIR. Every manifest records source paths, input rates, unit-energy IR normalisation, target
role, measured SNR, and the explicit statement that digital convolution is not simulated distance.
Keep output outside git (audio and private media must not be committed).

## Current observations and hypotheses

No clean team recordings, verified noise cache, or verified RIR was present in this worktree, so no
overnight render measurements were produced. The tests establish deterministic convolution,
resampling, additive stems, and provenance only. **Hypothesis:** a measured RIR may improve room
realism relative to the synthetic fallback; this must be tested with the actual local cache and
must not be described as a distance measurement.
