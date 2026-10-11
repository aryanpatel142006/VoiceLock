# Overnight acoustic preparation (T-239)

This is a preparation manifest, not a renderer. It inventories local cached
noise and room impulse-response files and emits deterministic convolution
recipes for T211. It performs no network access, opens no audio devices, and
writes no mixtures. Unknown licences, undecodable files, zero-valued impulse
responses, and RIRs whose sample rate is not 16 kHz are excluded rather than
paired. Every accepted source carries its path, SHA-256, licence, and any
available speaker/utterance/interval metadata into each recipe.

## Replay command

From the repository root:

```bash
PYTHONPATH=. .venv/bin/python -m eval.prep_acoustic_plan \
  --cache-root "${HEARING_FIXTURES_DIR:-eval/fixtures/generated}" \
  --output "${HEARING_RESULTS_DIR:-eval/results}/overnight_prep_acoustic.json" \
  --seed 239 --max-recipes 24
```

The output is atomically replaced. To resume without rewriting a verified
manifest, add `--resume`; the command checks every `config_hash` first and
rebuilds if the manifest is incomplete or tampered with. The plan caps itself
at 24 recipes, 500 discovered inputs, 20 minutes per consumer case, and 1 GiB
of derived bytes. `fixed_original_group` keeps recipe comparisons grouped by
the same original source.

## Recipe contract

Audio is mono float32 at 16 kHz. Each recipe declares the noise SNR
(`-5, 0, 5, or 10 dB`), RIR alignment (`peak_sample`), tail duration, and
peak scaling. If no valid local RIR is available, the plan declares one of
three synthetic unit-impulse variants with explicit exponential-tail
parameters; this is simulated, not a measured room. If no valid noise is
available, the recipe is still emitted as a planned pairing with a
`consumer must supply noise` marker, never as an available input.

## Availability and handoff

Availability is measured at invocation in the JSON `available` object and is
separate from `planned`. On this worktree there are no T225/T227 report files
or room/noise cache records to import, so no remote URL or remote availability
is claimed. T211 should consume the manifest with:

```bash
PYTHONPATH=. .venv/bin/python -c \
  'import json; p=json.load(open("eval/results/overnight_prep_acoustic.json")); print(p["available"], len(p["recipes"]))'
```

The recipes are composable inputs for T211's augmentation renderer; this task
does not enqueue jobs, tune held-out groups, allocate GPU, or produce quality
measurements.
