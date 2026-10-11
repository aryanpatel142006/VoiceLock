# T-214: Overnight mixer and AGC counterfactuals on fixed separator outputs

## Status: offline counterfactual evaluation framework implemented; blocked on upstream saved separator bank

### Context and scope

T-214 evaluates mixer and AGC counterfactual policies on frozen separator outputs without rerunning the upstream extraction models.
Under the Joseph explicit authorization and FABLE approved order/caps:
- Upstream separation outputs (selected stems, actual selection evidence/confidence, mix, target, interferer) from T-213 / T-289 are consumed as fixed inputs.
- Evaluation runs strictly offline without GPU duplication, without rerunning heavy separation models, without opening microphones or speakers, and without modifying runtime defaults, profiles, `PLAN.md`, `CONTRACTS.md`, or `ARCHITECTURE.md`.
- Upstream saved output bank remains pending from the GPU executor / T-289 handoff. Therefore, real-scene metrics are reported as **not measured** pending upstream artifacts. No synthetic or mock quality claims are made.

### Configurations evaluated (<= 4 configurations)

1. `baseline`: Frozen current baseline mixer (`confidence_floor_db` with `UNSURE_DB = 6.0` to `SURE_DB = 40.0`) + baseline T-169 AGC.
2. `defined24db`: Explicitly defined 24 dB minimum-attenuation floor candidate (`floor_db >= 24.0 dB`) respecting slider semantics (`max(room_gain(s), floor_gain)`) + baseline AGC.
3. `t180_k2`: T-180 K2 candidate (`HEARING_GATED_FLOOR_DB = 20 dB` for gated / zero-confidence windows) + baseline AGC.
4. `t179_agc`: Frozen baseline mixer + opt-in T-179 noise-floor-aware AGC (`FLOOR_RULE`: `never_up=True`, `max_gain_db=9.0 dB`).

### Decision gate criteria for candidate adoption

A candidate configuration is considered an improvement over the baseline if and only if across matched scenes:
- **Interferer leakage**: Improves leak by $\ge 3\text{ dB}$ (i.e. `candidate_leak_db <= baseline_leak_db - 3.0 dB`).
- **Wrong-speaker rate**: Does not increase wrong-speaker rate (i.e. `candidate_wrong_speaker_s <= baseline_wrong_speaker_s`).
- **True dropouts**: Additional true dropouts $\le 0.2\text{ s}$ per matched scene (i.e. `candidate_true_dropout_s <= baseline_true_dropout_s + 0.2 s`).

All failures against these gates, per-case delta numbers, and denominators are recorded in `summary.json`.

### Metrics measured per configuration

- Target-word retention rate and deletions / substitutions / insertions via Whisper small.en.
- Wrong-speaker duration (seconds) over 100 ms energy attribution windows.
- Interferer projection gain (leak, dB).
- Target-absent output energy and pause lift (dB).
- True dropouts (seconds where target is active but output gain $< -10\text{ dB}$).
- Pre-AGC vs post-AGC SI-SDRi and loudness matching for human-review listening clips (`out_matched.wav`).

### Reproducibility command

```sh
.venv/bin/python -m eval.overnight_output_ab \
  --bank /path/to/frozen_separator_bank.json \
  --model /Users/joeroche/Developer/selective-hearing/models/faster-whisper-small.en \
  --output eval/results/T-214-counterfactuals \
  --limit 12 \
  --seconds 3600
```

### Checkpointing and failure handling

- Each case is executed in a sandboxed subprocess and checkpoints its result to `result.json` atomically.
- Direct case failures and timeouts are checkpointed as status `error` with full error logs; no infinite retry loops.
- Provenance hashes (runner, model files, bank, input audio) are strictly checked to prevent stale or mismatched cache reuse.
