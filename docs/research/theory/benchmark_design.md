# Failure-localizing benchmark suite (T-248, theory R&D)

Status: design + metric prototype on **synthetic arrays only**. No real audio, model, or GPU was used.
Real inputs available: 0; real inputs used: 0. All numbers below are from the synthetic simulator and
validate the *metric code*, not any product behaviour.

Prototype: `experiments/theory/T-248/bench_design.py`; tests: `tests/test_theory_benchmark_design.py`.

## Principle
Aggregate SI-SDR/WER cannot say which stage failed. Every case is scored at five tap points and each
stage is replaced by its oracle in turn (causal ablation), so the loss is attributed to a stage.

Taps: (0) separator streams, gate fully open; (1) oracle selector; (2) actual selector; (3) after
residual/mixer/AGC; (4) after streaming deadline/fallback.

## Metrics (all computed per case, aggregated by source group)
| Question | Metric | Definition |
|---|---|---|
| Target preserved before selection? | `preservation_si_sdr_db` | SI-SDR of the target stream vs the target reference on target-word samples, selector bypassed |
| Ceiling | `ablation_oracle_selector` | SI-SDR with oracle frame choice, same separator and post stages |
| Selector error | `target_miss_rate`, `wrong_speaker_rate`, `false_accept_rate`, `frame_agreement` | frame-level vs oracle choice (target / other / none) |
| Residual/mixer/AGC | `delta_no_post_minus_full`, `gain_std_db`, `gain_speech_swing_db` | paired SI-SDR change from removing post stage; gain trajectory spread on target words |
| Target-absent false speech | `false_speech_s`, `target_absent_output_rate` | seconds of output frames above -40 dBFS where reference target is silent, in `target_absent` cases |
| Word-boundary loss | `onset_clip_ms`, `tail_clip_ms`, `head_db`, `tail_db` | per target word: ms before the target is selected / after it is dropped; energy kept in first/last 50 ms vs reference |
| Hysteresis / pause swaps | `pause_swap_rate` | share of 100-800 ms target pauses in which the output switched to the other talker |
| Switching / reacquisition | `reacq_latency_ms` | target onset after >= 500 ms absence until the target is selected; `inf` if never |
| Streaming deadline/fallback | `deadline_miss_rate`, `max_miss_burst_frames`, `delta_no_fallback_minus_full` | per-frame compute time vs 10 ms hop; fallback = hold previous choice on a miss; no-fallback = frame lost |

The 40 dB silence threshold, 50 ms edge, 100-800 ms pause and 500 ms absence windows are **measurement
parameters, not naturalness thresholds**; they define the event so runs are comparable.

## Case matrix (deterministic, seed-hashed)
Scenarios: `overlap`, `target_absent`, `pause_swap` (interferer fills target pauses), `switch_reacq`.
Crossed with both target roles (A, B), one documented voice bucket (`same`), target level 0/-10/-15 dB
relative to the interferer, room `matched`/`mismatched`. Planned: 480 synthetic cases (400 dev, 80
heldout; `target_absent` only at 0 dB). Split is by source group; heldout needs `--final-heldout`
and must not be used for tuning. With real data the room condition must also be disjoint between dev
and heldout; the prototype notes this but only splits by group.

## Uncertainty and causal claims
* Cluster bootstrap over **source groups** (cases from one speaker are correlated), 95% percentile CI.
* Attribution = paired per-case difference to the full pipeline, with the same bootstrap.
* Ablations: oracle selector, oracle separator, no post stage, no fallback. A gain from an oracle
  stage is the upper bound on what fixing that stage can buy, under the other stages as they are.
* Ablations are interventions on the pipeline, but interactions (selector x AGC) are not separated;
  a 2x2 factorial over (selector, post) is the follow-up when a real run shows both matter.

## Scientifically meaningful vs invented
Meaningful, to run on real systems: oracle-ceiling gap, preservation-before-selection, paired stage
ablations, reacquisition latency distribution, pause-swap rate at hysteresis hold in {0,3,6} frames,
false-speech seconds per target-absent minute, deadline-miss burst length with and without fallback.
Not defensible without listening tests: any universal "natural" threshold (e.g. "< 100 ms onset clip is
fine", "AGC swing < 6 dB"). Report the curves; let humans set pass lines per use.

## Prototype results (synthetic, 24 dev cases, seed 248, 9 groups)
Preservation 10.9 dB [9.5, 12.2]; oracle selector 8.8 [7.8, 9.8] vs full 8.4 [7.3, 9.5]; oracle separator
12.7 [11.4, 14.3]; false-accept rate 0.29; pause-swap rate 0.17; reacquisition 60 ms; deadline-miss
0.8%; no-fallback costs 0.3 dB. These reflect the simulator's injected knobs (onset delay 6 frames, hold 3,
p_false 0.4, 4 ms +- 2.5 compute) and show only that the metrics recover the injected faults.

## Consumer commands
```
.venv/bin/python experiments/theory/T-248/bench_design.py plan
.venv/bin/python experiments/theory/T-248/bench_design.py run --max-cases 24 --seed 248   # resumable
.venv/bin/python experiments/theory/T-248/bench_design.py report --max-cases 24 --seed 248
```
Outputs: `$HEARING_RESULTS_DIR/overnight_prep/T-248/<split>-seed<N>/` (per-case atomic JSON,
`manifest.json` with config hash and source hash, `summary.json`). Caps: `--max-cases` (<=300),
`--time-cap-s`, `--byte-cap`; a torn or stale-config checkpoint is recomputed, a case error stops the
run (`failed:<case>`), no retries.

To use on real systems: replace `selector_model` / `render` with recorded per-frame choices and
shadow-rendered component streams (target-only and interferer-only passes through the same stages);
the metric functions take arrays and spans and need no change. Real references must record speaker,
utterance, interval and hash (out of scope here).
