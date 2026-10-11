# Arena: autonomous eval + ASHA scheduler (T-331)

Every candidate renders audio from the **same fixed, hashed mixtures**, is scored per mixture,
and is promoted, stopped or sent to a human by an ASHA-style scheduler. Objective metrics never
declare a winner: the last rung ends in `HUMAN_REVIEW`, and only an Arena (blind A/B) tally states
a preference. Nothing here touches `hearing/`, `config/`, `scripts/` or `web/console/`.

```
.venv/bin/python -m experiments.arena run --set tiny --baseline dfn_ll \
    --candidates dfn,gtcrn,passthrough,muted=silence --dnsmos dnsmos \
    --pinned-reference $HEARING_RESULTS_DIR/arena/reference_dfn_ll_tiny.json
.venv/bin/python -m experiments.arena serve <run_dir> --rung 1 --port 0   # blind A/B page
.venv/bin/python -m experiments.arena tally <run_dir> --rung 1
```
Candidates are `id=spec` with `spec` any `eval.bench` pipeline string (`tse:...@2`, `remote(...)`,
...). Runs go to `$HEARING_RESULTS_DIR/arena/<stamp>/` (never git): `results.jsonl`, `state.json`,
`rows/<candidate-key>/<mixture>.json`, `audio/<candidate>/<mixture>.wav`, `arena/rung<r>/`.

## Files
| File | Role |
|---|---|
| `runner.py` | renders a candidate on mixtures (`eval.bench.run_stream`, 10 ms chunks), scores, caches rows |
| `stats.py` | paired **cluster** bootstrap (mixtures sharing source utterances resample together; windows never) |
| `scheduler.py` | pilot throughput -> rung sizes -> rungs with baseline-first, reference check, decisions, ASHA quota, checkpoint |
| `result.py` | canonical record `hearing.arena.result/1` + `validate()` |
| `arena.py`, `serve.py`, `ui/index.html` | blind A/B batch export, stdlib server (`POST /verdict`), tally |

## Metrics (per mixture, kept as four separate Pareto dimensions; no weighted aggregate)
- **intelligibility**: STOI, ESTOI, WER (`--asr`; only where `transcript_verified`, LibriSpeech refs count as verified).
- **naturalness** (auxiliary): DNSMOS P.835 SIG/BAK/OVRL. It can veto a STOP but never prunes, and it
  stops counting once the output drops the target (DNSMOS scores digital silence as clean background).
- **interference**: SI-SDRi vs the clean target stem, wrong-speaker fraction and switches/min (100 ms
  least-squares attribution to target vs interferer), target-present misses and target-absent leak
  (scored jointly), target dropouts (`eval.overnight_output_ab.compute_metrics`).
- **latency**: declared algorithmic latency (lookahead), compute p50/p95 per 10 ms hop, real-time
  overruns (hops > 10 ms), RTF.

## Scheduler rules
1. **Pilot**: each candidate and the baseline render one mixture; cost per audio-second sizes the
   rungs `n_r = n0 * eta**r` (n0 shrinks to fit the budget).
2. **Protected baseline**: `baseline_share` of the budget is reserved and never lent to candidates;
   the baseline renders first on every rung and is never pruned. Each rung it is re-rendered from
   scratch on reference mixtures and compared with the pinned reference (tolerance 1e-3); any drift
   turns every decision into `HUMAN_REVIEW`.
3. **Decisions** vs the baseline on the same mixtures (paired CI, equivalence margins in
   `DEFAULT_MARGINS`): `STOP` needs a quality metric confidently worse and nothing (quality or
   perceptual) better or uncertain; uncertain -> `CONTINUE` (more data, not failure); last rung ->
   `HUMAN_REVIEW` with the trade-off spelled out. Budget shortfall -> `HUMAN_REVIEW`.
4. **ASHA quota** keeps `ceil(n/eta)`; a candidate outside it is cut only if a kept candidate is
   confidently better on every scored metric.
5. **Checkpoint/resume**: `state.json` after every rung (plan, ledger, decisions, promoted
   candidates' checkpoint ids + sha256); rows are cached by content key, so `run --run-dir <same>`
   resumes without re-rendering.

## Canonical record (`results.jsonl`, one line per candidate x rung)
`schema`, `run_id`, `experiment_id`, `candidate{id, kind, model, checkpoint, checkpoint_sha256, spec,
preprocessing, seed, suppression}`, `baseline_id`, `rung`, `inputs{set, mixture_ids, mixture_sha256,
order_seed}`, `per_mixture[{mixture_id, metrics..., output, output_sha256, cluster}]`,
`dimensions{intelligibility, naturalness, interference, latency}`, `metrics` (flat means),
`paired_vs_baseline{metric: delta, lo, hi, n, clusters, verdict}`, `compute{unit, consumed_s, rung_s,
audio_s, candidate_pool_left_s, host, device, git_rev}`, `scorers`, `decision{action, reason,
next_rung_estimate}`, `baseline_reference`, `human_verdict` (null until an Arena tally), `created_at`.
`validate()` rejects `score`/`aggregate`/`winner` fields. Distillation jobs write the same record with
`candidate.kind = "distillation"`, `checkpoint` = the student checkpoint and `compute.unit` = their rung
unit (speakers / steps).

## Measured run (laptop-a, 2026-10-10, `swarm-lock --exclusive`, load avg 5.8 -> 5.5, device mps)
Set `tiny` (16 LibriSpeech two-talker + babble mixtures, 3 s, 2 source clusters), baseline `dfn_ll`,
rungs [4, 16], DNSMOS real. Run dir `eval/results/arena/T-331-tiny-20261010c` (main checkout).

| candidate | rung 0 | rung 1 (16 mixtures) | SI-SDRi | ESTOI | DNSMOS OVRL | wrong-spk | RTF | compute s (cumulative, render only) |
|---|---|---|---|---|---|---|---|---|
| dfn_ll (baseline) | protected | protected | -0.92 | 0.333 | 2.15 | 0.342 | 0.066 | 3.22 |
| dfn | CONTINUE | HUMAN_REVIEW: BAK better, STOI worse | -2.22 | 0.283 | 2.27 | 0.363 | 0.032 | 1.53 |
| gtcrn | CONTINUE | HUMAN_REVIEW: BAK better | -0.73 | 0.313 | 2.15 | 0.357 | 0.030 | 1.38 |
| passthrough | CONTINUE | HUMAN_REVIEW: SI-SDRi better, OVRL worse | 0.00 | 0.332 | 1.47 | 0.350 | 0.000 | 0.003 |
| muted (`silence`, negative control) | STOP (rung-0 means: SI-SDRi -99.7, ESTOI -0.004, OVRL 1.84) | - | - | - | - | - | 0 | 0.002 |

Baseline reference drift 0.0 on every rung of three consecutive runs. The tiny set has only two
source clusters, so most CIs stay wide: that is the intended outcome (more evaluation / human
review), and a real study should use `full` (96 mixtures) or the AV set. Arena batch: 12 blind
trials in `arena/rung1/`.
