# Optional isolation: real mixer and live-path diagnostic (T-206)

Offline eval only (`eval/isolation_live_path.py`, `tests/test_isolation_live_path.py`). No runtime source, config, registry, profile or checkpoint changed; no audio leaves the machine; nothing is adopted automatically.

## What it adds over T-200

T-200 (`eval/isolation_diagnostic.py`, unchanged) scored its "final mixer" with a proxy formula. This adapter reuses its replay loop and classifier and scores the same recorded input at every stage, each row carrying `class`, `real`, `proxy`, `deployable`:

| stage | class | label |
|---|---|---|
| `raw_mix` | input | baseline for SI-SDRi |
| `oracle_candidate` | `BlindSeparator`, picked with the clean target | oracle, not deployable |
| `actual_selected` | `EnrolledCandidate` (real ECAPA/LDA choice) | real |
| `proxy_mixer` | T-200 `mix_final` | **PROXY**, never called the real mixer |
| `real_mixer` | `hearing.fusion.mixer.Mixer`, 10 ms chunks, per-window `tse_confidence` (residual floor 6 + 34c dB) | real |
| `real_mixer_agc` | `Mixer` then `hearing.audio.agc.Agc` | real |
| `cf_floor24_mixer` | `Mixer(max_duck_db=24)`, confidence ignored | **counterfactual**, not a default |
| `live_wrapper` | `FocusHQ(threaded=False)` + `EnrolledCandidate` + `AutoEnroller.enroll_clip` + `Mixer`, delay-compensated (`--live`) | real; reports the exact error if it cannot run |

Metrics: `sisdr_absolute_db` (output vs clean target, mean over target-active windows) and `sisdri_db` (that minus the raw mix's SI-SDR over the same windows) are separate fields. Target-absent windows report `target_absent_leak_db` (output energy vs mix), never SI-SDR. `wrong_speaker_windows` counts overlap windows where SI-SDR(out, other) > SI-SDR(out, target). Per condition: deadline/raw-fallback/skipped counters, held/paused/vetoed counts, window/hop (default 2 s / 1 s; `--window-s/--hop-s` for others), the real mixer's mean/min residual floor, and a `failure_attribution` of `model_target_loss` / `wrong_selection` / `output_leak` / `scheduler_failure` / `none`, with the existing fix owner for that branch (conditional model trial for separation; T-178 profile/contrast for selection; T-180 floor/AGC for output; hop/compute budget for the scheduler). Coverage flags: both roles (`--roles both`), ratios 0/-5/-10/-15 dB (`--ratios`), `--noise` for the babble stem.

Orchestrator hypothesis to test with this tool: the confidence floor, not the embedding, leaves the loud talker audible (c = 0.32 gives only a 16.9 dB residual floor). `cf_floor24_mixer` vs `real_mixer` on the same selected signal is the direct comparison; it is a counterfactual, and no mixer default is edited here.

## Measurements

One condition measured (role A = scene target `aryan`, target at -10 dB vs the other voice, babble stem added, `demo-team-hard`, window 2 s / hop 1 s, held-out half of the windows for SI-SDR, MPS, run under `swarm-lock --exclusive`; load average 2.9 after the run, 4-13 during the earlier attempts; manifest `.swarm-run/one_cond.json`, not committed). Not measured: roles B, ratios 0/-5/-15 dB, `--live` wrapper replay, target-absent windows (the scene's held-out half had none, `n_target_absent_windows = 0`). A first 8-condition `--live` attempt ran about 19 min for 1.6 CPU-minutes without finishing on the loaded machine and was stopped.

| stage | label | SI-SDR absolute dB | SI-SDRi dB | wrong-speaker windows |
|---|---|---|---|---|
| raw_mix | input | -11.6 | 0.0 | 10 |
| oracle_candidate | oracle | -13.0 | -1.4 | 4 |
| actual_selected | real | -11.6 | 0.0 | 10 |
| proxy_mixer | PROXY | -11.6 | 0.0 | 10 |
| real_mixer | real | -11.6 | 0.0 | 10 |
| real_mixer_agc | real | -11.4 | 0.2 | 10 |
| cf_floor24_mixer | counterfactual | -11.6 | 0.0 | 10 |

Counters (all 29 windows of the clip): 15 of 15 target-active windows missed the 1 s hop (compute mean 2.64 s, max 3.09 s per 2 s window), so every window played the raw-mix fallback, which is why `actual_selected` equals `raw_mix` and all four mixer stages equal it. Real mixer residual floor (from the windows that carried a confidence): mean 13.1 dB, min 6.0 dB. Attribution: `scheduler_failure` (the only branch this run can support).

What this does and does not show: under this load the window loop cannot keep its 1 s hop on MPS, so the run says nothing about model target loss, selection, or the mixer floor (no window reached the mixer with a selected stream). The oracle row is the only separator evidence: the best of the two streams has -13.0 dB SI-SDR against the -10 dB-scaled target, i.e. at this ratio the separator output did not recover the quiet target in this scene (4 wrong-speaker windows even for the oracle stream); this is one scene, one run, not repeated, with the clean target used for the pick. Deadline status on a quiet machine, and the floor-24 counterfactual with selected windows, remain unmeasured. Offline arithmetic is not substituted. Rerun on a quiet machine:

```
.venv/bin/python -m eval.isolation_live_path --scene $HEARING_FIXTURES_DIR/t141/aryan/demo-team-hard \
    --profiles $HEARING_FIXTURES_DIR/profiles --noise --live --out <path>.json
```

The dry run (`--dry-run`, constructed stems and fake runners, labelled FAKE) only verifies that the adapter invokes the real `Mixer` and `Agc`, labels proxy/oracle/counterfactual rows, attributes constructed failures correctly and writes the manifest; it is not separation evidence.

## Gaps stated, not inferred

- Strict pause/gate session hooks, same_bucket/active tier, negative-active routing and profile readiness in a live session: not exercised (needs the session; the manifest `gaps` field says so; only `AutoEnroller.source/ready` and the candidate's `held/paused/vetoed` counters are recorded).
- Independent target-word checks: no transcripts supplied, `word_check.status = "missing_transcripts"`. SI-SDR and leak do not prove words were preserved.
- A digital mix of one scene is preliminary. A room conclusion needs independent same-mic/session recordings of the stage presenters.
- The mixer's confidence is held per hop slot (window i's confidence covers `[i*hop, (i+1)*hop)`); the live stage applies it per emitted chunk, so this is an approximation outside `live_wrapper`.
- Gender is not identity: scenes are two male voices matched by profile only.
