# Pick head on separated streams: frozen ECAPA cosine vs ridge/LDA, small MLP and bounded personal adaptation (T-437, Muse C)

Status: **HYPOTHESIS. Nothing below has run.** No numbers in this card come from a head. Every number cited is a
prior measurement with its source. No product default changes; nothing here is wired into `hearing/`.
Index as `status=hypothesis, adoption_scope=none` (KB record at the end).

Template note: the task names `selective-hearing-research/research/TEMPLATE.md`. That repo is not on this host
(`/Users/joeroche/Developer/selective-hearing-research/` does not exist on Aryans-MacBook-Pro), and no `TEMPLATE.md`
exists anywhere on disk. This card follows the field list of the KB record template in
`docs/research/KNOWLEDGE_BASE.md` and the layout of `research/afw-stage2-heldout.md` (T-398). Re-flow it into
TEMPLATE.md when the research repo is in reach; the content does not change.

## 1. Question

The T-374/T-387 fixed-order MossFormer2 separator separates the Aryan+Joseph pair well, but the **ECAPA pick**
(stream with the higher raw cosine to the target's profile) chooses the wrong stream much more often than the
fixed slot does. Can a small head on frozen ECAPA embeddings of the *separated streams* close part of that gap
while not getting worse on speakers the separator and the head never saw?

## 2. Hypothesis (falsifiable)

H1. On the T-387 separator's held-out team rows, a head trained on **separated-stream** ECAPA embeddings (the
domain the pick runs in) lowers pooled wrong-speaker % against raw cosine by **at least 5 points absolute**, with
the same direction in 3 of 3 seeds, and is **not worse by more than 2 points** on any speaker-disjoint generality row.

H0 (expected by default, given T-291/T-307/T-341/T-357): no head arm meets both conditions; raw cosine stays the pick.

What is new against the prior negatives: every earlier head (T-273/T-305 TeamProjectionHead, T-307 AP head, T-336
logistic head, T-357 residual adapter) was trained on **clean crops** and evaluated either on clean crops (where raw
ECAPA is at or near ceiling) or on a different separator's streams. This card trains and tests in the same domain:
streams of one frozen separator checkpoint, picked in live windows (1.0 s window, 0.5 s hop).

## 3. Prior evidence this must account for (cited)

| Source | Finding | What it forces in this design |
|---|---|---|
| `docs/T-387_fixed_wide_sir.md` (T-387, iLab A100 job 435775) | aryan:joseph holdout-sir0-cafe: fixed2/ecapa **14.5 %** wrong [7.2, 23.2] (69 windows) vs fixed2/fixed **2.9 %** [0.0, 7.2]; joseph:aryan 9.7 % vs 2.8 % (72). grid-s4:grid-s1 fixed2/ecapa **81.1 %** wrong, SI-SDRi −29.2 dB; grid-s1:grid-s4 15.7 %. | Headroom exists on team rows (≈ 7–12 points). On the grid rows the separator itself fails, so a head cannot be judged there without an oracle-pick ceiling (A∞). |
| `docs/decisions/T-374.md`, `experiments/gpu/T-374/README.md` | Separator trained on team takes profile-02..05, val profile-06; holdout-01/02, same-01/02, profile-01/07, fixture/cafe takes never read. Fixed pick shipped team-only, default off. | Head split must not reuse separator-train takes for validation or test (section 5). |
| `docs/decisions/T-307.md`, `docs/research/AP_HEAD.md` | AP and CE heads (32-d) on 30 LibriSpeech speakers lose to raw ECAPA on real data: team clean flip raw **2.6 %**, LibriSpeech held-out raw **0.17 %**, hard male/male pair raw **0.0 %**; pitch-shifted crops cause most flips. | Public head training needs ≥ 251 speakers (train-clean-100 on iLab), not test-clean's 30. No 32-d bottleneck as the only MLP arm. Pitch-shift augmentation is excluded from positives. |
| `docs/decisions/T-291.md` | `team_slices.npz` packages ECAPA embeddings of **clean** non-overlapping team slices (VAD peak ≥ 0.02, RMS ≥ 0.005). | Usable only as the clean-domain control arm, not as head-training data for the stream pick. |
| `docs/research/T341_TEAM_HEAD_FORENSICS.md`, KB-341 | TeamProjectionHead (14,560 params) "97.5 %" was 97 test windows, 2-way, no threshold; on separated streams 85.4 % top-1 vs raw **90.1 %**. | Clean-crop gains do not transfer. Test denominators must be stated; every arm is scored through the same separated-stream windows. |
| KB-357 (T-357 operator pilot, `results/T357-operator-20261010-1824/metrics.json` sha256 `7aabdd12…596b04`); T-357 itself is **Changes Requested, not on main** | 164,544-param residual adapter: raw and adapter both 10/10, CEILING_STOP; 0.1035 ms/query warmed adapter. | A ceiling stop is built in (section 8): if raw is already within 3 points of the oracle pick, the row cannot show a gain and is not counted. Nothing from T-357's branch is used. |
| `docs/decisions/T-407.md` | Learned continuous gate on v3 separator features not adopted: on aryan stranger-sir-6-cafe wrong doubled (7.6 vs 3.8 %). | The kill rule weighs "partner let in" higher than "target muted"; stranger rows are mandatory test rows. |
| `docs/BENCHMARKS.md` line 476–478 | ECAPA embed p50 18 ms (MPS); MossFormer2 + ECAPA compute p50 235–253 ms per balanced 1 s / 0.5 s window. | Head latency budget is small against this: ≤ 1 ms p95 CPU per decision. |

## 4. Arms

All arms consume the same per-window features from one dump of the **frozen T-387 checkpoint** (`fixed2`, iLab
`~/tmp/T-374b/run_fixed/best.pt`, 223 MB; sha256 to be recorded at dump time) and the frozen ECAPA
(`spkrec-ecapa-voxceleb`, already in `docs/MODELS.md`). Per window: `e0, e1` (192-d unit ECAPA of the two
rescaled streams), `p` (192-d unit ECAPA of the target's enrollment profile), label `y = argmin_k wrong(stream_k)`
from the reference projections that `eval_sep.py` already computes.

| Arm | Score for stream k | Trainable params | Fit |
|---|---|---|---|
| A0 raw (baseline) | `cos(e_k, p)` (exactly `eval_sep.py`'s `ecapa` pick) | 0 | none |
| A0f fixed slot (team only, reference) | slot 0 = Aryan, 1 = Joseph | 0 | none |
| A∞ oracle pick (ceiling) | `y` itself | 0 | none; measures what any pick could reach on this separator |
| A1 LDA/WCCN cosine | `cos(W e_k, W p)`, W = within-class whitening (shrinkage 0.1–0.5) then LDA to 64-d | 0 learned by SGD (closed form) | public speakers + team head-train streams |
| A2 ridge pairwise | logistic/ridge on `[cos(e0,p) − cos(e1,p), (e0 − e1) ⊙ p]` (193-d), L2 λ ∈ {0.1, 1, 10, 100} | 194 | closed form / L-BFGS |
| A3 MLP projection | `cos(g(e_k), g(p))`, g = Linear(192,128)-LN-GELU-Linear(128,64), L2-norm; AP loss (T-307 recipe) on streams | 33,216 | AdamW, ≤ 2,000 steps, early stop on val |
| A3b 1-layer projection | g = Linear(192,64), L2-norm | 12,352 | as A3 |
| A4 bounded personal adaptation | best of A1/A3 plus a per-profile residual `p' = p + U v`, U = top-8 within-class directions from public data, v ∈ R⁸ fit on that person's enrollment take only, L2 tether to 0, ≤ 50 steps, ‖v‖ ≤ 0.2 | 8 per enrolled person | enrollment take only (profile-01) |
| C clean-domain control | A3 trained on `team_slices.npz`-style clean crops (T-291 recipe) | 33,216 | as A3 |

A1 and A2 run first because they are cheap and cannot overfit 10k windows the way an MLP can. A3/A3b run only if
A1 or A2 shows ≥ 2 points on val (stop rule S2). C exists to show whether a gain comes from the stream domain.

## 5. Data, splits and leakage guard

### Team audio on laptop-a (`$HEARING_FIXTURES_DIR/team/`, checked Oct 10 2026)

| Speaker | Takes (duration s, sha256 prefix) | Provenance |
|---|---|---|
| aryan | profile-01 (50, `2ef0c878bdc9`), profile-02..07 (120 each), holdout-01/02 (20 each), same-01/02 (40 each), fixture-01 (58.7) | `meta.json`: Mac Voice Memos, **Oct 9, one session**; cuts disjoint |
| joseph | profile-01 (15.7, `e72d9fc70920`), profile-02..07 (120 each), holdout-01/02, same-01/02, fixture-01 (48.1), fixture-02-namecall (27.8) | `meta.json`: Voice Memo via Q-372 pack (sha256 `7e8feb4b…`), **one session**; cuts disjoint |
| ahmad | profile-01/02 (120 each), holdout-01/02 (20 each), same-01/02 (40 each) | no `meta.json`; **never read by the T-374/T-387 separator** |

All sha256 prefixes come from `sha256sum` of the WAVs on this host; the dump script records full hashes.

### Split (by take; a session split is impossible with this data)

| Role | Team takes | Public | Why |
|---|---|---|---|
| Separator train (frozen, not ours) | aryan/joseph profile-02..05 | DEMAND PCAFETER ch01 train seconds; LibriSpeech babble speakers `BABBLE_SPK["train"]` | `experiments/gpu/T-374/train_sep.py` |
| **Head train** | aryan/joseph profile-02..06 mixed with each other (SIR −10..+10 dB, cafe SNR 0–15 dB), streams from the frozen separator | LibriSpeech **train-clean-100** (251 speakers) pair mixes through the same separator | Separator saw 02..05 (streams are cleaner than at test: optimistic); 06 was its checkpoint-selection take only |
| **Head val** (model selection, λ, steps) | aryan/joseph **profile-07** mixes, enrollment = profile-01 | LibriSpeech **dev-clean speakers not in `BABBLE_SPK`** | Neither separator nor head trained on it |
| **Test** (touched once per arm) | `eval_sep.py`'s 8 team rows (holdout-01/02, same-01/02; enrollment profile-01), **plus** stranger rows: aryan:ahmad, joseph:ahmad, ahmad:aryan, ahmad:joseph on holdout-01/02 at SIR 0 and −6 dB cafe (ahmad enrollment = profile-01) | GRID s1↔s4 (`eval_sep.py --grid`); LibriSpeech **test-clean** 10 same-sex pairs (5 M/M, 5 F/F), each speaker disjoint from head train/val and from `BABBLE_SPK["test"]` | Speaker-disjoint generality: no test speaker, take or noise second appears in head train/val |

### Leakage guard (the dump script asserts each and writes the result into `run.json`)

1. Take-level: the sets of (speaker, take) in head-train, head-val and test are disjoint; any overlap aborts.
2. Enrollment: test rows enroll from profile-01 only; profile-01 never appears in head train or val as a stream source.
3. Speaker-level (public): LibriSpeech speaker IDs of train/val/test and of each `BABBLE_SPK` split are disjoint; GRID s1/s4 appear only in test.
4. Noise: cafe seconds follow `train_sep.py`'s time split (test = PCAFETER ch01 240–300 s); head-train never uses test seconds.
5. Hash: every input WAV's sha256 is logged; a repeated hash across roles aborts (catches renamed copies).
6. A4 adapts only on its own enrollment take; its v is reset per test row; it never sees the test stream.
7. Seeds: 3 seeds (0, 1, 2) for every learned arm; scene render seed fixed at 0 as in `eval_sep.py`, so all arms see identical windows.

## 6. Metrics

Primary (per row and pooled over rows of one group, with window counts):
- **wrong-speaker %**: target-active 1.0 s windows (0.5 s hop) whose pick holds more interferer than target energy (`eval_sep.py` definition); 95 % bootstrap CI over windows.
- **gap closed** = (A0 − arm) / (A0 − A∞) on that row; reported only where A0 − A∞ ≥ 3 points.

Secondary: SI-SDRi and leak dB of the picked stream (as `eval_sep.py`); pick margin `s_target − s_other`; head
inference time per decision (p50/p95, CPU and MPS, 1,000 calls, under `swarm-lock --exclusive`, `uptime` load
recorded); parameter count.

Statistics: mean ± SD over 3 seeds (`eval.paired_stats seeds`); exact McNemar on per-window correct/wrong against
A0 on **non-overlapping windows only** (every second window, so windows do not share audio), via
`eval.paired_stats mcnemar` after exporting T-323-style audit JSON. `eval/paired_stats.py` itself warns overlapping
windows violate independence; the p-value is a gate, not a claim of replication.

## 7. Compute and data budget

| Step | Where | Budget (hard cap) |
|---|---|---|
| D0: reproduce `eval_sep.py` baseline for fixed2 (A0, A0f) | iLab A100, job 435775 via `srun --overlap` | ≈ 2 min (T-387 reported ~2 min eval) |
| D1: dump per-window features, all roles | same | ≤ 25 GPU-min (est. 10k team head-train windows, 2.4k val, 1.1k team test, 4k public train, 1k public val/test) |
| F1: fit A1, A2 (3 seeds) | CPU (laptop-a or iLab CPU step) | ≤ 5 min |
| F2: fit A3, A3b, C (3 seeds), only if S2 passes | CPU or MPS | ≤ 15 min |
| F3: A4 on top of best global arm | CPU | ≤ 2 min |
| T: score every arm on test once; latency probe | CPU + `swarm-lock --exclusive` for latency | ≤ 5 min |

Total ≤ 30 A100-min and ≤ 30 CPU-min; stop at 45 GPU-min regardless of state. Storage: the npz dump stays on iLab
(team-derived, as T-407 did); only aggregates, hashes and fitted head weights (< 200 KB) are copied to
`$HEARING_RESULTS_DIR/T-437/`. No audio leaves iLab; nothing is committed except the card and code.

Memory: inference only, MossFormer2 forward at batch 1 (training at batch 4 needed ~24 GB, T-374); run next to
the hearing-remote servers without `--mem` changes.

## 8. Stop and kill rules

- **S1 ceiling stop (per row)**: if A0 − A∞ < 3 points, the row is reported but excluded from H1 (T-357's lesson).
  If fewer than 4 of the 8 team rows remain, stop: the pick is not the bottleneck on this separator.
- **S2 cheap-arm gate**: if neither A1 nor A2 improves pooled team **val** wrong % by ≥ 2 points, skip A3/A3b and
  record H0 for the closed-form family; still run A3 once at seed 0 to rule out a nonlinearity-only effect, then stop.
- **S3 overfit stop**: an arm whose val gain is ≥ 5 points but test gain < 1 point is recorded as overfit, not as a result.
- **Kill (H1 false)**: no arm meets all of (a) pooled team test wrong % ≤ A0 − 5 points, McNemar p < 0.05 on
  non-overlapping windows, 3/3 seeds same sign; (b) every stranger row (Ahmad) and every public row (GRID both
  directions, test-clean pairs) no worse than A0 + 2 points; (c) ≤ 100k params and ≤ 1 ms p95 CPU per decision.
  Then the line closes: write the negative into the KB and do not escalate to larger heads or more steps.
- **Promote (H1 holds)**: the arm becomes a *candidate for a listening check* only. It does not change a default;
  Joseph decides after the live check. It also must not beat A0f on team rows to be useful: its value is on rows
  where the fixed slot does not exist (strangers, other speakers).

## 9. Runnable commands

Existing code, runnable now on iLab (paths as in T-374's README; `ssh` through the `cm-ilab` ControlMaster):

```bash
# D0: baseline A0 / A0f for the frozen T-387 checkpoint
cd ~/tmp/T-374 && srun --jobid=435775 --overlap ~/hearing-remote/.venv/bin/python code/eval_sep.py \
  --ckpt fixed2=$HOME/tmp/T-374b/run_fixed/best.pt --data data --grid data/grid \
  --ecapa ~/hearing-remote/models/spkrec-ecapa-voxceleb --out ~/jobs/T-437/eval_a0.json
sha256sum ~/tmp/T-374b/run_fixed/best.pt > ~/jobs/T-437/ckpt.sha256
```

New code for Muse B / Codex to write (interface fixed here; neither file exists yet):

```bash
# D1: research/T-437/dump_pick_features.py  (imports scene/separate/Ecapa/load from experiments/gpu/T-374)
srun --jobid=435775 --overlap ~/hearing-remote/.venv/bin/python research/T-437/dump_pick_features.py \
  --ckpt ~/tmp/T-374b/run_fixed/best.pt --ecapa ~/hearing-remote/models/spkrec-ecapa-voxceleb \
  --team ~/hearing-data/team --data ~/tmp/T-374/data --grid ~/tmp/T-374/data/grid \
  --librispeech /freespace/local/$USER/librispeech/LibriSpeech --roles head_train,head_val,test --seed 0 \
  --out ~/jobs/T-437/features.npz --run-json ~/jobs/T-437/run.json
#   features.npz: role, row, window_k, t, e0, e1, p (float32 192), wrong0, wrong1 (bool), y (int), speaker ids
#   run.json: input sha256s, split sets, the 7 leakage assertions (all must be true), git sha, device

# F1-F3, T: research/T-437/fit_heads.py  (CPU, numpy/torch; no audio)
~/hearing-remote/.venv/bin/python research/T-437/fit_heads.py --features ~/jobs/T-437/features.npz \
  --arms A0,A0f,Aoracle,A1,A2 --seeds 0,1,2 --out ~/jobs/T-437/heads_a1a2.json
~/hearing-remote/.venv/bin/python research/T-437/fit_heads.py --features ~/jobs/T-437/features.npz \
  --arms A3,A3b,C,A4 --seeds 0,1,2 --out ~/jobs/T-437/heads_a3.json      # only if S2 passes
#   heads_*.json: T-307 shape (seed -> eval_set -> head -> metric) for eval.paired_stats seeds,
#   plus per-arm T-323 audit files (non-overlapping windows) for eval.paired_stats mcnemar

# Statistics (repo root, any host)
.venv/bin/python -m eval.paired_stats seeds $HEARING_RESULTS_DIR/T-437/heads_a1a2.json
.venv/bin/python -m eval.paired_stats mcnemar $HEARING_RESULTS_DIR/T-437/audit_A0.json $HEARING_RESULTS_DIR/T-437/audit_A1.json

# Latency (laptop-a, exclusive lock, record uptime)
uptime; swarm-lock --exclusive -- .venv/bin/python research/T-437/fit_heads.py --latency \
  --head $HEARING_RESULTS_DIR/T-437/best_head.pt --calls 1000
```

Both new scripts need an offline test with synthetic embeddings (no weights, < 2 s) before anything is queued
(KB section 4: no untested scripts on Slurm).

## 10. Handoff package (Muse B executes, Codex reviews)

1. Muse B: write `dump_pick_features.py` and `fit_heads.py` to the interface above, plus offline tests; Codex reviews the code before any iLab run.
2. Run D0 and compare with `docs/T-387_fixed_wide_sir.md` (fixed2/ecapa aryan:joseph holdout-sir0-cafe 14.5 %, 69 windows). A mismatch > 1 window means the scene or checkpoint differs: stop and report.
3. Run D1, check `run.json` (7 assertions true, non-empty hashes), then F1 → S2 → F2/F3 → T, in order, inside the 45 GPU-min cap.
4. Commit results to a run branch in the research repo (`research/run-T437-<jobid>`), with this card's status changed to `measured` and the numbers filled; Codex B reviews hashes and conditions before KB admission.
5. Whatever the outcome, the KB gets one record. A negative closes this line (section 8).

## 11. Missing data and unknowns (named)

- **Q-1662 source manifest**: not available on this host. Needed: a signed list of every team take with recording date, room, device and session ID. Until it arrives, the card assumes one session per speaker (from `meta.json`), so all team tests are **within-session**; a gain may not survive a new day or room.
- **Second-session team recordings** (each of Aryan, Joseph, Ahmad, different day/room/mic, ≥ 2 min each, with clean stems): needed for a real session-level split. Not present.
- **Live session captures** (`results/sessions/*/in.wav`) have no stems; their only labels would be an ECAPA proxy, which is circular for an ECAPA head. Excluded.
- **Ahmad meta.json** is missing (date/device unknown).
- **`selective-hearing-research` repo and `TEMPLATE.md`**: not on this host (Joseph's machine). The card is committed in this repo's `research/` instead.
- **T-357 code/split**: Changes Requested, not on main; only its KB-reviewed metrics hash is cited.
- **LibriSpeech train-clean-100 / dev-clean on iLab**: path taken from `experiments/gpu/T-307/run_ilab.sh` (`/freespace/local/$USER/librispeech/LibriSpeech`, which downloads train-clean-100 if absent); whether it is already there is not verified from this host (no non-interactive ssh). dev-clean is not in that script's download list: add it.
- **T-387 checkpoint hash**: never recorded in the repo; D0 records it.
- **Real-audio inference**: none run for this card. No gain is claimed.

## 12. KB record (for Muse to index)

`claim_id=KB-437; author=Claude (laptop-a, T-437, Muse C role); run/source_id=none (hypothesis);
source_locator=research/T-437-pick-head-on-separated-streams.md; artifact_sha256=unknown_reason: not run;
code=experiments/gpu/T-374/eval_sep.py (main 9a86d0fd) + research/T-437/* (not written); model=MossFormer2_SS_16K T-387 fixed2 (hash unknown) + spkrec-ecapa-voxceleb;
data/split=section 5; seed=0,1,2; host/device=iLab A100 (planned); measurement_stage=none; real/synthetic=planned real;
metrics=wrong-speaker %, gap closed, SI-SDRi, leak dB, p95 head ms; integrity_status=not_applicable;
condition_scoped_efficacy=untested; contradicts=none; related=KB-341, KB-357, T-307, T-291, T-407;
verdict=hypothesis; next_task=Muse B section 10; adoption_scope=none`
