# Muse theoretical research portfolio (T-253)

Research roadmap only: nothing here is adopted into production, PLAN.md or live config. No measurements were taken;
every number below is a prior **estimate** (scales in `experiments/theory/T-253/candidates.json`), not a result.

Replay: `.venv/bin/python experiments/theory/T-253/portfolio.py --out-dir <dir>` writes `ranked.json` and
`proposals.json` (task proposals; stdlib only, deterministic, no seed, input sha256 recorded in `meta`).
Score = impact x confidence / (cost x risk). Tests: `tests/test_theory_research_portfolio.py`.

## Ranking (new branches N*, reused prior branches P*)

| Rank | ID | Branch | Score | Blocked on |
|---|---|---|---|---|
| 1 | N1 | Open-set unknown-speaker rejection (calibrated thresholds + cohort score normalisation) | 1.200 | none |
| 2 | P2 | Confidence calibration of match score for gating (T-249 note) | 1.100 | none |
| 3 | N3 | On-device personalisation, local-only robust prototype update with rollback | 0.375 | none |
| 4 | P5 | Learned gated-window floor from calibrated confidence (T-180 follow-up) | 0.375 | depends on P2 |
| 5 | N5 | User-feedback objective vs quality metrics (pairwise listener preference) | 0.200 | human ratings |
| 6 | P4 | Conditioned TSE vs separate-then-select | 0.100 | GPU (Sol) |
| 7 | P1 | Separator fine-tune on target-room data (T-247 note) | 0.093 | GPU (Sol) |
| 8 | N4 | Beamforming front end | 0.083 | multi-mic hardware |
| 9 | N2 | Face/audio association with transient, never-stored match scores | 0.056 | consented video, privacy review |
| 10 | P7 | Camera targeting (subsumed by N2) | 0.045 | hardware |
| - | P3 | Profile learning curve (T-201, already measured; base for N1/N3) | reused | - |
| - | P6 | Two-male isolation grid (owned by T-206/208; not duplicated) | reused | - |

## Independent falsifiable paths useful after the hackathon
1. **N1 open-set rejection.** Falsified if score normalisation gains < 2 points absolute false-accept at 90% target recall, or loses > 3 points recall, on held-out speakers. CPU, existing embeddings, under 10 minutes.
2. **N3 on-device personalisation.** Falsified if EER degradation from 5% contaminated confirmations exceeds the clean-update gain for trimmed-mean updates. CPU simulation on existing embeddings.
3. **N5 feedback vs quality.** Falsified if every objective metric has Kendall tau < 0.3 against listener rankings over >= 30 pairs. Needs a human to rate existing renders.
4. **N4 beamforming** is closed with no work if the demo device exposes one mic channel (human capability query only).
5. **N2 face association** has a privacy gate first: nothing persisted, only a per-window boolean vote; falsified if active-speaker precision does not beat audio-only ECAPA precision on the same clips.

## Evidence and assumptions
- Reused evidence: `docs/research/PROFILE_LEARNING_CURVE.md` (per-profile threshold at 5% dev false accept, frozen ECAPA, only two speakers as negatives), `docs/research/CONDITIONED_TSE_FEASIBILITY.md`, commit 34e5673 (gating floor / UNSURE_DB), PLAN.md (post-HackRU camera targeting; 6-25% wrong-speaker in T-141).
- The T-224..227 Muse reports and the seven prior branches named in the dispatch were **not present in this worktree**; P1-P7 are mapped from repository docs and named dispatch notes (T-247, T-249), so the mapping is an assumption.
- Confidence values are subjective priors; costs are in agent+compute hours on a 1-5 scale.
- Final heldout source groups must not be used for tuning in any proposal.

## Machine-readable proposals
Generated `proposals.json` lists CPU-only proposals (N1, N3, N5) with caps 30 CPU minutes, 32 MiB new bytes, 0 GPU cases, `production_adoption: false`. GPU and hardware items are intentionally not proposed as tasks here.
