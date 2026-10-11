# Optional isolation: decision card and freeze gate (T-208)

Offline, report only. Nothing here was run or measured by T-208: no model, no audio, no hardware, no upload, no dependency, no runtime source, config, registry, active profile or checkpoint change. Every number below is quoted from the linked reports (read 2026-10-09) with its original conditions. Nothing is promoted and no runtime switch is added.

## 1. Decision

**RETAIN BASELINE.** No finalist exists. The optional two-male enhancement does not ship, is not on the stage script, and is not claimed.

Why: no candidate has the evidence the bar needs (section 3), and the three follow-ups that could have produced a finalist did not deliver one: the matcher A/B (T-207) and the adaptation probe (T-204) are **Blocked** on the board with no result; the conditioned-TSE track (T-203) has no run at all. The one measured live-path run (T-206) failed its deadline and says nothing about the separator or matcher.

**Core proceeds without the bonus.** The cafe-first demo (PLAN.md: one chosen saved person over cafe noise, Release/Raw, measured tier and delay) is the **fallback and the only required result**. Nothing in this track gates it.

## 2. Evidence reviewed (run and source links)

| Source | What it measured | Result | Limit |
|---|---|---|---|
| T-200 `docs/research/ISOLATION_DIAGNOSTIC.md` (`eval/isolation_diagnostic.py`), scene `t141/aryan/demo-team-same`, both roles, 0 and -10 dB, MPS, 1 run | oracle vs actual selection, per-window failure class | Oracle SI-SDR 12.7-18.2 dB; actual choice equals oracle in 50 of 50 windows; zero separation, selector or deadline failures; compute max 0.6 s per 2 s window at hop 1 s | One scene, no noise, `--noise`/-5/-15 dB not run. "Final mixer" was a proxy. Oracle row uses the clean target. No word retention |
| T-201 `docs/research/PROFILE_LEARNING_CURVE.md`, 1/2/4/8 confirmed 1.5 s examples, frozen ECAPA, disjoint holdout, 1 run | aggregation of examples | Clean TPR 1.00 both voices; only Aryan's quiet target has headroom (+0.1 to +0.3 TPR at 1 s crop, 0.11 at 2 s, 1-3 windows); not monotone in n; `reference_full` (existing profile) matches or beats every variant at 2 s. Verdict: no promotion, `embedding_gain_only` at best | 9-21 windows per cell; Aryan and Joseph on different mic paths so optimistic; no separator, ASR, words or delay |
| T-206 `docs/research/ISOLATION_LIVE_PATH.md`, role A, -10 dB, babble, `demo-team-hard`, window 2 s / hop 1 s, MPS, exclusive lock, 1 condition | real mixer and live-deadline attribution | **15 of 15 target-active windows missed the 1 s hop** (compute mean 2.64 s, max 3.09 s, machine load 3-13), so every window played raw-mix fallback; all stages equal raw (SI-SDR -11.6 dB, SI-SDRi 0.0); oracle stream -13.0 dB, 4 wrong-speaker windows. Attribution: `scheduler_failure` | 7 other conditions and `--live` wrapper not measured; floor-24 counterfactual never reached a selected window; no transcripts |
| T-203 `docs/research/CONDITIONED_TSE_FEASIBILITY.md` | REAL-TSE checkpoint feasibility | Inference NOT VERIFIED; weights only on Google Drive, not fetched; licence unstated; trained on clean Libri2Mix; no streaming API seen; conditional go only after the core demo | Research only, no numbers |
| T-204 (adaptation probe), T-207 (1 s vs 2 s crop on separated speech) | planned | **Blocked on the board; no report exists in this repo.** Recorded as not available, not as a negative result | n/a |

Earlier context quoted by those reports: T-141 two-male overlap wrong-speaker time 6-25%; T-069 male/male selection accuracy 0.742 (1 s) and 0.804 (2 s) against a 0.93-0.95 candidate ceiling, NO-GO. T-200/T-201 could not reproduce that gap on clean real stems, so it points at overlapped or synthetic windows, separation, or the mic path. That is unresolved.

## 3. Criteria for any improvement claim

| Criterion (task bar) | Status | Explanation |
|---|---|---|
| Level-matched human listening, both target roles, quieter target, swapped positions | **PENDING (human)** | No listening session is recorded anywhere. Not approved on anyone's behalf |
| Target-word retention (independent transcripts) | **UNMEASURED** | T-206 `word_check.status = missing_transcripts`; T-201 has no ASR; T-200 uses SI-SDR. SI-SDR and "no wrong voice" do not prove words were kept |
| False accepts (wrong speaker, target absent) | **PARTLY MEASURED, not on a finalist** | T-201: wrong-speaker FAR 0.00-0.10, absent FAR 0.00-0.05 on clean memos, 9-21 windows. T-200: absent-window leak -49.7 dB (5 windows). Not on mixed live audio with a finalist |
| Tolerable measured delay | **FAILED on the only live-path measurement** | T-206: compute 2.64 s mean against a 1 s hop under load; fell back to raw on 15 of 15 windows. T-200 saw 0.4 s per window on MPS but the loop was not run live. A quiet-machine deadline result does not exist |
| Both roles, quieter target, swapped positions, mic-matched | **PARTIAL** | T-200 both roles at 0/-10 dB on one synthetic scene; T-206 one role at -10 dB; swapped physical positions, the demo mic and the real room: none |
| A named finalist vs baseline on the same held-out set | **NOT MET** | None selected; the candidates differ in untested ways (crop length, REAL-TSE, adaptation) |

Failure explanations: delay failed because the 2 s window separator needs more than the hop on a loaded Mac; the measurement is confounded by that load, so it neither proves nor disproves quiet-machine feasibility. Words and listening are unmeasured, not failed. The earlier two-male overlap figure (6-25% wrong-speaker) means perfect rejection must not be claimed.

## 4. Human action (only if someone chooses to continue)

1. Nothing is required for the core demo.
2. Optional, after core rehearsal: on a quiet machine, run the T-206 command from `ISOLATION_LIVE_PATH.md` (`--noise --live`, both roles, ratios 0/-10) under `swarm-lock --exclusive`, with transcripts. Re-open this card only if deadline misses are zero **and** target-word retention holds.
3. A listener (Aryan or Joseph) hears raw / baseline / candidate level-matched through the actual headphones and records yes/no per role and per position. That record is the human approval; this card does not stand in for it.
4. A rented GPU, the REAL-TSE Google Drive download, or any new dependency needs a human decision first. None is assumed.

## 5. Freeze and rollback

- The demo clock is not in the repo: HACKATHON_PLAN.md gives only a 24-hour relative schedule (freeze of selected profiles and settings in hours 10-14; final freeze in hours 21-24) and the Saturday 2026-10-10 14:00-16:00 UTC T-171 mic test. So the gate is relative:
  - **Optional isolation work stops no later than the start of the profile/settings freeze (hour 10 of the 24 h plan), and in any case before the T-171 mic gate window opens on Saturday 2026-10-10 14:00 UTC if that comes first.** Nothing from this track enters after that point.
  - When the humans know the demo time, set the stop to demo time minus 14 h (hour 10) and write it on the stage card.
  - Any trial inside the window is time-boxed: stop at 45 minutes if executable weights, transcripts or a quiet machine are missing (T-203 rule).
- Rollback is trivial because nothing changed: no runtime, config, registry, active profile or checkpoint was touched, and no proposed profile (T-201 `exp-nN`, gitignored, `promoted:false`) is live. If something is adopted later, keep the original profile as the anchor version, switch only through the `current` pointer, and restore the pointer to revert.
- Never in the demo: a runtime switch for this track, a new live model, or a checkpoint promotion.

## 6. Claim wording

Allowed, and only these:

- "Choose a saved voice profile and hear that person over cafe noise; Release returns to Raw." (core, grounded in PLAN.md and the core acceptance, not in this track.)
- "In an offline test with two male voices, a saved-profile matcher picked the right voice in every test window of one scene; we have not tested it in a real room."
- "Personalization is a research direction: it would improve only if consented, versioned profile updates beat a disjoint held-out set with rollback. Our test so far showed no reliable gain."

Not allowed: that the system isolates or separates two similar voices, perfect rejection, preserved words, "learns your voice over time" or continuous learning, any dB figure as a demo result, or that gender classification identifies a person.

Two things that must stay separate:
- **Existing profile aggregation**: the saved profile already averages embeddings of recorded pieces. T-201 shows more averaging gave no reliable gain over it. This is not learning.
- **Learned weight adaptation** (fine-tuning a head or model, T-204): not run, no result, no weights. Only after consented, user-confirmed examples, on distinct held-out voices and utterances, with versioning and rollback, and never from the system's own uncertain output.

## 7. Remaining limitations

One synthetic scene and one live-path condition; one run each, no repeats; compute measured under varying load; no real-room, demo-mic or swapped-position recording; no word retention; no human listening; T-204 and T-207 produced no results; T-069's male/male selection gap is unexplained; strict pause, same_bucket and tier hooks are untested offline; gender is not identity.
