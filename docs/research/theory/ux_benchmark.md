# T-255 · Experiment comparison and listening UX (benchmark)

> **Historical round-one proposal, parked post-hackathon.** Recovered on
> 2026-10-10 from commit `f8ac019`; the claims below describe its design intent,
> not a validated listening study. Example scene labels, run IDs and conditions
> are illustrative placeholders, not measured provenance. See the prototype
> README's review blockers for discrepancies between intent and code.
> No audio was rendered, no models introduced, and no private inputs accessed.
> Preserve source-group final holdouts for any future study; do not train on them.
> Current verification is reported by the worker; the final caveat about the
> original clone's missing environment is historical.

## What exists today

Blind listening is already the project's decision procedure for audio quality,
but it runs on ad-hoc rails:

- **T-194** exports "three aligned A/B files per finalist to
  `results/finalists_<stamp>/`" — the raw material for human comparison.
- **PLAN.md §"Make feedback easy"** prescribes `results/listening_<stamp>.md`
  with per-scene tags: **Wrong person**, **Other voice leaks**,
  **Selected voice unclear/robotic**, **Too slow**, **Good** — plus run ID,
  time span, target, mic/positions, settings.
- In practice this means: someone drops WAVs in a folder, Joseph/Aryan listen
  in a media player, and tags go into a Markdown file by hand.

## What's wrong with that (hypotheses [H])

1. **No blinding enforcement [H].** Filenames like `finalist_contrast03_take2.wav`
   leak the condition. Listeners know which knob they're hearing, which biases
   preference toward the "new" thing.
2. **No randomization [H].** Fixed presentation order creates order effects —
   the second clip benefits from the first as reference, or suffers from
   listener fatigue on long sessions.
3. **No aggregation.** Markdown tags don't roll up into "finalist B wins 7/10
   trials on leak suppression." Someone hand-counts.
4. **No alignment guarantee.** "Three aligned A/B files" relies on the exporter;
   nothing in the listening step verifies the clips being compared are the same
   input segment.
5. **Scale ceiling [H].** Works for 2 finalists × 3 clips. Breaks at 5 finalists
   × 10 clips (45 pairwise comparisons) — which is where the overnight sweeps
   are heading.

## Proposed design

A static-page A/B (and A/B/X) harness. No server, no build step — a folder of
WAVs plus a `trials.json` manifest, opened in a browser. Design decisions:

### Trial model
Each trial compares N≥2 conditions on the **same input segment**:
```json
{
  "trial_id": "t01",
  "prompt": "Which clip isolates the target voice better?",
  "input_ref": "demo-team-easy 12.0-18.0s",
  "clips": [
    {"id": "c1", "file": "finalist_a_take2.wav", "condition": "contrast_0.3", "run_id": "..."},
    {"id": "c2", "file": "finalist_b_take2.wav", "condition": "contrast_0.1", "run_id": "..."}
  ]
}
```
`condition` is the truth; the page shows only randomized labels (X, Y, …).
Unblinding happens at export, never during listening.

### Listening flow (per trial)
1. Page presents clips as **X / Y** (randomized order per trial, per listener).
2. Listener must play each clip at least once before rating (enforced in JS —
   cheap blinding hygiene).
3. Rating: forced choice (X better / Y better / tie) + optional per-dimension
   1–5 scales: **clarity**, **leak suppression**, **naturalness**.
4. Failure-mode tags from the PLAN.md taxonomy as checkboxes:
   Wrong person · Other voice leaks · Unclear/robotic · Too slow · Good.
   Tags attach per clip, so "X: other voice leaks" survives unblinding as
   "contrast_0.3: other voice leaks".
5. Free-text comment per trial (optional, capped — long comments don't get read).
6. Progress bar; results accumulate in localStorage so a refresh doesn't lose a
   30-trial session.

### Aggregation
On "Finish", the page exports `results_<listener>_<stamp>.json`:
unblinded per-trial records (condition labels restored). A tiny Python
companion (`aggregate.py`, future work — out of this task's scope) rolls
multiple listeners' files into win-rate tables per condition × dimension,
plus tag frequency per condition. That table is what goes into BENCHMARKS.md
or the freeze decision, replacing hand-counted Markdown.

### What the prototype covers
`web/experiments/T-255/` implements the trial model, randomized blind
presentation, play-gating, forced choice + dimensions + PLAN.md tags, comment,
progress, localStorage checkpointing, and JSON export. It does NOT implement
the Python aggregator, multi-listener merging, or audio alignment verification
— those are explicit follow-ups (see Ideas).

## Minimal useful prototype checklist
- [x] `trials.json` manifest format with blind labels
- [x] Randomized X/Y order per trial
- [x] Must-play-before-rate gating
- [x] Forced choice + 1–5 dimensions + PLAN.md failure tags + comment
- [x] localStorage resume + JSON export with unblinding map
- [ ] Python aggregator (follow-up)
- [ ] Alignment self-check (follow-up)

## Ideas

1. **(no GPU) Build `aggregate.py`.** Reads N listener JSONs, outputs win-rate
   matrix per condition × dimension, tag histogram per condition, and
   inter-rater agreement (Cohen's κ on forced choice). This is the missing half
   of the prototype — without it the JSON exports pile up unread.
2. **(no GPU) A/B/X variant.** Add a hidden reference (X = known good) to catch
   inattentive listeners: anyone rating X worst on >30% of trials gets flagged.
   Standard MUSHRA practice; cheap to add to the trial model as `"reference": clip_id`.
3. **(no GPU) Alignment self-check.** At manifest build time, verify all clips
   in a trial have identical duration (±10 ms) and matching sample rates; refuse
   to start the session otherwise. Kills the "comparing different segments"
   failure mode at the source.
4. **(GPU, only if genuinely relevant) Perceptual metric correlation study.**
   Once human A/B data exists, correlate forced-choice outcomes with DNSMOS /
   PESQ / SI-SDRi on the same clips. If a cheap metric predicts human preference
   (Spearman ρ > 0.7), the overnight sweeps can pre-filter finalists before
   burning human listening time. Needs the A100 only for batch metric
   computation (~30 min); the human data is the long pole.

## Caveats / hypotheses
- All UX claims above are [H] until tested with actual listeners (n=2: Joseph,
  Aryan). The "must-play-before-rate" gating may annoy expert listeners who can
  judge in 3 seconds — consider a "skip gating after trial 5" escape hatch.
- No measurements were taken; there is no prototype validation data yet.
- `verify_fast.sh` could not run (no `.venv` in this clone, ruff missing) —
  reported honestly; JS/HTML untouched by pytest in any case.
