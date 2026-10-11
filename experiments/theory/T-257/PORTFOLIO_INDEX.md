# Muse parent portfolio (T-257): index of delegated theoretical branches

Index only. No production adoption, no measurements taken here, nothing trained, no GPU. Every theoretical outcome in
the child branches is a **hypothesis**. Commit hashes were read with `git log` in this worktree (local branches, Oct 10 2026).
Not to be confused with `docs/research/theory/research_portfolio.md` (T-253's ranking); read that for scores.

## What this run did, actual vs planned
| Planned (task text) | Actual |
|---|---|
| Spawn <=4 Muse subagents for T-246..T-253 plus UX branches | **Not done in this run.** This worker had no Muse runner or subagent quota; I spawned none. The child branches below already existed locally, produced by earlier runs. I indexed them only. |
| Acknowledge via parent feedback | Not possible from the worktree; this report is the trace. |
| Findings to Sol T217 and Fable Q570 | Sent as `messages` / `notes_for_reviewer` in the report (T-217); Q570 is not a task id I can message: exclusion. |
| Attach full text to board pages | Not done (no board access); files are in the branches listed. |
| Keep Blocked reservations until Review | Not touched. |

## Child branches (local tips; "own commit" is the latest non-merge commit)
| Task | Branch tip | Own commit | Artifact in branch | Status |
|---|---|---|---|---|
| T-246 projection head on frozen ECAPA | 51c8c59e | 1dd56163 | docs/research/theory/projection_head.md, experiments/theory/T-246 (ablation_report.json is synthetic) | in main-merged docs |
| T-247 separator fine-tune plan | a322e14a | 4044d8f8 | separator_finetune.md, T-247/plan.py | local |
| T-248 failure-localizing benchmark | d8dd2046 | 0f598c8c | benchmark_design.md, T-248/bench_design.py | local |
| T-249 selection calibration + reject | a130df97 | 7c59385a | selection_calibration.md, T-249 | local |
| T-250 active enrollment | 05dee5f8 | aaeea0e1 | enrollment_active.md, T-250/enroll_sim.py | local |
| T-251 target-conditioned extraction | c89d707d | c32300da | target_conditioned.md, T-251/prep.py | local |
| T-252 latency/quality controller | b296084e | 15942320 | latency_controller.md, T-252 | local |
| T-253 ranked portfolio | 5c9291c4 | 9c3db863 | research_portfolio.md, T-253/portfolio.py | local |
| T-254 personalization UX + GPU preference job | e432fb05 | eabcce53 | personalization UX theory, prototype, GPU job prep | local only |
| T-255 blind A/B listening harness | 1cf5f374 | f8ac0192 | HTML/JS prototype, manifest contract | local only |
| T-256 opt-in preference layer | e46a01fa | a4072fa0 | prototype, theory, GPU job prep | local only |
| T-258 gain/pause render prep | a935d432 | b54c7b1e | offline deterministic renderer, atomic checkpoint | local only |
| T-259 room/noise render prep | 2a2e49be | d57672c9 | bounded room-response renderer | local only |

T-254..T-256 stack on one another (T-254 contains the T-255 and T-256 commits). Branches seen only on `origin` or the
cloud (`origin/task/T-258-muse-c-fix`) are not laptop-available and are excluded.

## Consumer instructions
1. Inspect a child: `git log --no-merges main..task/T-NNN`, then `git show <own commit> --stat`. Do not merge into main from here.
2. Highest-value next work (hypotheses, from T-253's ranking): open-set rejection (N1) and match-score calibration (T-249), both CPU on existing embeddings; GPU items (T-246 head, T-247 fine-tune, T-251 conditioned TSE) go to Sol's serial queue only after review.
3. Keep the source-group final holdout out of any training or tuning in all children.
4. Renderers (T-258, T-259) write private audio to ignored dirs; never commit it.

## Hypotheses vs observations
Observations: none were produced by T-257. Child prototypes run on synthetic data; their numbers show the prototype
logic works, not that any product claim holds. All benefit claims (lower false accept, better preference scores,
latency gain) are untested hypotheses until run on real fixtures.

## Hacks and assumptions
- Index relies on local branch state at write time; tips will move.
- "Own commit" is the newest non-merge commit, assumed to be the child's final work.
- Whether the children passed their own verify was not rechecked here.

## Filename note
The scoped path `docs/research/theory/RESEARCH_PORTFOLIO.md` differs only by case from T-253's `research_portfolio.md`;
on macOS (case-insensitive) writing it overwrites T-253's file and breaks `tests/test_theory_research_portfolio.py`.
This index therefore lives here instead. A rename of one file is needed before both can coexist.
