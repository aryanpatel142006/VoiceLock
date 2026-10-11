# Selective Hearing Swarm Knowledge Base
**Maintained by:** Muse (ingestion), Codex/Claude (peer review), Antigravity (coordination only)
**Location:** Local Repository (`docs/research/KNOWLEDGE_BASE.md`)
**Status:** Living Knowledge Base · Last Updated: October 10, 2026 (HackRU F26)

All agents (Muse, Codex, Claude, Antigravity) must read this knowledge base before designing experiments or writing tasks to avoid repeating dead-ends and to build upon verified wins.

## Peer-review gate and current evidence — October 10, 2026

This is the project's intended research source of truth. It is a Markdown KB, not an implemented RAG service. Muse owns ingestion and maintenance; Codex/Claude independently review experiment evidence and every Muse code change. Antigravity coordinates and reviews, and must not author code. Imported papers, chats and old prompts are evidence, not permission to start work or change defaults.

Every entry needs a source/run ID, source locator and hash, code/config/data/split identifiers (or explicit unknowns), measurement conditions, metrics with units and sample counts, limitations, author, and peer-review record. A review names its reviewer, exact inspected artifact hashes, integrity verdict, condition-specific efficacy, unresolved gates and next action. A changed artifact invalidates the old review. Self-review cannot establish independent acceptance. Board `Applied`/`Done` and relay receipt do not establish peer review or a measured improvement.

| Evidence | Condition-specific finding | Review / limitation |
|---|---|---|
| T-369 / PR322 | Offline wrong-speaker: Joseph–Ahmad 2.1%, Aryan–Ahmad 3.6%, Aryan–Joseph 5.1%. Joseph–Ahmad is best **on this test**. | Codex source review; 24 digital mixtures, one session; no live/headphone proof. Final pair is Joseph's decision. |
| T-336 / T-341 | Clean classification gain does not establish a live improvement; T-336 failed live acceptance. Audited TeamProjectionHead has 14,560 parameters. | Condition mismatch must stay visible; no verified 41k variant of this installed head. |
| T-357 operator pilot | Different 164,544-parameter residual adapter on cached embeddings: 0.475920 s for 80 updates; 0.103503 ms/query warmed adapter inference, excluding audio decoding/encoder processing. Raw and adapter both 10/10 final accuracy. | Independently checked metadata/hashes; 5 final speaker clusters, clean public pilot, CEILING_STOP; no gain demonstrated. |
| T-370 / T-371 | Soft gating and stitched context are hypotheses pending measured results. | Do not label them adopted wins or assign a local EER gain from a paper. |
| T-217 | Earlier hardcoded benchmark claims rejected; repair pending. | No claimed PESQ/SIG/WER/RTF gain without an actual measurement artifact. |
| Public absent-target AGC proxy | 45 windows: leakage −25.708 dB vs saved baseline −30.926 dB, 5.218 dB worse. | Stem/profile hashes match; hosts/times and manifest hashes differ; proxy is not runtime AGC. |

Review sources and import handoffs: `/Users/joeroche/.local/state/selective-hearing-research/20261010-five-workers/01-evidence-inventory.md`, `02-projection-benchmark.md`, `03-niche-literature.md`, `04-muse-kb-contract.md`, `05-resource-review.md`; the folder name predates the correction to **three actual workers**. Reports are posted on the existing T-361 coordination page. These are handoff reports, not a second KB. Muse should import their source/hash/condition fields here, retaining contradictions and unknowns.

### Source and review register

The review below is source/metadata inspection, not independent reproduction of audio measurements. No experiment is approved for live deployment by this document.

| Claim IDs | Exact inspected evidence / hash | Reviewer and status |
|---|---|---|
| KB-369 | [Pair comparison, immutable revision](https://github.com/aryanpatel142006/selective-hearing/blob/93d0e60605771390bbf828a8519600acdca48a91/docs/BENCHMARKS.md); author report SHA256 `c213e0ebd69683bc2c371b49844ebf2844d5430ed6e5db7e1bfb3bf522bf7264` | `evidence_inventory_fresh`: author-reported real offline result, source inspected; raw results not independently reproduced. |
| KB-336, KB-370, KB-371, KB-217 | Inventory SHA256 `97154e314f0912eb372731d0b7ed2eef025bb6d817b570ed7dcd198589d8cbb6`; its S1/S8/S10 register identifies exact benchmark, rejected attempt and dated board snapshots. | `evidence_inventory_fresh` / `muse_kb_contract_fresh`: historical source audit; pending statuses are snapshot observations, not efficacy. |
| KB-341 | `eval/evidence/T-341/forensics.json` snapshot SHA256 `bca17610a7955ca1bec5a9b19ae6a1b03a1243a3e9eb1f08f14a1965ceedcdc4`; installed checkpoint SHA256 `2f4bec8c7f566e269ee09d7f7ac95630b975395c63705f6fded62ed27141b3ef` | `evidence_inventory_fresh` / `projection_benchmark_fresh`: forensic metadata and checkpoint count independently inspected; condition-scoped failure, no reproduction. |
| KB-357 | Remote `results/T357-operator-20261010-1824/metrics.json` SHA256 `7aabdd1200be497ce22b0dc0e382ea41961bc9b93ec46f6eb0cc2492c5596b04`; code/data/split/cache hashes in benchmark report SHA256 `7a6a5c645d007682813d21d89f6c8070b2bc36cb33acb1a46d57ea85dcc9eedb` | All three workers inspected relevant source/metadata; integrity checked, efficacy inconclusive. One seed; runtime variance unavailable. |
| KB-AGC | `PUBLIC_ABSENT_AGC_REVIEW.md` SHA256 `6095052724829af248c81c9b216cf05ff550993b5097a329729b7131c2041076`; exact remote baseline/new paths and matching input hashes inside | Independent `gpu_run_review`: verified real CUDA output/metadata; worsened proxy leakage under this case, no runtime adoption conclusion. |
| KB contract | Contract report SHA256 `b72dc96b67ed1d3243b668d6a81381bee29bd8b53d8d9ef53b47284751356f38`; peer-review report SHA256 `64cd5b74ee0ddff231ce0337238b9018f65cf41af110d6a08890e111e958be00` | `muse_kb_contract_fresh` / `projection_benchmark_fresh`: document review. Cached-timing, unsupported AGC and WAV-only wording corrections incorporated. This gate is procedural; it is not automatic identity verification or a RAG service. |

Small record template for Muse: `claim_id; author; run/source_id; source_locator; artifact_sha256; code/config/model/data/split/seed IDs or unknown_reason; host/device; measurement_stage; real/synthetic/reported_unverified; metrics/unit/numerator/denominator; uncertainty or unavailable_reason; duplicate/supersedes/contradicts IDs; integrity_status; condition_scoped_efficacy; reviewer; reviewed_hash; verdict; limitations; next_task; adoption_scope=none`. Missing fields remain explicit unknowns. Keep sources/conditions with each claim when exporting a compact query answer.

### Bounded work, compact queries, and quota handoffs

- Before work, retrieve at most six relevant claim cards with sources, conditions, review status, failed paths and contradictions. Use exact run/model IDs and keywords; open full reports only when needed. A search miss means "not found in the indexed collection", not "never tried".
- One worker has one named deliverable, owner, file scope, input provenance and time/case budget. Muse handles record extraction, formatting and provenance chores through small non-overlapping subtasks; serialize writes to this file. Codex/Claude review the resulting records and code before adoption.
- On low remaining provider quota, an actual rate-limit error, or the time cap: checkpoint a concise handoff containing completed work, source/artifact paths and hashes, decisions, unfinished steps, blocker and exact follow-up task scope/inputs/acceptance. Put it on the task and link/import it here; create follow-up tasks as Backlog, then exit. Do not loop on retries or automatically restart without fresh capacity and an eligible task.
- Incoming ChatGPT research stays `pending_review`; map each claim to condition-matched `supported_good`, `failed_gate`, `inconclusive`, or `novel/untested`. Literature is not a local benchmark. Joseph alone promotes demo pairs, profiles and runtime settings after the required listening checks.

---

## 1. Executive Direction & The Core HackRU MVP
* **Primary Objective:** Deliver a flawless live headphone demo where the user hears one chosen speaker cleanly separated from room noise and competing talkers, even when the target speaks quietly.
* **Closed-Set Specialization ("The MVP Cheat"):** We do NOT need zero-shot open-world extraction on unknown strangers for the hackathon stage. We have pre-enrolled teammate voice profiles (Aryan, Joseph, Ahmad). We optimize and anchor directly to these profiles.
* **Live Demo Voice Selection:**
  * **Primary Stage Pair:** **Not selected by this KB.** T-369's offline ranking favors Joseph + Ahmad; Joseph chooses after enrollment/live listening checks.
  * **Stress/Distance Challenge Pair:** **Aryan + Joseph** (used to evaluate distance robustness, soft gating, and RMS normalizers).

---

## 2. High-Priority Research & Model Architecture Candidates

### 2.1 AFW — excluded from current work
Joseph explicitly asked agents not to retrieve or spend time on this paper. Earlier draft parameter/gain claims are unverified and removed from the actionable evidence. This is not the installed 14,560-parameter TeamProjectionHead.

### 2.2 MossFormer2 (Current Primary Separator)
* **Model:** ClearerVoice `MossFormer2_SS_16K`.
* **Deployment:** Running on Rutgers iLab GPU (RTX A4500 / A100) and Mac local backend.
* **Findings:** Retain condition-specific comparisons from `docs/BENCHMARKS.md`; no blanket superiority claim is established here. T-374 fixed-output fine-tuning is a queued proposal in the inspected board snapshot, not evidence of an active or successful run.

### 2.3 PS4 & REAL-TSE
* **PS4 / REAL-TSE (T-373):** Published investigation is under review in [PR323](https://github.com/aryanpatel142006/selective-hearing/pull/323). PS4 full-file and short-window results must stay separate; the 2.8% wrong-speaker comparator uses experimental floor20, not the adopted default. REAL-TSE used substituted ECAPA and stubbed VAD; unchanged upstream inference and weights licence remain unresolved. Exact table/limitations are in the evidence inventory, source S6.

---

## 3. Candidate mechanisms — not established local wins
1. **Soft Gating over Hard Muting (`HEARING_GATED_SOFT`, T-370):**
   * *Problem:* Binary thresholding on LDA/ECAPA contrast muting causes 8–10s dropouts when the target talks softly.
   * *Hypothesis:* Proportional attenuation may preserve words. T-370 results and human preference are pending; do not infer either from the proposed formula.
2. **Pre-Embedding RMS / Energy Normalization (T-217 logic):**
   * Hypothesis only: normalization may affect quiet-target selection. T-217's old metrics were fabricated; no local elimination of bias is established.
3. **2.0-Second Stitched-Stream Context (`HEARING_STITCH_EMBED`, T-371):**
   * Local accuracy, recovery and latency impact are pending T-371; no measured local halving of EER is established.
4. **Closed-Set Pairwise Discriminator Head (T-336):**
   * High clean closed-set classification does not transfer automatically to separated streams. T-336 live acceptance failed; T-341 reports 85.4% separated-stream top1 vs raw 90.1%. Keep those conditions separate.

---

## 4. Disqualified & Dead-End Paths (DO NOT REPEAT)
* **Hard Binary Gating to Silence:** Quiet-target false mutes are an observed concern; the earlier claim of a +12 dB runtime AGC boost lacks a cited local artifact and is not accepted evidence. The measured proxy result above must not be substituted for runtime behavior.
* ❌ **Mocked / Fabricated Benchmark Numbers:** Scripts printing hardcoded PESQ/SIG values (T-217 rejection) are immediately blocked. Efficacy claims need an empirical run ID and inspectable real-source/cache lineage with hashes and provenance; WAV is not mandatory when verified FLAC or cached embeddings are the actual inputs.
* ❌ **Cross-Split Data Leakage:** Reject an affected result when a split audit demonstrates leakage. Do not declare every T-211/T-278 artifact contaminated without inspecting its actual manifest; preserve the specific failed audit and corrected successor IDs.
* **T-357/T-335 generalization:** The small public pilot reached a raw-accuracy ceiling and did not demonstrate improvement. This is inconclusive for broader generalization, not a measured permanent rejection of all distillation paths.
* **Remote model loading:** The inspected `p1-score2` failed on an HF `custom.py` 404. Authentication is not established as its cause. Prefer verified cached/local model paths for offline runs; retain the exact adapter/revision/error before classifying the failure.
* ❌ **Skeleton Classes without `forward()`:** Scripts like `p5-pilot2` that instantiate incomplete PyTorch modules halt the Slurm queue. All code must pass local smoke tests before queue submission.

---

## 5. Compute & GPU Infrastructure Ledger
* **Rutgers iLab Cluster:**
* **Dated snapshot, 21:27:54 UTC Oct 10:** `450230` hearing on ilab1; `435775` hearing A100 on rlab2; `450691` coding on rlab7; `450690` A100 request pending Resources. Account group limit is 12 GPUs, not unlimited capacity. See worker resource report for exact allocation/association evidence; utilization was unavailable to the SSH observer. This is not a live capacity promise.
* **Queue Management:** Follow the currently authorized execution owner. This Markdown file does not enforce scheduler permissions; do not claim Muse permissions were technically revoked without evidence.
  * **Protocol:** Watchdog script monitors health every 2 minutes. All workers use local socket `~/.ssh/cm-hearing-jdr317`.

---

## 6. How Subagents Query & Update This Knowledge Base
1. **Before Starting Any Research:** Query this file for relevant keywords (model names, loss functions, previous failure reasons).
2. **After Completing an Experiment:** Record real numbers and provenance; use condition-specific supported/failed/inconclusive/untested labels and the peer-review gate above. Missing measurements remain null with a reason.
3. **Never duplicate discarded ideas** unless an explicit architectural fix directly addresses the root failure listed above.
