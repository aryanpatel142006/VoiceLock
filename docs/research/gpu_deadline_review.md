# GPU Deadline Review: Live Streaming Deadlines, Measured Bottlenecks, and Safe Next Steps

## 1. Executive Summary and Evaluation Mandate

This review audits the live-wrapper deadline ladder, streaming runtime feasibility, and separation baseline status following the delivery of **T-215** (`OVERNIGHT_RUNTIME.md`, `eval/overnight_runtime.py`, `tests/test_overnight_runtime.py`). It adheres to the Joseph + Fable approved execution order, caps, and ground rules for the voice backend of the HackRU assistive hearing app.

### Core Findings:
1. **Live Streaming Deadline Ladder Status (Item 2 of Caps)**:
   - **Real GPU execution was NOT completed**. T-215 successfully built and unit-tested the offline bounded runner (`eval/overnight_runtime.py`), supervisor, and validation framework (15 passed tests), but correctly marked all real GPU runtime rows as **missing / not measured**.
   - Per orchestrator guidance, the requested `t141` Replay12 dataset was withheld on `laptop-a` and absent on the remote cluster (`eval/fixtures/generated/t141` absent both locally and at `/common/home/jdr317/hearing-joseph-overnight`).
   - T-215 adhered strictly to ground rules by refusing to fabricate synthetic Replay12 clips or invent fake GPU timings. Consequently, live-wrapper deadlines (1.0s/0.5s and 2.0s/1.0s), missed hop rates, raw fallback weighted exposure, and target-switch delays remain **unmeasured on real GPU hardware**.
2. **Empirical Evidence from Completed GPU Runs (T-210 and T-289)**:
   - While T-215 executed 0 GPU cases, real GPU execution was completed by T-210 (`team-baseline.json`, `t210-a4500-20261010T0315`) and T-289 (`t289-a4500-20261010T0440`, `t289-bank.log`, `t213-scorer-manifest.json`) on Slurm job `447101` (ilab1, NVIDIA RTX A4500, 19,190 MiB VRAM).
   - **Measured Compute Bottleneck**: For 2.0s window / 1.0s hop, steady-state compute averages **202–330 ms** (p50: ~300 ms). However, cold start is severe: model load is **10.09 s**, first-window warmup is **7.03 s**, and in continuous 15-window replay, p95 compute reaches **2257–2382 ms** due to initial initialization.
   - **Tight Headroom for Streaming Ladder**: A 1.0s window / 0.5s (500 ms) hop allows only ~200 ms compute headroom against the ~300 ms baseline compute. Any initial warmup latency or compute jitter exceeding 500 ms will trigger immediate deadline misses, forcing the wrapper into raw-gain fallback.
3. **Target Separation Quality vs Interference (T-289 Bank)**:
   - On the Joseph/Aryan team pair, MossFormer2 achieves strong target recovery at 0 dB and -5 dB SIR for Joseph (16.66 dB and 14.25 dB SI-SDR; 0.0s wrong-speaker time; 0 flips).
   - For Aryan, performance degrades under increasing interferer volume:
     - SIR 0 dB: 13.31 dB SI-SDR, 1.0s (6.2%) wrong-speaker time, 6 flips.
     - SIR -5 dB: 12.28 dB SI-SDR, 1.75s (10.9%) wrong-speaker time, 8 flips.
     - SIR -10 dB: 10.18 dB SI-SDR, 2.25s (14.1%) wrong-speaker time, 10 flips.
   - This empirically confirms `PLAN.md`'s warning that wrong-speaker time in two-male overlap reaches 6–25%, especially under unfavorable SIR.
4. **SepFormer WHAMR16k Comparison (Item 1 of Caps)**:
   - **Omitted entirely**. Neither T-210, T-215, nor T-289 executed a comparison between MossFormer2 and SepFormer WHAMR16k.
5. **Architectural & Demo Recommendation**:
   - **DO NOT PROMOTE** FocusHQ live streaming tier to production defaults.
   - Maintain the HackRU core demo path: explicit pre-enrolled profile selection with stable fast-path tiers (DeepFilterNet / GTCRN) and reliable return to Raw/Open.

---

## 2. Quantitative Evidence and Audit

### 2.1 Hardware and Cluster State
- **Allocation**: Slurm Job `447101` on node `ilab1` (NVIDIA RTX A4500, 19,190 MiB VRAM; CUDA True, PyTorch `2.8.0+cu128`).
- **Load Overhead**:
  - Model loading (`MossFormer2_SS_16K` + `spkrec-ecapa-voxceleb`): **10.0899 s**.
  - Cold-start window warmup: **7.0250 s**.
  - Faster-Whisper `base.en`: Load **3.4733 s**, warmup **0.5318 s**.
- **Queue and Results**:
  - Prior completed runs: `overnight-first-separation.done`, `t289-bank.done`.
  - Stored outputs: `results/team-baseline.json`, `results/t210-a4500-20261010T0315`, `results/t289-a4500-20261010T0440`.
  - Queue is currently idle (no active jobs running in `queue/`).

### 2.2 Measured Separation and Wrong-Speaker Statistics (T-289 15-Window Evaluation)

From `/common/home/jdr317/hearing-joseph-overnight/results/t289-a4500-20261010T0440/t289-bank.log`:

| Case ID | Target Role | SIR (dB) | Delivered SI-SDR (dB) | Delivered SI-SDRi (dB) | Compute p50 (ms) | Compute p95 (ms) | Wrong Time (s) | Wrong % | Dominant Flips | Other Leak (dB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `team-joseph-sir0` | Joseph | 0 | 16.67 | +16.61 | 302.1 | 2258.0 | 0.00 | 0.0% | 0 | -32.5 |
| `team-joseph-sir-5` | Joseph | -5 | 14.23 | +19.11 | 297.8 | 2284.4 | 0.00 | 0.0% | 0 | -34.8 |
| `team-joseph-sir-10` | Joseph | -10 | 11.11 | +20.91 | 303.3 | 2281.9 | 0.25 | 1.8% | 2 | -35.2 |
| `team-aryan-sir0` | Aryan | 0 | 13.36 | +13.29 | 298.9 | 2259.5 | 1.00 | 6.2% | 6 | -34.2 |
| `team-aryan-sir-5` | Aryan | -5 | 12.30 | +17.19 | 314.8 | 2259.8 | 1.75 | 10.9% | 8 | -35.9 |
| `team-aryan-sir-10` | Aryan | -10 | 9.68 | +19.48 | 314.7 | 2382.1 | 2.25 | 14.1% | 10 | -34.8 |
| `public-1320-sir0` | LibriSpeech | 0 | 10.96 | +11.05 | 296.0 | 384.3 | 0.00 | 0.0% | 0 | -34.0 |
| `public-6930-sir0` | LibriSpeech | 0 | 17.78 | +17.85 | 289.2 | 376.3 | 0.25 | 1.1% | 2 | -35.6 |

*Key Insights*:
- **P95 Latency Spikes in Unwarmed Sessions**: In team cases, p95 compute is 2.26–2.38 s because window 0 includes initial tensor allocation/JIT kernel compilation. In contrast, steady-state public cases (`public-1320`, `public-6930`) run at p50 289–296 ms and p95 376–384 ms.
- **Asymmetric Rejection**: Joseph's enrollment cleanly isolates the target even at -10 dB SIR (only 0.25s / 1.8% wrong-speaker time). Aryan's enrollment shows severe leakage against Joseph's louder interference, reaching 14.1% wrong speaker time with 10 dominant flips at -10 dB SIR.

### 2.3 Acoustic Probe Margins (from `team-baseline.json`)
- **Crop Duration Margin**:
  - 0.5s crop: `0.171` margin
  - 1.0s crop: `0.514` margin
  - 2.0s crop: `0.635` margin
  - 4.0s crop: `0.721` margin
- **Contamination Sensitivity**:
  - 0% contamination: `0.635` margin
  - 10% contamination: `0.328` margin
  - 30% contamination: `-0.003` margin (fails discrimination)
  - 50% contamination: `-0.282` margin (inversion/wrong speaker)
- *Implication*: Shorter estimation windows (<1.0s) severely degrade embedding separation margin from 0.635 to 0.171, increasing the likelihood of speaker misassignment during streaming transitions.

---

## 3. Audit of Missing and Skipped Metrics against Caps

| Approved Item | Cap / Constraint | Measured Status | Audit Notes |
| :--- | :--- | :--- | :--- |
| **(1) MossFormer2 vs SepFormer WHAMR16k** | <= 90 GPU min, both roles, SIR 0/-10, ceiling vs actual | **SKIPPED / MISSING** | SepFormer WHAMR16k was never evaluated. Only MossFormer2 was run. |
| **(2) Real live-wrapper deadline ladder** | <= 90 GPU min, transcripts 1s/0.5s and 2s/1s | **SKIPPED / MISSING** | T-215 runner implemented offline; zero GPU cases run due to absent `t141` dataset. |
| **(3) Contrast + floor_ab both roles** | <= 45 GPU min | **NOT RUN** | Not queued or executed on GPU cluster. |
| **(4) Projection head pilot** | <= 45 GPU min, dev only | **NOT RUN** | Prerequisite items 1–3 were not complete before 08:00 Eastern cutoff. |
| **Target-switch delay** | Replay12 switch arm, 100ms proxy | **UNMEASURED** | Implemented in `eval/overnight_runtime.py`, but not run on GPU. |
| **Controlled concurrent load arm** | 1024x1024 matmul paired arm | **UNMEASURED** | Synthetic process implemented; not run on GPU. |
| **Remote WebSocket session smoke** | Ephemeral port, `/ws/session` probe | **UNMEASURED** | Optional HTTP health probe supported; WebSocket session smoke not performed. |
| **Hardware mic-to-ear latency** | Real audio I/O | **NOT RUN** | Prohibited in headless automated tasks by Ground Rules. |

---

## 4. Source Revision and Environmental Drift

1. **Code Revisions**:
   - `t210` executed on snapshot SHA `34e567386606885f7a453eaae898cb2483242993`.
   - `T-215` implemented on commit `53f9e88cc3387790c6a9cafa812565cb3ae7880e`.
   - Current repository HEAD: `4565143` (`T-214: Overnight mixer and AGC counterfactuals`).
2. **Cluster Allocation Discrepancy**:
   - The task prompt referenced Slurm allocation `447008` on `rlab2`.
   - However, inspection of the active cluster via ControlMaster reveals that job `447008` is expired/absent, while job `447101` on node `ilab1` is the active, running allocation (up 7+ hours). Future GPU invocations must target `447101` via `ILAB_REUSE=447101` rather than stale job IDs.
3. **Model & Weight Provenance**:
   - Local model files present and verified:
     - `models/MossFormer2_SS_16K/last_best_checkpoint.pt` (`00a3a48b...`, 670,353,271 bytes).
     - `models/spkrec-ecapa-voxceleb/embedding_model.ckpt` (`0575cb64...`, 83,316,686 bytes).
   - Remote execution logs revealed unauthenticated Hugging Face Hub download warnings for `faster-whisper-base.en`, which was fetched dynamically rather than resolved from local repository snapshots.

---

## 5. Next Measurable Decision and Safe Action Plan

### Decision: Enforce Production Freeze; Reject Live Streaming FocusHQ Adoption
- **Stance**: Live streaming FocusHQ (whether 2s/1s or 1s/0.5s) **MUST NOT** be promoted to demo defaults or live pipeline configurations.
- **Rationale**:
  1. No live streaming deadline ladder has been verified under real GPU load.
  2. Baseline compute latency (p50 ~300 ms, p95 up to 2.38 s without warmup) leaves zero safety margin for low-latency streaming without severe audio stutter or fallback.
  3. Wrong-speaker leakage on the Aryan target reaches 14.1% at -10 dB SIR with 10 dominant flips.
  4. SepFormer WHAMR16k bakeoff was not executed, so there is no empirical evidence that SepFormer would outperform MossFormer2.

### Safe Next Step (Bounded Falsifying Experiment for Morning Queue)
If the orchestrator or human schedules a follow-up GPU verification before the 10:00 AM Eastern hard stop:
- **Scope**: Exactly **ONE focused measurement** using existing verified team fixtures (`data/team_baseline`) instead of the missing `t141`.
- **Protocol**:
  - Run `eval/overnight_runtime.py` with `configs: [[2, 1], [1, 0.5]]` on 2 verified clips (Joseph target, Aryan target) with 2 repeats = **12 total cases**.
  - Warm up the model with 2 dummy windows before recording deadline metrics, eliminating the 2.2s PyTorch allocation artifact.
  - Measure true steady-state compute p50/p95, missed hop count, and raw fallback exposure seconds under both 1000 ms and 500 ms deadlines.
- **Bounds**: <= 20 GPU minutes, max 15 evaluation cases, reuse existing allocation `447101` via `launch.sh`.
- **Success Criteria**:
  - 1.0s/0.5s ladder achieves compute p95 < 350 ms and 0 missed hops across all steady-state windows.
  - If p95 exceeds 500 ms or missed hops > 0, the streaming deadline ladder is falsified and dropped.

---

## 6. Relays and Handoff Messages

- **To Parent Task (T-215)**: Review complete. T-215's decision to mark GPU rows as unmeasured rather than fabricating Replay12 inputs was correct and preserves data integrity. Offline harness and cap guards are verified.
- **To Fable / Board**: No new GPU jobs should be launched without explicit fixture alignment. The 08:00 AM Eastern cutoff for new jobs is respected. Live demo defaults remain frozen on pre-enrolled profiles with Raw/Open fallback.
