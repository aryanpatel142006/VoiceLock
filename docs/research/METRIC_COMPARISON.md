# Metric Comparison Eval Harness: Cosine vs Mahalanobis vs Projection Head

- **Task ID:** T-292
- **Milestone:** Post-M10 Creative R&D / Grunt for muse-c
- **Date:** October 10, 2026
- **Status:** Complete (Verified)
- **Primary Artifacts:**
  - `eval/compare_metrics.py` (CLI evaluation harness)
  - `tests/test_compare_metrics.py` (Unit tests for harness and metrics)
  - `docs/research/METRIC_COMPARISON.md` (This document)
  - `results/metrics/T-292/` (Output directory with Markdown table, JSON schema, and PNG plots)
- **Depends on:**
  - T-291: Packaged frozen ECAPA-TDNN (192-d) and projection head (32-d) embedding slices (`eval/embeddings/T-291/team_slices.npz`)
  - T-274: Calibrated pairwise Mahalanobis distance metric specification (`eval/mahalanobis_metric.py`, `docs/decisions/T-274.md`)

---

## 1. Executive Summary & Purpose

Under direction from **muse-c** (investigating speaker-verification calibration, rejection theory, and the 0.45 $\to$ 0.15 short-window margin collapse), task **T-292** provides a unified head-to-head evaluation harness (`eval/compare_metrics.py`) comparing three speaker-verification backends across three window lengths (0.5s, 1.0s, and 2.0s):

1. **Stock ECAPA-TDNN Cosine Similarity (192-d):** Standard baseline cosine on frozen hyperspherical embeddings $S^{191}$.
2. **Calibrated Pairwise Mahalanobis Metric (192-d):** Closed-form Ledoit-Wolf regularized whitening ($L = \Sigma_{\text{reg}}^{-1/2}$) derived in T-274 to downweight shared male acoustic and vocal tract dimensions between teammates Aryan and Joseph.
3. **Discriminative Projection Head Cosine (32-d):** 192 $\to$ 64 $\to$ LayerNorm $\to$ GELU $\to$ 32 projection head trained with contrastive margin loss (`models/team_projection_head.pt`).

All metrics are benchmarked on the ground-truth T-291 team embedding dataset (`team_slices.npz`), which contains 600 total slices across 0.5s, 1.0s, and 2.0s audio chunks from real Aryan and Joseph recordings.

---

## 2. Key Findings & Empirical Results

### Summary Comparison Table

The table below reports measured performance on all $N(N-1)/2$ pairwise verification trials per window length (52,975 pairs at 0.5s; 16,290 pairs at 1.0s; 4,278 pairs at 2.0s):

| Window | Backend | Measured | Dim | EER (%) | minDCF ($p=0.01$) | Mean Margin | Median Margin | Same Cos Mean | Cross Cos Mean |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `0.5s` | **`cosine`** | `true` | 192 | **26.20%** | **0.971** | +0.128 | +0.135 | +0.177 | +0.049 |
| `0.5s` | **`mahalanobis`** | `true` | 192 | **9.50%** | **0.793** | +0.157 | +0.161 | +0.076 | -0.082 |
| `0.5s` | **`head`** | `true` | 32 | **8.05%** | **0.657** | **+1.003** | **+1.147** | +0.616 | -0.390 |
| `1.0s` | **`cosine`** | `true` | 192 | **15.10%** | **0.723** | +0.235 | +0.252 | +0.289 | +0.052 |
| `1.0s` | **`mahalanobis`** | `true` | 192 | **0.05%** | **0.001** | +0.350 | +0.356 | +0.170 | -0.180 |
| `1.0s` | **`head`** | `true` | 32 | **3.48%** | **0.134** | **+1.322** | **+1.436** | +0.763 | -0.562 |
| `2.0s` | **`cosine`** | `true` | 192 | **4.75%** | **0.151** | +0.390 | +0.406 | +0.464 | +0.071 |
| `2.0s` | **`mahalanobis`** | `true` | 192 | **0.00%** | **0.000** | +1.015 | +1.017 | +0.501 | -0.513 |
| `2.0s` | **`head`** | `true` | 32 | **0.00%** | **0.000** | **+1.620** | **+1.655** | +0.898 | -0.725 |

### Major Insights

1. **Confirmation of Short-Window Collapse:**
   - Under stock ECAPA cosine, shortening the analysis window from 2.0s to 0.5s causes the Equal Error Rate to explode from **4.75% up to 26.20%** ($\approx 5.5\times$ degradation) and minDCF to climb from **0.151 to 0.971** (near-random default guessing).
   - This empirically confirms why streaming TSE stream assignment flips and retargeting fails during brief 0.5s pauses or turns under stock ECAPA.

2. **Mahalanobis Whitening Yields Massive Gain Without Extra Weights:**
   - At 0.5s, the closed-form regularized Mahalanobis metric drops EER from **26.20% down to 9.50%** ($\approx 64\%$ relative error reduction).
   - At 1.0s, Mahalanobis drops EER from **15.10% down to 0.05%** and minDCF to **0.001**, completely separating the two speakers.
   - At 2.0s, Mahalanobis achieves **0.00% EER** and **0.000 minDCF**.
   - Crucially, this requires **zero neural network parameters or backpropagation**: it is a pure $192 \times 192$ matrix multiplication ($<0.01$ ms per inference).

3. **Projection Head Provides Superlative Margin Expansion:**
   - The learned 32-d projection head achieves the lowest 0.5s EER (**8.05%**) and minDCF (**0.657**).
   - The decision margin is expanded by **$7.8\times$** at 0.5s (from $+0.128$ to $+1.003$) and drives cross-speaker cosine antipodal to **$-0.390$**, providing wide noise and leakage margins for threshold gating.

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Equal Error Rate (EER)

Let positive (target/same-speaker) scores be $\mathcal{S}_{\text{pos}}$ and negative (impostor/different-speaker) scores be $\mathcal{S}_{\text{neg}}$.
For any threshold $\tau \in \mathbb{R}$:
$$\text{FRR}(\tau) = \frac{1}{|\mathcal{S}_{\text{pos}}|} \sum_{s \in \mathcal{S}_{\text{pos}}} \mathbf{1}(s < \tau)$$
$$\text{FAR}(\tau) = \frac{1}{|\mathcal{S}_{\text{neg}}|} \sum_{s \in \mathcal{S}_{\text{neg}}} \mathbf{1}(s \ge \tau)$$

The harness evaluates all unique trial scores plus boundary operating points ($\min - \delta$ and $\max + \delta$). At the sign change of $\text{FAR}(\tau) - \text{FRR}(\tau)$, exact linear interpolation computes:
$$\alpha = \frac{\Delta_k}{\Delta_k - \Delta_{k+1}}, \quad \Delta = \text{FAR} - \text{FRR}$$
$$\text{EER} = \text{FRR}_k + \alpha (\text{FRR}_{k+1} - \text{FRR}_k)$$

### 3.2 Minimum Detection Cost Function (minDCF)

Per NIST SRE convention, detection cost at operating point $\tau$ is:
$$C_{\text{det}}(\tau) = C_{\text{miss}} \cdot P_{\text{target}} \cdot \text{FRR}(\tau) + C_{\text{fa}} \cdot (1 - P_{\text{target}}) \cdot \text{FAR}(\tau)$$
$$C_{\text{default}} = \min(C_{\text{miss}} \cdot P_{\text{target}}, \, C_{\text{fa}} \cdot (1 - P_{\text{target}}))$$
$$C_{\text{norm}}(\tau) = \frac{C_{\text{det}}(\tau)}{C_{\text{default}}}$$
$$\text{minDCF} = \min_\tau C_{\text{norm}}(\tau)$$

Standard parameters: $P_{\text{target}} = 0.01$, $C_{\text{miss}} = 1.0$, $C_{\text{fa}} = 1.0$.

### 3.3 Calibrated Pairwise Mahalanobis Metric

Given speaker slice embeddings $e_i \in \mathbb{R}^{192}$:
1. Intra-speaker centroids: $\mu_0 = \frac{1}{N_0} \sum e_i^{(0)}$, $\mu_1 = \frac{1}{N_1} \sum e_i^{(1)}$.
2. Pooled within-speaker covariance:
   $$\Sigma_{\text{pooled}} = \frac{1}{2} (\Sigma_0 + \Sigma_1)$$
3. Ledoit-Wolf shrinkage regularization ($\lambda = 0.30$ per T-274 spec):
   $$\Sigma_{\text{reg}} = (1 - \lambda)\Sigma_{\text{pooled}} + \lambda \frac{\text{tr}(\Sigma_{\text{pooled}})}{D} I_{D}$$
4. Whitening transform matrix $L = \Sigma_{\text{reg}}^{-1/2} = V \text{diag}(w^{-1/2}) V^T$.
5. Whitened, centered unit embeddings:
   $$z_i = \frac{L (e_i - \mu_{\text{pooled}})}{\|L (e_i - \mu_{\text{pooled}})\|_2}$$
6. Scoring: $\cos_M(u, v) = z_u \cdot z_v$.

---

## 4. Visualizations & Plots

The harness automatically renders and saves high-resolution figures into `results/metrics/T-292/`:

1. **`margin_histograms.png`**: Multi-panel probability density distributions of decision margins for cosine vs Mahalanobis vs projection head across 0.5s, 1.0s, and 2.0s windows.
2. **`det_curves.png`**: Detection Error Tradeoff curves (FAR % vs FRR %) showing separation boundaries and the EER diagonal.
3. **`metric_comparison_bars.png`**: Bar charts comparing EER (%), minDCF, and Mean Margin across all combinations.

---

## 5. Machine-Readable Schema & Provenance

The JSON summary at `results/metrics/T-292/summary.json` adheres strictly to the required machine-readable specification:

```json
{
  "task": "T-292",
  "timestamp": "2026-10-10T15:50:27.976161+00:00",
  "dataset_path": "eval/embeddings/T-291/team_slices.npz",
  "trial_mode": "all_pairs",
  "records": [
    {
      "backend": "cosine",
      "measured": true,
      "window": "0.5s",
      "window_s": 0.5,
      "dim": 192,
      "n_slices": 326,
      "n_pos_trials": 26442,
      "n_neg_trials": 26533,
      "eer": 0.2620,
      "eer_percent": 26.20,
      "eer_threshold": 0.1035,
      "min_dcf": 0.9711,
      "min_dcf_threshold": 0.3753,
      "pairwise_margin": 0.1286,
      "pos_score_mean": 0.1772,
      "neg_score_mean": 0.0486,
      "margin_mean": 0.1277,
      "margin_std": 0.0601,
      "margin_p05": 0.0169,
      "margin_p25": 0.0897,
      "margin_p50": 0.1354,
      "margin_p75": 0.1725,
      "margin_p95": 0.2166,
      "dcf_p_target": 0.01,
      "dcf_c_miss": 1.0,
      "dcf_c_fa": 1.0,
      "trial_mode": "all_pairs"
    },
    {
      "backend": "mahalanobis",
      "measured": true,
      "window": "0.5s",
      ...
    },
    {
      "backend": "head",
      "measured": true,
      "window": "0.5s",
      ...
    }
  ],
  "provenance": {
    "cosine": {
      "backend": "cosine",
      "measured": true,
      "embedding_dim": 192,
      "backbone": "speechbrain/spkrec-ecapa-voxceleb"
    },
    "mahalanobis": {
      "backend": "mahalanobis",
      "measured": true,
      "embedding_dim": 192,
      "shrinkage": 0.30,
      "centered": true,
      "backbone": "speechbrain/spkrec-ecapa-voxceleb + T-274 regularized pooled whitening"
    },
    "head": {
      "backend": "head",
      "measured": true,
      "embedding_dim": 32,
      "model_path": "models/team_projection_head.pt"
    }
  }
}
```

---

## 6. How to Run the Harness

```bash
# Full evaluation on default dataset with table and plots output to results/metrics/T-292/
.venv/bin/python eval/compare_metrics.py

# Custom dataset and output location
.venv/bin/python eval/compare_metrics.py \
    --dataset eval/embeddings/T-291/team_slices.npz \
    --output-dir results/metrics/custom \
    --windows 0.5s 1.0s \
    --backends cosine mahalanobis head

# Cross-session evaluation (exclude intra-session pairs)
.venv/bin/python eval/compare_metrics.py --trial-mode cross_session

# Fast headless run without plot generation
.venv/bin/python eval/compare_metrics.py --no-plots
```
