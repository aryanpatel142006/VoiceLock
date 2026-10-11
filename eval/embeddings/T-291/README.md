# T-291 Team Embedding Analysis Dataset (`team_slices.npz`)

Analysis dataset packaged for muse-c's T-249 speaker-calibration and rejection theory work. Contains frozen ECAPA-TDNN embeddings (192-d), discriminative projection head outputs (32-d), and per-slice metadata across window lengths (0.5s, 1.0s, 2.0s) extracted over real team recordings of Aryan and Joseph.

---

## 1. Provenance & Audio Sources

- **Team Audio:**
  - **Aryan Patel:** `eval/fixtures/generated/team/aryan/fixture-01.wav` (58.69s), `eval/fixtures/generated/team/aryan/profile-01.wav` (50.00s) — Total 108.69s.
  - **Joseph:** `eval/fixtures/generated/team/joseph/fixture-01-20261009-1353.wav` (48.15s), `eval/fixtures/generated/team/joseph/fixture-02-namecall-20261009-1353.wav` (27.80s), `eval/fixtures/generated/team/joseph/profile-01-20261009-1353.wav` (15.70s) — Total 91.65s.
- **Audio Framing:** 16 kHz mono, non-overlapping windows (stride = window length).
- **Voice Activity Filter:** Voiced chunks meeting standard acoustic gate (`peak >= 0.02` and `RMS >= 0.005`).
- **Embedding Backbone:** Frozen `speechbrain/spkrec-ecapa-voxceleb` (192-d unit hypersphere embeddings $S^{191}$).
- **Projection Head:** `models/team_projection_head.pt` (Linear 192 $\to$ 64 $\to$ LayerNorm $\to$ GELU $\to$ Linear 64 $\to$ 32 $\to$ L2-norm on $S^{31}$).

---

## 2. Dataset Field Specification

Stored as a single compressed NumPy archive at `eval/embeddings/T-291/team_slices.npz`.

### Per-Window Arrays (for `<w>` $\in \{$`0.5s`, `1.0s`, `2.0s`$\}$)

| Key in `.npz` | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `embeddings/<w>` | `(N, 192)` | `float32` | Frozen ECAPA-TDNN 192-dimensional embeddings, normalized to $\|e\|_2 = 1.0$. |
| `head_out/<w>` | `(N, 32)` | `float32` | Output vectors from `team_projection_head.pt`, normalized to $\|z\|_2 = 1.0$. |
| `meta/<w>/speaker` | `(N,)` | `int64` | Numeric speaker label: `0` for Aryan, `1` for Joseph. |
| `meta/<w>/speaker_id` | `(N,)` | `object` (`str`) | String speaker identifier (`"aryan"` or `"joseph"`). |
| `meta/<w>/speaker_name` | `(N,)` | `object` (`str`) | Canonical speaker name (`"aryan"` or `"joseph"`). |
| `meta/<w>/window_length` | `(N,)` | `float32` | Window duration in seconds (`0.5`, `1.0`, or `2.0`). |
| `meta/<w>/session` | `(N,)` | `int64` | Recording index within that speaker's audio files (take ID). |
| `meta/<w>/recording` | `(N,)` | `object` (`str`) | Source filename (e.g. `"profile-01.wav"`, `"fixture-01.wav"`). |
| `meta/<w>/start_s` | `(N,)` | `float32` | Start timestamp in seconds within the source recording. |
| `meta/<w>/snr_db` | `(N,)` | `float32` | Estimated signal-to-noise ratio in dB (overall window RMS vs 10th-percentile 20ms frame energy). |
| `meta/<w>/cosine_margin_stock` | `(N,)` | `float32` | Per-window margin in stock ECAPA space: mean cosine similarity to all other windows of same speaker minus mean cosine similarity to all windows of other speaker. |
| `meta/<w>/projected_margin_head`| `(N,)` | `float32` | Per-window margin in 32-d projected head space: mean cosine similarity to all other projected windows of same speaker minus mean cosine similarity to other speaker. |

### Global Metadata Fields

| Key in `.npz` | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `windows` | `(3,)` | `<U4` | Window names: `['0.5s', '1.0s', '2.0s']`. |
| `speakers` | `(2,)` | `<U6` | Registered speakers: `['aryan', 'joseph']`. |
| `margins_json` | `()` | `str` | JSON string with complete summary statistics (mean, std, percentiles) per window length. |
| `provenance_json` | `()` | `str` | JSON string with recording names, model checkpoints, device, and extraction settings. |

---

## 3. Slice Counts per Window Length

| Window | Aryan Slices | Joseph Slices | Total Slices ($N$) | Total Audio Duration |
| :--- | :---: | :---: | :---: | :---: |
| **0.5s** | 169 | 157 | **326** | 163.0 s |
| **1.0s** | 97 | 84 | **181** | 181.0 s |
| **2.0s** | 50 | 43 | **93** | 186.0 s |

---

## 4. Margin Distribution & The 0.45 $\to$ 0.15 Collapse Claim

The core empirical motivation for T-249 and T-273 is the **severe margin collapse** exhibited by unprojected acoustic speaker embeddings as window length decreases from 2.0s down to 0.5s.

### Measured Margin Distributions on Real Team Audio

```
==============================================================================
  T-291 SPEAKER EMBEDDING ANALYSIS: MARGIN DISTRIBUTIONS & COLLAPSE CLAIM
==============================================================================

[Window Length: 0.5s] (Total Slices: 326 | Aryan: 169, Joseph: 157)
  Metric                Mean    Std    Min    P05    P25    P50    P75    P95    Max
  Stock ECAPA (192-d):  0.128  0.060 -0.028  0.017  0.090  0.135  0.173  0.217  0.264
  Proj Head (32-d):     1.003  0.393 -1.088  0.138  0.926  1.147  1.242  1.308  1.340
  Pairwise Cosines:    Stock [same=0.177, cross=0.049, margin=0.129]
                       Head  [same=0.616, cross=-0.390, margin=1.005]
  Head/Stock Expansion: 7.85x gain in separability

[Window Length: 1.0s] (Total Slices: 181 | Aryan: 97, Joseph: 84)
  Metric                Mean    Std    Min    P05    P25    P50    P75    P95    Max
  Stock ECAPA (192-d):  0.235  0.089 -0.030  0.060  0.185  0.252  0.301  0.354  0.381
  Proj Head (32-d):     1.322  0.324 -0.259  0.584  1.306  1.436  1.492  1.530  1.558
  Pairwise Cosines:    Stock [same=0.289, cross=0.052, margin=0.237]
                       Head  [same=0.763, cross=-0.562, margin=1.324]
  Head/Stock Expansion: 5.61x gain in separability

[Window Length: 2.0s] (Total Slices: 93 | Aryan: 50, Joseph: 43)
  Metric                Mean    Std    Min    P05    P25    P50    P75    P95    Max
  Stock ECAPA (192-d):  0.390  0.103  0.024  0.167  0.355  0.406  0.459  0.510  0.576
  Proj Head (32-d):     1.620  0.161  0.403  1.451  1.617  1.655  1.691  1.707  1.726
  Pairwise Cosines:    Stock [same=0.464, cross=0.071, margin=0.392]
                       Head  [same=0.898, cross=-0.725, margin=1.622]
  Head/Stock Expansion: 4.16x gain in separability
```

### Key Findings:
1. **The 0.45 $\to$ 0.15 Collapse Confirmed:**
   - At 2.0s, stock ECAPA provides clean separation: same-speaker cosine mean = 0.464, cross-speaker cosine = 0.071, giving a margin of **0.390** (median 0.406, p95 0.510).
   - At 0.5s, intra-speaker cosine collapses to 0.177, causing the stock margin to drop to **0.128** (median 0.135).
   - This represents a **67.2% margin collapse**, explaining why 0.5s live streaming target selection triggers false re-targets and stream flips without projection.
2. **Projection Head Recovery:**
   - `team_projection_head.pt` widens the 0.5s margin from **0.128 up to 1.003** (median 1.147, p95 1.308) — a **7.85x expansion**.
   - Intra-speaker cosine improves from 0.177 to +0.616, and cross-speaker cosine is driven antipodal to -0.390.

---

## 5. Usage in Downstream Theory Work (T-249 / Muse-C)

```python
from eval.embeddings.T_291.loader import load_slices, load_margins, train_test_split_by_session

# Load 0.5s slice embeddings and metadata
data_05 = load_slices("0.5s")
embs = data_05["embeddings"]              # (326, 192) float32
head_z = data_05["head_out"]              # (326, 32) float32
speakers = data_05["speaker"]             # (326,) int64: 0 or 1
margins = data_05["cosine_margin_stock"]  # (326,) float32
snr_db = data_05["snr_db"]                # (326,) float32

# Load margin summaries across all 3 window lengths
margins = load_margins()
print("0.5s Stock Margin:", margins["0.5s"]["stock_cosine"]["mean"])
print("2.0s Stock Margin:", margins["2.0s"]["stock_cosine"]["mean"])
```

To run the stats script directly from the shell:
```bash
.venv/bin/python eval/embeddings/T-291/stats.py
```

To re-extract or regenerate the dataset:
```bash
.venv/bin/python eval/embeddings/T-291/extract_slices.py
```
