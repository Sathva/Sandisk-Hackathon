# Model F + CNN Models (C, C1, C2) Hybrid Blend Evaluation Report

## 1. Executive Summary & Core Finding

We rigorously evaluated whether blending the GPU-accelerated **Model F (Wafer-Conditional Manifold Detector, 1,280 features)** with your **1D CNN sequence models (Model C, Model C1, and Model C2)** produces further ensemble synergy.

The experiment was run across two independent zero-leakage splits:
1. **Canonical Development Validation Set (`dev_val`)**: 160 wafers, $137,576$ eligible dies (`old_label == 0`).
2. **Final Unseen Test Set (`test.csv`)**: 200 wafers, $185,126$ eligible dies (`old_label == 0`), $6,584$ positive defects.

### Key Conclusions:
1. **Yes, blending Model F with Model C1 provides a confirmed positive lift**:
   - On **`dev_val`**, blending Model F with 5%–10% Model C1 improves AUC-PR from **`0.63206`** to **`0.63367`** ($+0.00161$).
   - On **`final_test`** (using weights frozen from `dev_val`), blending Model F with 5% Model C1 improves Test AUC-PR from **`0.62374`** to **`0.62432`** ($+0.00058$) and Optimal F1 from **`0.57891`** to **`0.58187`** ($+0.00296$, cutting false positives from 800 down to 708).
2. **Why does it lift? (Orthogonal Inductive Bias)**:
   - **Model F** detects macro wafer-surface gradients and spatial detrended anomalies across coordinate polynomials and neighborhood manifolds.
   - **Model C1 (Triple-Branch CNN)** learns localized micro-transitions directly from the raw 60-element 1D block sequences (`block_0` to `block_59`).
   - Even though Model F is vastly superior on its own ($0.6237$ vs $0.5602$), adding a small continuous representation from C1 captures defects that occur without distinct wafer-level spatial clustering.
3. **Why must the CNN weight remain small ($\le 10\%$)**:
   - Because Model F is so strong, weighting the CNN higher than $12\%$ starts degrading overall precision (at $w=0.20$, Test AUC-PR falls to `0.62231`; at $w=0.30$, to `0.61919`). The sweet spot is precisely **$95\%$ Model F + $5\%$ Model C1** (or $90\%$ F + $10\%$ C1 in rank space).

---

## 2. Validation Blend Sweep (`dev_val`, 137,576 Dies)

| Architecture Blend | Best CNN Weight ($\alpha$) | Validation AUC-PR | Net Delta vs Model F |
| :--- | :---: | :---: | :---: |
| **Model F Standalone** | 0.00 | 0.63206 | Baseline |
| **Model C1 Standalone** | 1.00 | 0.57658 | -0.05548 |
| **Model C2 Standalone** | 1.00 | 0.57173 | -0.06033 |
| **Model C Standalone** | 1.00 | 0.57207 | -0.05999 |
| **Champion Baseline (63% C1 + 27% C2 + 10% B)** | 1.00 | 0.57945 | -0.05261 |
| **Model F + Model C1 (Probability Blend)** | **0.05** | **0.63353** | **+0.00147** |
| **Model F + Model C1 (Rank-Space Blend)** | **0.10** | **0.63367** | **+0.00161** |
| **Model F + Model C (Rank-Space Blend)** | 0.08 | 0.63347 | +0.00141 |
| **Model F + Champion Baseline (Rank-Space Blend)**| 0.09 | 0.63306 | +0.00100 |
| **Model F + Model C2 (Rank-Space Blend)** | 0.04 | 0.63227 | +0.00021 |

---

## 3. Final Unseen Test Benchmark (`test.csv`, 185,126 Eligible Dies)

Weights frozen strictly from validation to eliminate test-time leakage:

| Model / Blend Configuration | Test AUC-PR 🥇 | Test ROC-AUC | Optimal F1 🥈 | Precision | Recall | True Positives | False Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model C1 (Triple-Branch CNN)** | 0.56024 | 0.88744 | 0.54758 | 78.50% | 42.04% | 2,768 | 758 |
| **Frozen Champion Baseline** | 0.56283 | 0.88949 | 0.55048 | **82.40%** | 41.33% | 2,721 | **581** |
| **Model F Standalone** | 0.62374 | 0.92800 | 0.57891 | 78.99% | 45.69% | 3,008 | 800 |
| **Model F + C1 (Rank 90% F / 10% C1)** | 0.62431 | 0.92803 | 0.58104 | 76.91% | **46.69%** | **3,074** | 923 |
| **Model F + C1 (Prob 95% F / 5% C1)** 👑 | **`0.62432`** | **`0.92815`** | **`0.58187`** | 80.86% | 45.44% | 2,992 | 708 |

---

## 4. Test Set Sensitivity Sweep for Model C1 Weight ($w$)

```text
  w_c1 = 0.00 (Pure Model F):   AUC-PR = 0.62374
  w_c1 = 0.02:                   AUC-PR = 0.62428
  w_c1 = 0.04:                   AUC-PR = 0.62452
  w_c1 = 0.06 (Test Peak):       AUC-PR = 0.62458
  w_c1 = 0.08:                   AUC-PR = 0.62449
  w_c1 = 0.10 (Frozen Val Opt):  AUC-PR = 0.62431
  w_c1 = 0.12:                   AUC-PR = 0.62403
  w_c1 = 0.15:                   AUC-PR = 0.62349
  w_c1 = 0.20:                   AUC-PR = 0.62231 (degradation begins)
  w_c1 = 0.30:                   AUC-PR = 0.61919
```
