# Model D Evaluation & Controlled Feature Ablation Report

**Date**: September 8, 2026  
**Subject**: Evaluation of Model D (Targeted Hybrid CatBoost) on Canonical Development Split  
**Evaluation Scope**: Strictly Canonical Development Split (640 train wafers, 160 dev-val wafers, 137,576 eligible dies, 5,367 validation positives).  
**Holdout Protection**: **Final 200 test wafers remain COMPLETELY UNTOUCHED.** Zero 5-fold CV executed in this step.

---

## 1. Executive Summary & Core Results

Model D investigates whether targeted, leak-free feature engineering inspired by teammate Adit's branch can improve tabular performance over our canonical **Model B (LightGBM baseline, AUC-PR = 0.5543)** and complement our deep sequence models (**Model C1 CNN, AUC-PR = 0.5766**).

### Winning Model D Configuration:
- **Best Variant**: **D4 (Full Model D)**
- **Total Features**: 595 (555 base + targeted additions)
- **Validation AUC-PR**: **0.5681**
- **Validation ROC-AUC**: **0.8912**
- **Tuned F1-Score**: **0.5467** (Precision: 0.7717, Recall: 0.4233)
- **Defects Caught (TP)**: **2,272 / 5,367** (42.33%) at 672 False Positives
- **Optimal Threshold**: $T^* = 0.9161$

---

## 2. Controlled Feature Ablation Table (D0 through D4)

Every variant was evaluated on the **exact same 137,576 eligible validation dies** using identical CatBoost GPU hyperparameters (`iterations=1200, lr=0.04, depth=7, l2_leaf_reg=6.0, scale_pos_weight=26.07, seed=42`):

| Variant | Features | AUC-PR | ROC-AUC | F1 | Precision | Recall | Specificity | TP | FP | Thresh | Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D0** | 555 | **0.5467** | 0.8739 | 0.5367 | 0.7920 | 0.4058 | 0.9957 | 2,178 | 572 | 0.8176 | 27.6s |
| **D1** | 565 | **0.5668** | 0.8915 | 0.5452 | 0.7381 | 0.4323 | 0.9938 | 2,320 | 823 | 0.8866 | 4.4s |
| **D2** | 572 | **0.5637** | 0.8918 | 0.5444 | 0.7494 | 0.4274 | 0.9942 | 2,294 | 767 | 0.8915 | 4.5s |
| **D3** | 578 | **0.5671** | 0.8917 | 0.5463 | 0.7409 | 0.4326 | 0.9939 | 2,322 | 812 | 0.8866 | 4.4s |
| **D4** | 595 | **0.5681** | 0.8912 | 0.5467 | 0.7717 | 0.4233 | 0.9949 | 2,272 | 672 | 0.9161 | 4.7s |

---

## 3. Comparison Against Canonical Benchmarks

| Model Architecture | Canonical AUC-PR | Canonical ROC-AUC | Canonical F1 | Model D Delta (AUC-PR) | Model D Delta (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model A LightGBM** | 0.4937 | 0.8228 | 0.4984 | **+0.0744** | +0.0483 |
| **Model B LightGBM** | 0.5543 | 0.8760 | 0.5397 | **+0.0138** | +0.0070 |
| **Model C1 CNN** | 0.5766 | 0.8920 | 0.5524 | **-0.0085** | -0.0057 |
| **Model C2 CNN** | 0.5717 | 0.8872 | 0.5480 | **-0.0036** | -0.0013 |
| **Grand Tri-Blend Champion** | 0.5795 | 0.8933 | 0.5553 | **-0.0114** | -0.0086 |

---

## 4. Feature Provenance & Leakage Audit

All additional features obey strict causal and inference constraints:


| Feature Name / Group | Information Used | Inference Available? | Uses `old_label`? | Fitted on `dev_train`? |
| :--- | :--- | :---: | :---: | :---: |
| **Model B Base (555)** | 500 Parametric + 19 Spatial + 36 Block Stats | ✅ Yes | ✅ Yes (pre-test only) | ❌ No (deterministic) |
| **`pca_01` ... `pca_10`** | 500 Parametric Tests | ✅ Yes | ❌ No | ✅ **Yes** (`StandardScaler` + `PCA(10)` on `dev_train`) |
| **`exact_edt_distance`** | Pre-test wafer defect coordinates | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`nearest_defect_cluster_size`** | Pre-test wafer defect 2D connected components | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`nearest_defect_log_cluster_size`** | Log of nearest pre-test cluster size | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`defect_vector_dr`, `defect_vector_dc`** | Relative vector from die to nearest pre-test defect | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`is_defect_neighbor_1hop`, `2hop`** | Discrete spatial adjacency flags (distance $\le 1.5, 2.5$) | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic per wafer) |
| **`block_rolling_mean_350`** | Raw 2,000 block sequence (window 350) | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`max_rolling_mean_350`** | Maximum rolling mean over window 350 | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`min_rolling_mean_350`** | Minimum rolling mean over window 350 | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`max_rolling_std_350`** | Maximum rolling std over window 350 | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`max_rolling_mean_350_start_idx`** | Location index $[0, 1]$ of maximum rolling mean | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`burst_excess_350`** | Excess burst over die global block mean | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`tophat_peak_100`, `200`** | 1D Morphological white top-hat filter peak | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`tophat_energy_100`, `200`** | 1D Morphological white top-hat squared energy | ✅ Yes | ❌ No | ❌ No (per-die deterministic) |
| **`zernike_Z1_neg1` ... `Z4_0`** | Normalized circular die coordinates $(\rho, \phi)$ | ✅ Yes | ❌ No | ❌ No (closed-form formula) |
| **`reticle_pos`, `is_reticle_corner`** | $4\times 4$ stepper exposure grid coordinates | ✅ Yes | ❌ No | ❌ No (closed-form formula) |
| **`inter_pca01_x_roll350`** | Product: `pca_01` $\times \max(0, \text{Roll350} - 100)$ | ✅ Yes | ❌ No | ✅ Indirect (uses `pca_01`) |
| **`inter_pca01_x_cluster_size`** | Product: `pca_01` $\times$ `log_cluster_size` | ✅ Yes | ✅ Yes (`old_label == 1`) | ✅ Indirect (uses `pca_01`) |
| **`inter_pca01_x_edt`** | Ratio: `pca_01` / (`exact_edt_distance` + 0.02) | ✅ Yes | ✅ Yes (`old_label == 1`) | ✅ Indirect (uses `pca_01`) |
| **`inter_pca01_x_edge`** | Product: `pca_01` $\times (1 - \text{edge\_dist})$ | ✅ Yes | ❌ No | ✅ Indirect (uses `pca_01`) |
| **`inter_roll350_x_cluster`** | Product: `burst_350` $\times$ `log_cluster_size` | ✅ Yes | ✅ Yes (`old_label == 1`) | ❌ No (deterministic) |


---

## 5. Feature Importance Analysis (Top 25 Features)

| Rank | Feature Name | CatBoost Gain Importance |
| :---: | :--- | :---: |
| 1 | `pca_01` | 41.7715 |
| 2 | `inter_pca01_x_roll350` | 24.6441 |
| 3 | `wafer_die_count` | 4.5525 |
| 4 | `block_mean_top200` | 2.8253 |
| 5 | `max_rolling_mean_400` | 2.6686 |
| 6 | `inter_pca01_x_edge` | 2.5100 |
| 7 | `wafer_old_fail_rate` | 2.4567 |
| 8 | `wafer_old_fail_count` | 2.1558 |
| 9 | `max_rolling_mean_200` | 2.0959 |
| 10 | `inter_pca01_x_cluster_size` | 1.6410 |
| 11 | `max_rolling_mean_350` | 1.2598 |
| 12 | `block_q75` | 1.2220 |
| 13 | `block_mean` | 0.7344 |
| 14 | `block_q95` | 0.5751 |
| 15 | `inter_pca01_x_edt` | 0.5621 |
| 16 | `zernike_Z2_0` | 0.4089 |
| 17 | `max_rolling_mean_100_start_idx` | 0.3898 |
| 18 | `block_q50` | 0.3463 |
| 19 | `block_mean_top100` | 0.2802 |
| 20 | `max_rolling_mean_100` | 0.2244 |
| 21 | `radius` | 0.1959 |
| 22 | `block_mean_top50` | 0.1790 |
| 23 | `distance_to_edge` | 0.1756 |
| 24 | `inter_roll350_x_cluster` | 0.1560 |
| 25 | `feature_405` | 0.1481 |

---

## 6. Analysis: Which Feature Groups Actually Helped?

1. **Baseline CatBoost (D0 vs. LightGBM B)**:
   - D0 established the CatBoost baseline on canonical 555 features at AUC-PR = **0.5467** (compared to LightGBM B at 0.5543).
2. **Impact of Parametric PCA (D1 vs. D0)**:
   - Adding 10 PCA components shifted AUC-PR from **0.5467 to 0.5668** (Delta: **+0.0201**).
3. **Impact of Wafer Cluster Topology & EDT (D2 vs. D1)**:
   - Adding cluster connected component sizes and exact Euclidean distance transforms shifted AUC-PR from **0.5668 to 0.5637** (Delta: **-0.0031**).
4. **Impact of W=350 Block Features (D3 vs. D2)**:
   - Adding per-die W=350 rolling mean and std features shifted AUC-PR from **0.5637 to 0.5671** (Delta: **+0.0034**).
5. **Impact of Full Model D (D4 vs. D3)**:
   - Adding top-hat filters, geometric Zernike coordinates, and cross-resolution bilinear interactions shifted AUC-PR from **0.5671 to 0.5681** (Delta: **+0.0009**).

---

## 6.1 Complementarity & Two-Stage Blending Analysis

To assess whether Model D4 adds orthogonal information to our existing models, we evaluated prediction correlations and two-stage ensembling against our canonical models on `dev_val`:

### Prediction Correlation Matrix:
| Correlation ($r$) | Grand Tri-Blend Champion | Model C1 CNN | Model B LightGBM | Model D4 CatBoost |
| :--- | :---: | :---: | :---: | :---: |
| **Grand Tri-Blend Champion** | 1.0000 | 0.9632 | 0.9415 | **0.9296** |
| **Model C1 CNN** | 0.9632 | 1.0000 | 0.8876 | **0.8987** |
| **Model B LightGBM** | 0.9415 | 0.8876 | 1.0000 | **0.8964** |
| **Model D4 CatBoost** | **0.9296** | **0.8987** | **0.8964** | 1.0000 |

### Two-Stage Blend Evaluation:
$$P_{\text{Blend}}(\alpha) = \alpha \cdot P_{\text{Champion}} + (1 - \alpha) \cdot P_{\text{Model D4}}$$

| Blend Weight $\alpha$ (Champion) | Model D4 Weight ($1-\alpha$) | Blend Validation AUC-PR | Delta vs. Frozen Champion |
| :---: | :---: | :---: | :---: |
| 1.00 (Champion Alone) | 0.00 | 0.57945 | +0.00000 |
| 0.95 | 0.05 | 0.58013 | +0.00068 |
| 0.90 | 0.10 | 0.58065 | +0.00120 |
| 0.85 | 0.15 | 0.58102 | +0.00157 |
| 0.80 | 0.20 | 0.58126 | +0.00181 |
| 0.75 | 0.25 | 0.58140 | +0.00195 |
| **0.70 (Optimal)** | **0.30** | **0.58141** | **+0.00195** |
| 0.65 | 0.35 | 0.58131 | +0.00185 |
| 0.60 | 0.40 | 0.58110 | +0.00165 |
| 0.50 | 0.50 | 0.58039 | +0.00094 |
| 0.00 (Model D4 Alone) | 1.00 | 0.56807 | -0.01138 |

**Key Finding**:
Although Model D4 alone (0.5681) does not exceed Model C1 CNN (0.5766), blending $30\%$ Model D4 with $70\%$ Grand Tri-Blend achieves **0.58141 AUC-PR** on the canonical validation set—delivering a **$+0.00195$ lift** over the champion. This proves Model D's feature-engineered tabular predictions are genuinely complementary to our sequence CNNs.

---

## 7. Decisive Recommendation

### Status: **COMPLEMENTARY TABULAR UPGRADE — WORTH CONDITIONAL 5-FOLD PROMOTION**

Model D4 achieved **0.5681 AUC-PR** standalone (+0.0138 lift over Model B LightGBM at 0.5543), driven overwhelmingly by `pca_01` (41.77% gain) and `inter_pca01_x_roll350` (24.64% gain). When blended with our current Grand Tri-Blend champion ($70\%$ Champion + $30\%$ Model D4), canonical validation AUC-PR improves from **0.57945 to 0.58141 (+0.00195 lift)**.

> [!IMPORTANT]
> **Next Steps**:
> In strict adherence to your instructions:
> 1. **Zero 5-Fold CV has been run.**
> 2. **The current champion Grand Tri-Blend remains completely untouched.**
> 3. **The final 200-wafer test set remains completely untouched.**
> 4. Execution has stopped to present these findings for your strategic decision.
