# SanDisk Hackathon — Tree Model Bake-Off & Multi-Model Evaluation

## 1. Executive Summary & Context

A controlled, rigorous tree-model bake-off was conducted using the exact canonical development partition (640 `dev_train` wafers, 160 `dev_val` wafers; eligible dies `old_label == 0` only; target `label`). 

The objective was to determine whether independent tree model families (**XGBoost**, **CatBoost**, **Random Forest**) can:
1. **Outperform LightGBM** on identical feature sets and populations.
2. **Provide complementary predictions** that improve upon the current benchmark champion: the **Model B + Model C1 probability ensemble** ($\text{AUC-PR} = 0.57861$, $\text{ROC-AUC} = 0.89203$, $\text{F1} = 0.55342$).
3. **Validate the multi-resolution A $\to$ B improvement** (+36 engineered block features) independently outside LightGBM.

All experiments maintained strict data hygiene: **zero contamination of the untouched 200-wafer test set**, zero wafer leakage between train and validation, and identical evaluation code.

---

## 2. Experimental Setup & Feature Sets

| Dimension | Specification | Notes |
| :--- | :--- | :--- |
| **Train Population** | 640 wafers, 651,337 eligible dies (`old_label == 0`) | 24,063 Failures (3.694% positive base rate) |
| **Validation Population** | 160 wafers, 137,576 eligible dies (`old_label == 0`) | 5,367 Failures (3.901% positive base rate) |
| **Imbalance Handling** | `scale_pos_weight = 26.067988` (or `class_weight="balanced"`) | Exact negative-to-positive ratio in `dev_train` |
| **Model A Feature Set** | 500 Parametric + 19 Spatial Context = **519 features** | Excludes identifiers and 36 block features |
| **Model B Feature Set** | 500 Parametric + 19 Spatial + 36 Block Context = **555 features** | Includes multi-resolution block statistics |
| **Hardware** | NVIDIA RTX 4500 Ada (24GB VRAM) + 28 CPU threads | GPU hist/Logloss for XGB/CatBoost; CPU for RF |

### Hyperparameters:
- **XGBoost Model A & B**:
  - `tree_method="hist"`, `device="cuda"`, `eval_metric="aucpr"`
  - `learning_rate=0.05`, `max_depth=6`, `min_child_weight=5`
  - `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0`, `reg_lambda=1`
  - `n_estimators=1500`, early stopping rounds = 50 on dev-val AUC-PR
- **CatBoost Model B**:
  - `task_type="GPU"`, `loss_function="Logloss"`, `eval_metric="Logloss"`
  - `iterations=1500`, `learning_rate=0.05`, `depth=7`, `l2_leaf_reg=5.0`
  - Early stopping rounds = 50 on dev-val Logloss
- **Random Forest Model B**:
  - `n_estimators=400`, `max_features="sqrt"`, `min_samples_leaf=3`
  - `class_weight="balanced"`, `n_jobs=28`, `random_state=42`

---

## 3. Comprehensive Master Bake-Off Leaderboard

All metrics evaluated on the exact 137,576 eligible dev-val dies:

| Rank | Model | Architecture | Feature Set | AUC-PR | ROC-AUC | Tuned F1 | Precision | Recall | Specificity | Tuned Thresh | Training Time |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **B + C1 Ensemble** | **Blend (LGB B + CNN C1)** | **555 Feats + Raw 2K Seq** | **0.57861** | **0.89203** | **0.55342** | 0.7672 | 0.4328 | 0.9947 | 0.900 | N/A (blend) |
| 2 | **Model C1** | Triple-Branch 1D CNN + MLP | 555 Feats + Raw 2K Seq | 0.57658 | 0.89131 | 0.55038 | 0.7429 | 0.4371 | 0.9939 | 0.925 | 84.8 s |
| 3 | **LightGBM B** | Gradient Boosted Trees | Model B (555 feats) | 0.55432 | 0.87598 | 0.54045 | 0.8217 | 0.4026 | 0.9965 | 0.815 | ~12.0 s |
| 4 | **CatBoost B** | Symmetric Oblivious Trees | Model B (555 feats) | 0.54920 | **0.87614** | 0.53847 | 0.7324 | 0.4257 | 0.9937 | 0.800 | 27.6 s |
| 5 | **XGBoost B** | Depth-Wise Hist Trees | Model B (555 feats) | 0.54474 | 0.87017 | 0.53637 | 0.7929 | 0.4053 | 0.9957 | 0.765 | 18.8 s |
| 6 | **LightGBM A** | Gradient Boosted Trees | Model A (519 feats) | 0.49366 | 0.83175 | 0.52030 | 0.9419 | 0.3594 | 0.9991 | 0.810 | ~10.0 s |
| 7 | **XGBoost A** | Depth-Wise Hist Trees | Model A (519 feats) | 0.49176 | 0.82825 | 0.51778 | 0.9540 | 0.3553 | 0.9993 | 0.780 | 23.4 s |
| 8 | **Random Forest B** | Bagged Decision Trees | Model B (555 feats) | 0.36977 | 0.83789 | 0.37587 | 0.4340 | 0.3315 | 0.9825 | 0.250 | 204.2 s |

---

## 4. Key Questions & Empirical Answers

### (1) Did XGBoost beat LightGBM?
- **No.** LightGBM B achieved **0.5543 AUC-PR** vs. XGBoost B's **0.5447 AUC-PR** ($\Delta = -0.0096$).
- Similarly on Model A, LightGBM A achieved **0.4937 AUC-PR** vs. XGBoost A's **0.4918 AUC-PR** ($\Delta = -0.0019$).
- LightGBM's leaf-wise (best-first) tree growth strategy captures asymmetric wafer edge and corner defect clusters slightly better than XGBoost's depth-wise splits on this dataset.

### (2) Did the Model A $\to$ Model B Multi-Resolution Improvement Replicate Independently?
- **Yes, decisively!**
  - **XGBoost Model A $\to$ XGBoost Model B**:
    - AUC-PR increased from **0.4918 to 0.5447** ($+0.0529$, **$+10.77\%$ relative lift**).
    - ROC-AUC increased from **0.8282 to 0.8702** ($+0.0419$).
    - Tuned F1 increased from **0.5178 to 0.5364** ($+0.0186$).
  - This independently corroborates LightGBM's earlier A $\to$ B gain ($+12.29\%$), confirming that the 36 engineered block features provide real physical signal rather than framework-specific artifacts.

### (3) Did CatBoost Help?
- **Yes, as a strong second tree alternative.**
  - Standalone AUC-PR: **0.5492** (trailing LightGBM B by only 0.0051).
  - Standalone ROC-AUC: **0.87614** (surpassing LightGBM B's 0.87598).
  - Tuned F1: **0.5385** (close to LightGBM B's 0.5405).
  - Inference speed was blistering on GPU: **3.18 million dies/second** (0.04s for the entire dev-val set).

### (4) Did Random Forest Help?
- **No.** Random Forest achieved **0.3698 AUC-PR** and **0.3759 F1**.
- Under 26:1 class imbalance, unweighted bagged trees suffer from severe probability dilution. Even with `class_weight="balanced"`, lack of iterative residual focusing (boosting) leaves leaf-level probability calibrations poor, producing 2,320 false positives at optimal threshold compared to 469 for LightGBM B.

---

## 5. Prediction Complementarity & Error Diversity Analysis

Examining error overlaps and correlation on the 137,576 dev-val dies:

| Model Pair | Pearson Corr ($r$) | Spearman Rank Corr ($\rho$) | Disagreements | Both Caught | Only X Caught | Only Ref Caught | Both Missed | Unique Defects Caught | Coverage % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost B vs LightGBM B** | 0.9298 | 0.9141 | 677 (0.49%) | 2,065 | 110 | 96 | 3,096 | 2,271 | 42.31% |
| **CatBoost B vs LightGBM B** | 0.9353 | 0.9262 | 872 (0.63%) | 2,099 | 186 | 62 | 3,020 | 2,347 | 43.73% |
| **XGBoost B vs Model C1** | **0.8141** | **0.8154** | 1,089 (0.79%) | 2,102 | 73 | 244 | 2,948 | 2,419 | 45.07% |
| **CatBoost B vs Model C1** | **0.8502** | **0.8327** | 1,156 (0.84%) | 2,162 | 123 | 184 | 2,898 | 2,469 | 46.00% |
| **LightGBM B vs Model C1** | **0.8318** | **0.8311** | 1,044 (0.76%) | 2,096 | 65 | 250 | 2,956 | 2,411 | 44.92% |
| **Random Forest B vs C1** | 0.7009 | 0.6764 | 3,351 (2.44%) | 1,528 | 251 | 818 | 2,770 | 2,597 | 48.39% |

### Key Complementarity Takeaway:
- All three gradient-boosted tree models (LightGBM, XGBoost, CatBoost) are highly correlated with each other ($r > 0.92$).
- However, all tree models exhibit **substantial error diversity relative to Model C1** ($r \approx 0.81–0.85$, with over 1,000 disagreements).
- Model C1 catches ~250 defects that LightGBM B misses, while LightGBM B catches ~65 defects that Model C1 misses.
- CatBoost B catches 123 defects that Model C1 misses.

---

## 6. Targeted Ensemble Sweeps

| Ensemble Combination | Optimal $\alpha_{\text{C1}}$ | Partner Weight | AUC-PR | ROC-AUC | Tuned F1 | Optimal Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model C1 + LightGBM B (B+C1 Benchmark)** | **0.860** | **0.140** | **0.57861** | **0.89202** | **0.55382** | **0.905** |
| Tri-Blend: C1 + LightGBM B + CatBoost B | 0.850 | LGB=0.120, CAT=0.030 | 0.57855 | **0.89207** | 0.55369 | 0.895 |
| Tri-Blend: C1 + LightGBM B + XGBoost B | 0.850 | LGB=0.120, XGB=0.030 | 0.57846 | 0.89201 | 0.55313 | 0.890 |
| Model C1 + CatBoost B | 0.900 | 0.100 | 0.57766 | 0.89185 | 0.55274 | 0.900 |
| Model C1 + XGBoost B | 0.936 | 0.064 | 0.57734 | 0.89155 | 0.55064 | 0.905 |
| Model C1 + Random Forest B | 0.995 | 0.005 | 0.57685 | 0.89132 | 0.55057 | 0.925 |

### Analysis:
- The **B + C1 ensemble (LightGBM B + Model C1)** remains the top performer on AUC-PR (**0.57861**).
- Adding CatBoost B in a tri-blend (`0.85 C1 + 0.12 LGB + 0.03 CAT`) yields a fractional boost in ROC-AUC (**0.89207** vs 0.89202) and an F1 of **0.55369**, but virtually ties B+C1 in AUC-PR while adding architectural complexity.
- Pure tree-tree blending (e.g. LightGBM + XGBoost) yields no gains over LightGBM B alone because the correlation ($r = 0.93$) is too high.

---

## 7. Decision Summary & Clear Recommendations

| Category | Finding / Decision |
| :--- | :--- |
| **CURRENT CHAMPION** | **B + C1 Ensemble (AUC-PR = 0.57861, ROC-AUC = 0.89203, F1 = 0.55342)** |
| **BEST NEW STANDALONE MODEL** | **CatBoost Model B (AUC-PR = 0.54920, ROC-AUC = 0.87614, F1 = 0.53847)** |
| **BEST NEW ENSEMBLE** | **Tri-Blend C1 + LightGBM B + CatBoost B (ROC-AUC = 0.89207, AUC-PR = 0.57855)** |
| **MODELS TO DISCARD** | **Random Forest Model B** (severe degradation under extreme imbalance; discard completely). |
| **TREE SEARCH CONCLUSION** | Additional tabular tree tuning offers rapidly diminishing returns ($\le 0.0001$). LightGBM B remains the definitive tabular tree champion. |
| **WHETHER C2 MULTI-SCALE CNN IS WORTH TESTING** | **YES, STRONGLY RECOMMENDED.** |

### Why Model C2 is the Highest-ROI Next Step:
1. The bake-off demonstrates that all tree models hit an empirical plateau around $\text{AUC-PR} \approx 0.554$ because they operate on summary statistics rather than spatial-temporal waveforms.
2. In contrast, **Model C1 operates directly on the raw 2,000-reading sequence**, achieving $\text{AUC-PR} = 0.5766$ standalone and driving the ensemble to $\mathbf{0.5786}$.
3. Developing **Model C2** (Multi-Scale 1D CNN with dilated receptive fields to capture both localized high-frequency glitches and long-range drift across the 2,000 readings) addresses the primary bottleneck: the representation quality of the raw block readings. Any gain in C2 directly elevates the final ensemble.
